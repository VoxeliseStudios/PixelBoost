import SwiftUI
import UIKit

/// Pixel Art controls. Styles, palettes, option lists and the op JSON all
/// come from the Rust core (`pixel*` calls); this view only binds controls
/// to a `PixelArtSettings` record.
struct PixelArtControls: View {
    @Binding var settings: PixelArtSettings
    @Binding var style: String
    /// Preview of the photo (seed colours for a new palette).
    var base: RgbaImage? = nil
    @State private var mine: [PixelPalette] = Core.palettes.list()
    @State private var editing: PixelPalette?

    private static let styles = Core.engine.pixelStyles()
    private static let palettes = Core.engine.pixelPalettes()
    private static let dithers = Core.engine.pixelDithers().map { ($0.id, $0.name) }
    private static let outlines = Core.engine.pixelOutlines().map { ($0.id, $0.name) }

    static var defaultStyle: PixelStyle? { styles.first }

    var body: some View {
        VStack(spacing: 10) {
            ChipPicker(options: Self.styles.map { ($0.id, $0.name) }, selection: styleBinding)
                .sheet(item: $editing) { p in
                    PaletteEditor(palette: p, base: base) { saved in
                        mine = Core.palettes.list()
                        if let saved { select(saved) } else if settings.palette == p.id { settings.custom = []; settings.palette = "auto" }
                    }
                    .presentationDetents([.medium, .large])
                    .presentationBackground(Theme.ink)
                }
            if let s = Self.styles.first(where: { $0.id == style }) {
                Text(s.blurb).font(.caption).foregroundStyle(Theme.dim)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            SliderRow(label: "Pixels across", value: int(\.width), range: 32...320, format: { "\(Int($0))" })
            paletteRow
            if adaptive {
                SliderRow(label: "Colours", value: int(\.colors), range: 4...64, format: { "\(Int($0))" })
            }
            ChipPicker(options: Self.dithers, selection: $settings.dither)
            if settings.dither != "none" {
                SliderRow(label: "Dither amount", value: num(\.ditherAmount))
            }
            ChipPicker(options: Self.outlines, selection: $settings.outline)
            SliderRow(label: "Line detail", value: num(\.lines))
            SliderRow(label: "Rim light", value: num(\.rim))
            if settings.rim > 0.01 {
                SliderRow(label: "Light from", value: num(\.light), range: 0...360, format: { "\(Int($0))°" })
            }
            SliderRow(label: "Hue shift", value: num(\.hueShift))
            SliderRow(label: "Punch", value: num(\.punch), range: -1...1)
            SliderRow(label: "Simplify", value: num(\.smooth))
            SliderRow(label: "Sparkles", value: num(\.sparkle))
            Toggle("Clean up stray pixels", isOn: $settings.cleanup)
            Toggle("Keep pixel size (tiny image)", isOn: $settings.native)
        }
    }

    private var adaptive: Bool {
        if !settings.custom.isEmpty { return false }
        return Self.palettes.first(where: { $0.id == settings.palette })?.adaptive ?? true
    }

    private func select(_ p: PixelPalette) {
        settings.palette = p.id
        settings.custom = p.swatches
    }

    private var styleBinding: Binding<String> {
        Binding(get: { style }, set: { id in
            style = id
            if let s = Self.styles.first(where: { $0.id == id }) { settings = s.settings }
        })
    }

    private var paletteRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Self.palettes + mine, id: \.id) { p in
                    let on = settings.palette == p.id
                    let custom = p.id.hasPrefix("custom:")
                    Button { if custom { select(p) } else { settings.palette = p.id; settings.custom = [] } } label: {
                        VStack(spacing: 4) {
                            HStack(spacing: 0) {
                                if p.swatches.isEmpty {
                                    Image(systemName: p.id == "auto" ? "wand.and.stars" : "paintpalette")
                                        .font(.caption).frame(height: 12)
                                } else {
                                    ForEach(Array(p.swatches.prefix(12).enumerated()), id: \.offset) { _, v in
                                        Rectangle().fill(Self.rgb(v)).frame(width: 6, height: 12)
                                    }
                                }
                            }
                            Text(p.name).font(.caption2.weight(on ? .semibold : .regular))
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(on ? AnyShapeStyle(Theme.energy) : AnyShapeStyle(Theme.card2),
                                    in: RoundedRectangle(cornerRadius: 10))
                        .foregroundStyle(on ? Color.black : Color.white)
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        if custom {
                            Button("Edit") { editing = p }
                            Button("Delete", role: .destructive) { try? Core.palettes.delete(id: p.id); mine = Core.palettes.list() }
                        }
                    }
                }
                Button { editing = PixelPalette(id: "", name: "My palette", swatches: [], adaptive: false) } label: {
                    Label("New", systemImage: "plus").font(.caption2.weight(.semibold))
                        .padding(.horizontal, 10).padding(.vertical, 12)
                        .background(Theme.card2, in: RoundedRectangle(cornerRadius: 10))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func num(_ kp: WritableKeyPath<PixelArtSettings, Float>) -> Binding<Double> {
        Binding(get: { Double(settings[keyPath: kp]) }, set: { settings[keyPath: kp] = Float($0) })
    }

    private func int(_ kp: WritableKeyPath<PixelArtSettings, UInt32>) -> Binding<Double> {
        Binding(get: { Double(settings[keyPath: kp]) }, set: { settings[keyPath: kp] = UInt32(max(0, $0.rounded())) })
    }

    private static func rgb(_ v: UInt32) -> Color {
        Color(red: Double((v >> 16) & 255) / 255, green: Double((v >> 8) & 255) / 255, blue: Double(v & 255) / 255)
    }
}


extension PixelPalette: Identifiable {}

/// Custom palette editor: colour wells, add / remove, seed from the photo.
/// Storage and colour extraction are in Rust (PaletteStore, extractPalette).
struct PaletteEditor: View {
    let palette: PixelPalette
    let base: RgbaImage?
    let done: (PixelPalette?) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var colors: [Color] = []
    @State private var seedCount = 12.0

    var body: some View {
        NavigationStack {
            Form {
                TextField("Name", text: $name)
                Section("Colours") {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 6), spacing: 10) {
                        ForEach(colors.indices, id: \.self) { i in
                            ColorPicker("", selection: $colors[i], supportsOpacity: false).labelsHidden()
                                .contextMenu { Button("Remove", role: .destructive) { colors.remove(at: i) } }
                        }
                        Button { colors.append(colors.last ?? .gray) } label: { Image(systemName: "plus.circle.fill").font(.title2) }
                            .disabled(colors.count >= 64)
                    }
                    Text("Hold a colour to remove it.").font(.caption2).foregroundStyle(Theme.dim)
                }
                if let base {
                    Section("From this photo") {
                        SliderRow(label: "Colours", value: $seedCount, range: 2...32, format: { "\(Int($0))" })
                        Button("Take colours from the photo") {
                            let v = (try? Core.engine.extractPalette(image: base, colors: UInt32(seedCount))) ?? []
                            if !v.isEmpty { colors = v.map(Self.color) }
                        }
                    }
                }
                if !palette.id.isEmpty {
                    Button("Delete palette", role: .destructive) { try? Core.palettes.delete(id: palette.id); done(nil); dismiss() }
                }
            }
            .scrollContentBackground(.hidden)
            .navigationTitle(palette.id.isEmpty ? "New palette" : "Edit palette")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        let p = try? Core.palettes.save(id: palette.id, name: name, swatches: colors.map(Self.hex))
                        done(p)
                        dismiss()
                    }
                    .disabled(colors.count < 2)
                }
            }
            .onAppear {
                name = palette.name
                colors = palette.swatches.map(Self.color)
                if colors.isEmpty { colors = [.black, .white] }
            }
        }
    }

    static func color(_ v: UInt32) -> Color {
        Color(red: Double((v >> 16) & 255) / 255, green: Double((v >> 8) & 255) / 255, blue: Double(v & 255) / 255)
    }

    static func hex(_ c: Color) -> UInt32 {
        var (r, g, b, a): (CGFloat, CGFloat, CGFloat, CGFloat) = (0, 0, 0, 0)
        UIColor(c).getRed(&r, green: &g, blue: &b, alpha: &a)
        let q = { (v: CGFloat) in UInt32(max(0, min(255, (v * 255).rounded()))) }
        return (q(r) << 16) | (q(g) << 8) | q(b)
    }
}
