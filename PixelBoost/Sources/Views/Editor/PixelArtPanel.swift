import SwiftUI

/// Pixel Art controls. Styles, palettes, option lists and the op JSON all
/// come from the Rust core (`pixel*` calls); this view only binds controls
/// to a `PixelArtSettings` record.
struct PixelArtControls: View {
    @Binding var settings: PixelArtSettings
    @Binding var style: String

    private static let styles = Core.engine.pixelStyles()
    private static let palettes = Core.engine.pixelPalettes()
    private static let dithers = Core.engine.pixelDithers().map { ($0.id, $0.name) }
    private static let outlines = Core.engine.pixelOutlines().map { ($0.id, $0.name) }

    static var defaultStyle: PixelStyle? { styles.first }

    var body: some View {
        VStack(spacing: 10) {
            ChipPicker(options: Self.styles.map { ($0.id, $0.name) }, selection: styleBinding)
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
        Self.palettes.first(where: { $0.id == settings.palette })?.adaptive ?? true
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
                ForEach(Self.palettes, id: \.id) { p in
                    let on = settings.palette == p.id
                    Button { settings.palette = p.id } label: {
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
                }
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
