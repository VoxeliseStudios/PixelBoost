import SwiftUI

/// Everything about an upscale, top to bottom in the order people decide:
/// how big, which look, how careful. Everything runs on this iPhone.
struct UpscaleSheet: View {
    @EnvironmentObject var app: AppModel
    let onDevice: () -> Void
    let choice = UpscaleChoice()
    @State private var autoPick: String?
    @State private var analysis: PhotoAnalysis?
    /// The manual choices with their Auto fields filled in for this photo.
    @State private var manual: PhotoAnalysis?
    @State private var analyzing = false
    @State private var showWhy = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Upscale").font(.title.weight(.bold))
                    Spacer()
                    Button { dismiss() } label: { Image(systemName: "xmark").font(.headline).foregroundStyle(Theme.dim) }
                }
                deviceCard
                smartCard
                if !choice.smart { manualCards }
                PrimaryButton(title: "Upscale on this iPhone", icon: "wand.and.stars") { onDevice() }
                .padding(.top, 4)
            }
            .padding(18)
        }
        .task {
            if let url = app.current {
                autoPick = await Task.detached { try? Core.engine.autoModel(inputPath: url.path) }.value
            }
        }
        .task(id: "\(choice.maxWait)|\(app.current?.path ?? "")") {
            guard let url = app.current else { return }
            analyzing = true
            analysis = await choice.analysis(for: url)
            analyzing = false
        }
        .task(id: "\(choice.smart)|\(choice.maxWait)|\(app.current?.path ?? "")|\(choice.model)|\(choice.scale)|\(choice.fidelity)|\(choice.quality)|\(choice.detailAuto)|\(choice.detail)|\(choice.adaptive)|\(choice.realism)") {
            guard !choice.smart, let url = app.current else { manual = nil; return }
            manual = await choice.manualAnalysis(for: url)
        }
    }

    // MARK: smart

    private var smartCard: some View {
        Card {
            HStack(spacing: 10) {
                Image(systemName: "sparkles").font(.title3).foregroundStyle(Theme.energy)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Smart settings").font(.subheadline.weight(.bold))
                    Text(choice.smart ? "Reads this photo and picks everything" : "Off — using your settings below")
                        .font(.caption).foregroundStyle(Theme.dim)
                }
                Spacer()
                Toggle("", isOn: choice.$smart).labelsHidden().tint(Theme.amber)
            }
            if choice.smart {
                if let a = analysis {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(a.summary).font(.headline)
                        Text(smartLine(a)).font(.caption.monospacedDigit()).foregroundStyle(Theme.dim)
                        HStack(spacing: 6) {
                            meter("Noise", min(Double(a.noise) / 12, 1))
                            meter("Blocks", Double(a.blocking))
                            meter("Sharp", min(max((Double(a.sharpness) - 0.4) / 0.45, 0), 1))
                        }
                        Button { withAnimation(.snappy) { showWhy.toggle() } } label: {
                            Label(showWhy ? "Hide why" : "Why these settings", systemImage: showWhy ? "chevron.up" : "chevron.down")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.plain).foregroundStyle(Theme.amber)
                        if showWhy {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(a.reasons, id: \.self) { r in
                                    HStack(alignment: .top, spacing: 6) {
                                        Text("•").foregroundStyle(Theme.amber)
                                        Text(r).font(.caption)
                                    }
                                }
                            }
                        }
                        Text("Longest wait on this iPhone").font(.caption).foregroundStyle(Theme.dim).padding(.top, 2)
                        ChipPicker(options: [(60.0, "1 min"), (180.0, "3 min"), (600.0, "10 min"), (0.0, "No limit")], selection: choice.$maxWait)
                        Button("Customize these settings") { choice.adopt(a) }
                            .font(.caption.weight(.semibold)).foregroundStyle(Theme.dim).padding(.top, 2)
                    }
                } else if analyzing {
                    HStack(spacing: 8) { ProgressView(); Text("Looking at the photo…").font(.caption).foregroundStyle(Theme.dim) }
                } else {
                    Text("Pick a photo first.").font(.caption).foregroundStyle(Theme.dim)
                }
            }
        }
    }

    private func smartLine(_ a: PhotoAnalysis) -> String {
        let s = a.settings
        let w = Int(Float(a.width) * s.scale), h = Int(Float(a.height) * s.scale)
        var parts = ["\(Int(s.scale))× → \(w) × \(h)", UpscaleChoice.name(s.model), s.fidelity.capitalized, "detail \(Int(s.detail * 100))%"]
        if s.denoise > 0 { parts.append("denoise \(Int(s.denoise * 100))%") }
        if s.model == "photo", s.adaptive { parts.append("Adaptive") }
        if s.model == "photo", s.realism < 0.995 { parts.append("realism \(Int(s.realism * 100))%") }
        if s.ensemble > 1 { parts.append(s.ensemble >= 8 ? "Max" : "High") }
        if s.prescale < 0.99 { parts.append("works on the real \(Int(Float(a.width) * s.prescale)) × \(Int(Float(a.height) * s.prescale))") }
        if a.estSeconds > 0 { parts.append("≈ \(formatDuration(Double(a.estSeconds)))") }
        return parts.joined(separator: " · ")
    }

    private func meter(_ label: String, _ v: Double) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(Theme.dim)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.card2)
                    Capsule().fill(Theme.energy).frame(width: max(4, g.size.width * v))
                }
            }
            .frame(height: 5)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: manual

    @ViewBuilder
    private var manualCards: some View {
                Card(title: "Size", trailing: outputText) {
                    ChipPicker(options: [(2.0, "2×"), (3.0, "3×"), (4.0, "4×"), (6.0, "6×"), (8.0, "8×")], selection: choice.$scale)
                }
                manualSummary
                Card(title: "Look") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 96), spacing: 8)], spacing: 8) {
                        ForEach(UpscaleChoice.models) { m in
                            modelTile(m)
                        }
                    }
                    Text(modelBlurb).font(.caption).foregroundStyle(Theme.dim)
                }
                Card(title: "Quality") {
                    ChipPicker(options: [(0, "Auto"), (1, "Standard"), (2, "High · 2 passes"), (8, "Max · 8 passes")], selection: choice.$quality)
                    if choice.quality == 0, let m = manual {
                        autoNote("Auto: \(m.settings.ensemble >= 8 ? "Max · 8 passes" : m.settings.ensemble > 1 ? "High · \(m.settings.ensemble) passes" : "Standard") — what fits your longest wait")
                    }
                }
                Card(title: "Clarity") {
                    ChipPicker(options: [("auto", "Auto"), ("off", "Raw AI"), ("natural", "Natural"), ("faithful", "Faithful")], selection: choice.$fidelity)
                    Text(fidelityBlurb).font(.caption).foregroundStyle(Theme.dim)
                    if choice.fidelity == "auto", let m = manual { autoNote("Auto: \(m.settings.fidelity.capitalized) for this photo") }
                    Toggle(isOn: choice.$detailAuto) {
                        Text("Auto detail").font(.subheadline)
                    }
                    .tint(Theme.amber)
                    if choice.detailAuto {
                        if let m = manual { autoNote("Auto: detail \(Int(m.settings.detail * 100))% for this photo") }
                    } else {
                        SliderRow(label: "Detail", value: choice.$detail)
                    }
                    if choice.model == "fast" { SliderRow(label: "Denoise", value: choice.$denoise) }
                }
                if choice.model == "photo" {
                    Card(title: "Texture") {
                        Toggle(isOn: choice.$adaptive) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Adaptive").font(.subheadline.weight(.medium))
                                Text("Rebuilds texture only where your photo has it — skin, sky and walls stay clean. About 2× the work.")
                                    .font(.caption).foregroundStyle(Theme.dim)
                            }
                        }
                        SliderRow(label: "Realism", value: choice.$realism)
                        Text(choice.adaptive ? "How far textured areas go towards the full AI texture." : "Blends the Photo network towards Clean: lower is calmer, colours stay locked to the original.")
                            .font(.caption).foregroundStyle(Theme.dim)
                    }
                }
    }

    /// What Auto decided for this photo, plus what the analysis saw.
    private func autoNote(_ text: String) -> some View {
        Label(text, systemImage: "sparkles").font(.caption).foregroundStyle(Theme.amber)
    }

    @ViewBuilder
    private var manualSummary: some View {
        if let m = manual {
            VStack(alignment: .leading, spacing: 4) {
                Text("Looked at this photo: \(m.kind == "anime" ? "drawn art" : "photo / render") · \(m.width) × \(m.height)\(m.realResolution < 0.95 ? " · real detail \(Int(Float(m.width) * m.realResolution)) × \(Int(Float(m.height) * m.realResolution))" : "")")
                    .font(.caption).foregroundStyle(Theme.dim)
                if choice.model == "auto" { autoNote("Auto look: \(UpscaleChoice.name(m.settings.model))\(m.settings.adaptive ? " · Adaptive" : "")") }
            }
        }
    }

    // MARK: where

    private var deviceCard: some View {
        HStack(spacing: 12) {
            Image(systemName: "iphone").font(.title2).foregroundStyle(Theme.energy)
            VStack(alignment: .leading, spacing: 2) {
                Text("On this iPhone").font(.subheadline.weight(.bold))
                Text("Your photo never leaves your phone").font(.caption2).foregroundStyle(Theme.dim)
            }
            Spacer()
            if let e = deviceEstimate {
                Text("≈ \(formatDuration(e))").font(.caption.monospacedDigit()).foregroundStyle(.white)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Theme.line))
    }

    private func modelTile(_ m: UpscaleChoice.Option) -> some View {
        let on = choice.model == m.id
        return Button { choice.model = m.id } label: {
            VStack(spacing: 6) {
                Image(systemName: m.icon).font(.title3)
                Text(m.name).font(.caption.weight(.medium)).lineLimit(1).minimumScaleFactor(0.8)
            }
            .frame(maxWidth: .infinity, minHeight: 66)
            .background(on ? AnyShapeStyle(Theme.amber.opacity(0.18)) : AnyShapeStyle(Theme.card2), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(on ? Theme.amber : .clear, lineWidth: 1.5))
        }
        .buttonStyle(.plain)
    }

    // MARK: text + estimates

    private var effectiveModel: String { choice.model == "auto" ? (manual?.settings.model ?? autoPick ?? "photo") : choice.model }

    private var outputText: String? {
        guard let i = app.info else { return nil }
        let w = Int(Double(i.width) * choice.scale), h = Int(Double(i.height) * choice.scale)
        let mp = Double(w) * Double(h) / 1e6
        return "\(w) × \(h) · \(String(format: mp >= 10 ? "%.0f" : "%.1f", mp)) MP"
    }

    private func gflop(_ model: String) -> Double? {
        guard let i = app.info else { return nil }
        let p = Core.engine.plan(width: i.width, height: i.height, model: model, scale: Float(choice.scale), tile: 128)
        let twoPass = choice.scale > 6 ? 17.0 : 1.0
        let fused = model == "photo" && choice.adaptive ? 2.0 : 1.0
        return p.gflop * Double(max(choice.quality, 1)) * twoPass * fused
    }

    private var deviceEstimate: Double? {
        if choice.smart, let a = analysis, a.estSeconds > 0 { return Double(a.estSeconds) }
        if choice.smart, let a = analysis, let i = app.info {
            let s = a.settings
            let p = Core.engine.plan(width: i.width, height: i.height, model: s.model, scale: s.scale, tile: 128)
            return p.gflop * Double(max(s.ensemble, 1)) * (s.scale > 6 ? 17 : 1) * (s.adaptive ? 2 : 1) / max(choice.gflops, 1)
        }
        if !choice.smart, let m = manual, m.estSeconds > 0 { return Double(m.estSeconds) }
        return gflop(effectiveModel).map { $0 / max(choice.gflops, 1) }
    }

    private var modelBlurb: String {
        switch choice.model {
        case "auto": return "Picks per photo — this one looks like \(UpscaleChoice.name(effectiveModel))."
        case "anime": return "Drawn art, anime, cel shading: crisp lines, flat colour stays flat."
        case "photo": return "Photos, game screenshots and 3D renders: rebuilds real texture."
        case "clean": return "No invented texture — best for faces."
        case "fast": return "Quick everyday upscale with a denoise slider."
        case "lineart": return "Anime video frames, line art, compressed screenshots. Very fast."
        default: return "Real 2× network for photos."
        }
    }

    private var fidelityBlurb: String {
        switch choice.fidelity {
        case "auto": return "Picked from the photo: Natural for noisy or compressed pictures, Faithful for portraits and clean drawn art."
        case "off": return "The network's output as-is."
        case "faithful": return "Every output pixel pulled to agree with your photo as closely as possible."
        default: return "Keeps the AI's sharp edges, restores colour and shading it flattened, removes halos."
        }
    }
}
