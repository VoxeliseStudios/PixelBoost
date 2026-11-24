import SwiftUI

/// The upscale choices, persisted. Shared by the sheet and batch.
struct UpscaleChoice: DynamicProperty {
    @AppStorage("up.model") var model = "auto"
    @AppStorage("up.scale") var scale = 4.0
    /// "auto" = chosen from the photo.
    @AppStorage("up.fidelityA") var fidelity = "auto"
    @AppStorage("up.detail") var detail = 0.35
    /// Detail chosen from the photo.
    @AppStorage("up.detailAuto") var detailAuto = true
    /// Self-ensemble passes; 0 = chosen from the photo and the time budget.
    @AppStorage("up.qualityA") var quality = 0
    @AppStorage("up.denoise") var denoise = 0.0
    /// Photo model texture strength (1 = full GAN texture).
    @AppStorage("up.realism") var realism = 1.0
    /// Photo model: also run Clean and mix per pixel by the photo's texture.
    @AppStorage("up.adaptive") var adaptive = false
    /// Learned speed of this phone (GFLOP/s).
    @AppStorage("up.gflops") var gflops = 40.0
    /// Let the Rust analyzer choose every setting per photo.
    @AppStorage("up.smart") var smart = true
    /// Longest on-device run Smart may plan for (seconds, 0 = no limit).
    @AppStorage("up.maxWait") var maxWait = 180.0

    var settings: UpscaleSettings {
        UpscaleSettings(model: model, scale: Float(scale), fidelity: fidelity, detail: detailAuto ? -1 : Float(detail),
                        denoise: Float(denoise), ensemble: UInt32(max(quality, 0)), tile: 128, maxOutputMp: 120,
                        realism: Float(realism), adaptive: adaptive, prescale: 0, dehalo: -1)
    }

    /// Smart analysis of a photo for this phone (cached per photo).
    func analysis(for url: URL) async -> PhotoAnalysis? {
        await SmartCache.shared.analysis(for: url, gflops: gflops, maxWait: maxWait)
    }

    /// The manual settings with everything left on Auto decided from this
    /// photo (real resolution, content, noise, blocks, softness, budget).
    func manualAnalysis(for url: URL) async -> PhotoAnalysis? {
        await SmartCache.shared.resolve(url, settings: settings, gflops: gflops, maxWait: maxWait)
    }

    /// Settings to run with: Smart's for this photo, else the manual ones
    /// with their Auto fields filled in from the photo.
    func resolved(for url: URL) async -> UpscaleSettings {
        if smart, let a = await analysis(for: url) { return a.settings }
        if let a = await manualAnalysis(for: url) { return a.settings }
        return settings
    }

    /// Copy Smart's choices into the manual controls (to tweak them).
    func adopt(_ a: PhotoAnalysis) {
        let s = a.settings
        model = s.model
        scale = Double(s.scale)
        fidelity = s.fidelity
        detail = Double(s.detail)
        detailAuto = false
        denoise = Double(s.denoise)
        quality = Int(s.ensemble)
        realism = Double(s.realism)
        adaptive = s.adaptive
        smart = false
    }

    struct Option: Identifiable {
        let id: String
        let name: String
        let icon: String
    }

    static let models: [Option] = [
        Option(id: "auto", name: "Auto", icon: "wand.and.stars"),
        Option(id: "anime", name: "Anime & Art", icon: "paintpalette"),
        Option(id: "photo", name: "Photo & Render", icon: "camera"),
        Option(id: "clean", name: "Clean", icon: "face.smiling"),
        Option(id: "fast", name: "Fast", icon: "hare"),
        Option(id: "lineart", name: "Line Art", icon: "scribble.variable"),
        Option(id: "native2x", name: "Native 2×", icon: "2.square"),
    ]

    static func name(_ id: String) -> String { models.first { $0.id == id }?.name ?? id }
}

/// Smart analyses, one per (photo, budget). The analysis reads the
/// whole photo, so it runs once and off the main thread.
@MainActor
final class SmartCache {
    static let shared = SmartCache()
    private var done: [String: PhotoAnalysis] = [:]
    private var running: [String: Task<PhotoAnalysis?, Never>] = [:]

    func analysis(for url: URL, gflops: Double, maxWait: Double) async -> PhotoAnalysis? {
        let key = "\(url.path)|\(Int(gflops / 5))|\(Int(maxWait))"
        if let a = done[key] { return a }
        if let t = running[key] { return await t.value }
        let path = url.path
        let budget = Float(maxWait <= 0 ? 1e9 : maxWait)
        let t = Task.detached(priority: .userInitiated) { () -> PhotoAnalysis? in
            try? Core.engine.analyze(inputPath: path, deviceGflops: Float(gflops), maxSeconds: budget, maxOutputMp: 120)
        }
        running[key] = t
        let a = await t.value
        running[key] = nil
        if let a {
            if done.count > 40 { done.removeAll() }
            done[key] = a
        }
        return a
    }

    /// Manual settings with their Auto fields resolved (cached per choice).
    func resolve(_ url: URL, settings s: UpscaleSettings, gflops: Double, maxWait: Double) async -> PhotoAnalysis? {
        let key = "r|\(url.path)|\(Int(gflops / 5))|\(Int(maxWait))|\(s.model)|\(s.scale)|\(s.fidelity)|\(s.detail)|\(s.ensemble)|\(s.denoise)|\(s.realism)|\(s.adaptive)"
        if let a = done[key] { return a }
        if let t = running[key] { return await t.value }
        let path = url.path
        let budget = Float(maxWait <= 0 ? 1e9 : maxWait)
        let t = Task.detached(priority: .userInitiated) { () -> PhotoAnalysis? in
            try? Core.engine.resolve(inputPath: path, settings: s, deviceGflops: Float(gflops), maxSeconds: budget)
        }
        running[key] = t
        let a = await t.value
        running[key] = nil
        if let a {
            if done.count > 40 { done.removeAll() }
            done[key] = a
        }
        return a
    }
}

func formatDuration(_ s: Double) -> String {
    if s < 1 { return "<1s" }
    if s < 90 { return "\(Int(s.rounded()))s" }
    if s < 3600 { return "\(Int(s / 60))m \(Int(s.truncatingRemainder(dividingBy: 60)))s" }
    return "\(Int(s / 3600))h \(Int(s.truncatingRemainder(dividingBy: 3600) / 60))m"
}

func formatBytes(_ b: UInt64) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file)
}
