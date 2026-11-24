import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject var app: AppModel
    @AppStorage("pb.server") private var server = Prefs.defaultServer
    @AppStorage("pb.apiKey") private var apiKey = ""
    @AppStorage("pb.telemetry") private var telemetry = true
    @AppStorage("pb.replaceMethod") private var replaceMethod = PhotoLibrary.Method.edit.rawValue
    @AppStorage("pb.compute") private var compute = "auto"
    @State private var computeInfo: ComputeInfo?
    @State private var health: String?
    @State private var exports: [StoredImage] = []
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Run networks on", selection: $compute) {
                        Text("Auto").tag("auto")
                        Text("GPU").tag("gpu")
                        Text("CPU").tag("cpu")
                    }
                    .pickerStyle(.segmented)
                    if let c = computeInfo {
                        LabeledContent("GPU", value: c.gpu ?? "Not available")
                        LabeledContent("CPU", value: c.cpuKernel)
                        if c.gpuGflops > 0 {
                            LabeledContent("Measured", value: String(format: "GPU %.0f · CPU %.0f GFLOP/s", c.gpuGflops, c.cpuGflops))
                        }
                    } else {
                        HStack { ProgressView(); Text("Measuring this iPhone…").font(.caption).foregroundStyle(Theme.dim) }
                    }
                } header: { Text("Compute") } footer: {
                    Text(computeFooter)
                }
                Section("Engine") {
                    LabeledContent("Inference", value: "PixelBoost NN (Rust)")
                    LabeledContent("Kernel", value: Core.engine.kernel())
                    ForEach(app.models, id: \.id) { m in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(m.name)  ·  \(m.scale)×").font(.subheadline)
                            Text(m.bestFor).font(.caption).foregroundStyle(Theme.dim)
                        }
                    }
                }
                DownloadsSection()
                Section {
                    Toggle("Help tune PixelBoost", isOn: $telemetry)
                    LabeledContent("Waiting to send", value: "\(Core.telemetry.pending()) events")
                } header: { Text("Telemetry") } footer: {
                    Text("Sends measurements only — timings, chosen settings, what the engine measured (noise, halos, real resolution…), GPU speed and whether you kept a result. Never pictures, file names or text. Examples are sent only when you tap “Send as an example” on a result. Turning this off deletes anything not yet sent.")
                }
                Section {
                    Picker("Replace method", selection: $replaceMethod) {
                        ForEach(PhotoLibrary.Method.allCases) { m in Text(m.title).tag(m.rawValue) }
                    }
                } header: { Text("Replacing originals in Photos") } footer: {
                    Text("Edit in place keeps the photo where it is (albums, favourite, date, place) and Photos can revert it. New photo + delete adds the result with the original's date, place and albums, then iOS asks before moving the original to Recently Deleted. Either way the original's camera details and colour profile are carried over.")
                }
                Section {
                    TextField("Server URL", text: $server).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("API key (optional)", text: $apiKey)
                    Button("Test connection") { test() }
                    if let health { Text(health).font(.caption).foregroundStyle(Theme.dim) }
                } header: { Text("Server") } footer: {
                    Text("PixelBoost Cloud upscales run on the PixelBoost server through this address; results are kept 7 days. On-device upscales never leave the phone.")
                }
                Section("Temporary uploads (24h)") {
                    Button("Refresh") { refreshExports() }
                    ForEach(exports, id: \.id) { e in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(e.label.isEmpty ? e.filename : e.label).font(.subheadline)
                                Text("expires \(e.expiresAt)").font(.caption2).foregroundStyle(Theme.dim)
                            }
                            Spacer()
                            Button { download(e) } label: { Image(systemName: "arrow.down.circle") }
                        }
                        .swipeActions {
                            Button("Delete", role: .destructive) { try? Core.server.deleteExport(id: e.id); refreshExports() }
                        }
                    }
                }
                Section("About") {
                    LabeledContent("Version", value: Core.appVersion)
                    Text("Upscaling, fidelity refinement and every editing tool run in PixelBoost's own Rust engine. Models: Real-ESRGAN (BSD-3-Clause), converted to PixelBoost's .pbnn format.")
                        .font(.caption).foregroundStyle(Theme.dim)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.canvas)
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
            .onChange(of: server) { _, _ in Core.server.configure(baseUrl: server, apiKey: Prefs.apiKey) }
            .onChange(of: compute) { _, v in Core.engine.setCompute(mode: v); loadCompute() }
            .task { loadCompute() }
            .onChange(of: apiKey) { _, _ in Core.server.configure(baseUrl: server, apiKey: Prefs.apiKey) }
            .onChange(of: telemetry) { _, on in Core.telemetry.setEnabled(on: on); if on { Feedback.flushSoon() } }
        }
    }

    private var computeFooter: String {
        guard let c = computeInfo else { return "PixelBoost's own GPU kernels (Metal) run the upscaling networks; the first visit measures this iPhone once." }
        if c.gpu == nil { return "No usable GPU — everything runs on the CPU kernels." + (c.gpuNote.map { " (\($0))" } ?? "") }
        switch compute {
        case "cpu": return "Networks run on the CPU kernels only."
        case "gpu": return "Networks always run on the GPU (Metal). If the app goes to the background mid-run, the rest continues on the CPU."
        default: return c.autoUsesGpu ? "Auto runs networks on the GPU (faster here) and falls back to the CPU in the background." : "Auto stays on the CPU: it measured faster than the GPU on this device."
        }
    }

    private func loadCompute() {
        Task.detached(priority: .userInitiated) {
            let c = Core.engine.computeInfo()
            await MainActor.run { computeInfo = c }
        }
    }

    private func test() {
        health = "Checking…"
        Task.detached {
            let r = Result { try Core.server.health() }
            await MainActor.run {
                switch r {
                case .success(let v): health = "Connected · server \(v)"
                case .failure(let e): health = describe(e)
                }
            }
        }
    }

    private func refreshExports() {
        Task.detached {
            let r = Result { try Core.server.listExports() }
            await MainActor.run {
                switch r {
                case .success(let v): exports = v
                case .failure(let e): app.message = describe(e)
                }
            }
        }
    }

    private func download(_ e: StoredImage) {
        Task.detached {
            let dest = Core.newFile(URL(fileURLWithPath: e.filename).pathExtension.isEmpty ? "png" : URL(fileURLWithPath: e.filename).pathExtension)
            let r = Result { try Core.server.downloadExport(id: e.id, destPath: dest.path) }
            await MainActor.run {
                switch r {
                case .success: app.setPhoto(dest); app.mode = .upscale; dismiss()
                case .failure(let err): app.message = describe(err)
                }
            }
        }
    }
}

/// Optional models: download / remove (Rust ModelStore does the transfer,
/// resume and checksum).
struct DownloadsSection: View {
    @State private var items: [ModelDownload] = []
    @State private var progress: [String: Double] = [:]
    @State private var error: String?

    var body: some View {
        Section {
            ForEach(items, id: \.id) { m in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(m.title).font(.subheadline.weight(.semibold))
                        Spacer()
                        if let p = progress[m.id] {
                            ProgressView(value: p).frame(width: 90)
                        } else if m.installed {
                            Button("Remove", role: .destructive) { try? Core.downloads.remove(id: m.id); reload() }.font(.caption)
                        } else {
                            Button(m.partial > 0 ? "Resume" : "Get · \(formatBytes(m.bytes))") { download(m) }.font(.caption.weight(.semibold))
                        }
                    }
                    Text(m.detail).font(.caption).foregroundStyle(Theme.dim)
                }
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
        } header: { Text("Extra models") } footer: {
            Text("Downloaded from the PixelBoost server, checked, and kept on this iPhone (\(formatBytes(Core.downloads.diskBytes())) used).")
        }
        .onAppear(perform: reload)
    }

    private func reload() { items = Core.downloads.list() }

    private func download(_ m: ModelDownload) {
        progress[m.id] = 0
        error = nil
        let id = m.id
        let listener = DownloadRelay { r, t in Task { @MainActor in progress[id] = t > 0 ? Double(r) / Double(t) : 0 } }
        Task.detached(priority: .userInitiated) {
            let r = Result { try Core.downloads.download(server: Core.server, id: id, progress: listener) }
            await MainActor.run {
                progress[id] = nil
                if case .failure(let e) = r { error = describe(e) }
                reload()
            }
        }
    }
}

final class DownloadRelay: ModelProgress, @unchecked Sendable {
    private let f: (UInt64, UInt64) -> Void
    init(_ f: @escaping (UInt64, UInt64) -> Void) { self.f = f }
    func onProgress(received: UInt64, total: UInt64) { f(received, total) }
}

/// Cutout quality: HQ needs the IS-Net download.
struct CutoutQualityRow: View {
    @Binding var hq: Bool
    @State private var installed = Core.downloads.installed(id: "cutout-hq")
    var body: some View {
        HStack {
            Toggle("HQ edges (IS-Net)", isOn: $hq).font(.subheadline)
            if hq && !installed {
                Text("Download it in Settings › Extra models").font(.caption2).foregroundStyle(Theme.amber)
            }
        }
        .onAppear { installed = Core.downloads.installed(id: "cutout-hq") }
    }
}
