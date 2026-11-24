import ActivityKit
import PhotosUI
import SwiftUI

/// Queue many photos, upscale them one after another with the current
/// Upscale settings, save each to Photos.
struct BatchScreen: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var job = UpscaleJob()
    @State private var items: [PhotosPickerItem] = []
    @State private var queue: [BatchItem] = []
    @State private var running = false
    @State private var cancelled = false
    /// "new" (save results as new photos), "replace" (replace each
    /// original in the library) or "none".
    @AppStorage("batch.output") private var output = "new"
    /// "upscale" or "edit" (apply a preset / copied edits to every photo).
    @AppStorage("batch.task") private var task = "upscale"
    /// Preset id, or "copied" for the copied edits.
    @AppStorage("batch.recipe") private var recipe = "copied"
    @State private var token = CancelToken()

    let choice = UpscaleChoice()

    struct BatchItem: Identifiable {
        let id = UUID()
        var source: URL?
        var asset: String?
        var status = "Waiting"
        var result: URL?
        var thumb: UIImage?
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Picker("Do", selection: $task) {
                        Text("Upscale").tag("upscale")
                        Text("Apply edits").tag("edit")
                    }
                    .pickerStyle(.segmented)
                    .disabled(running)
                    if task == "edit" {
                        Picker("Edits", selection: $recipe) {
                            if app.copiedEdits != nil { Text("Copied edits").tag("copied") }
                            ForEach(Core.presets.list(), id: \.id) { p in Text(p.name).tag(p.id) }
                        }
                    }
                }
                Section {
                    PhotosPicker(selection: $items, maxSelectionCount: 50, matching: .images, photoLibrary: .shared()) {
                        Label("Add photos", systemImage: "plus.circle.fill")
                    }
                    .disabled(running)
                    Picker("Results", selection: $output) {
                        Text("Save as new photos").tag("new")
                        Text("Replace the originals").tag("replace")
                        Text("Keep in the app").tag("none")
                    }
                    if task == "upscale" {
                    Text(choice.smart
                         ? "Runs on this iPhone. Smart settings look at each photo and choose its size, model and clean-up."
                         : "Runs on this iPhone with your upscale settings: \(UpscaleChoice.name(choice.model)), \(Int(choice.scale))×, \(choice.fidelity).")
                        .font(.caption).foregroundStyle(Theme.dim)
                    } else {
                        Text("The same edits on every photo, at full resolution, on this iPhone.").font(.caption).foregroundStyle(Theme.dim)
                    }
                    if output == "replace" {
                        Text(PhotoLibrary.method == .edit
                             ? "Each original is replaced as an edit — Revert in Photos brings it back."
                             : "Each result is added with the original's date, place and albums, then iOS asks to delete the originals.")
                            .font(.caption).foregroundStyle(Theme.dim)
                    }
                }
                if !queue.isEmpty {
                    Section("Queue") {
                        ForEach(queue) { item in
                            HStack(spacing: 12) {
                                Group {
                                    if let t = item.thumb { Image(uiImage: t).resizable().scaledToFill() } else { Theme.panel2 }
                                }
                                .frame(width: 48, height: 48).clipShape(RoundedRectangle(cornerRadius: 8))
                                Text(item.status).font(.subheadline)
                                Spacer()
                                if let r = item.result { ShareLink(item: r) { Image(systemName: "square.and.arrow.up") } }
                            }
                        }
                        .onDelete { idx in if !running { queue.remove(atOffsets: idx) } }
                    }
                    if running {
                        Section {
                            ProgressView(value: job.progress) { Text("Current photo") }
                            Button("Stop after this photo", role: .destructive) { cancelled = true; token.cancel() }
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.canvas)
            .navigationTitle("Batch on iPhone")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(running ? "Running…" : "Start") { start() }
                        .disabled(running || queue.allSatisfy { $0.result != nil })
                }
            }
        }
        .onChange(of: items) { _, new in
            guard !new.isEmpty else { return }
            let picked = new
            items = []
            Task {
                for p in picked {
                    var b = BatchItem()
                    b.asset = p.itemIdentifier
                    if let u = try? await PhotoIO.importItem(p) {
                        b.source = u
                        b.thumb = (try? Core.engine.loadPreview(path: u.path, maxSide: 160))?.uiImage
                    } else {
                        b.status = "Couldn't read"
                    }
                    queue.append(b)
                }
            }
        }
    }

    private func start() {
        if task == "edit" { startEdits(); return }
        running = true
        cancelled = false
        let activity = BatchActivity.start(total: queue.count)
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            var finished = 0
            for i in queue.indices {
                if cancelled { break }
                guard queue[i].result == nil, let src = queue[i].source,
                      let info = try? Core.engine.imageInfo(path: src.path) else { continue }
                queue[i].status = "Upscaling…"
                BatchActivity.update(activity, done: finished, total: queue.count, name: "Photo \(i + 1)")
                let s = await choice.resolved(for: src)
                let scale = Double(s.scale)
                let o = await job.run(input: src, settings: s, base: nil,
                                      outW: Int(Double(info.width) * scale), outH: Int(Double(info.height) * scale))
                if let o {
                    let url = URL(fileURLWithPath: o.outputPath)
                    queue[i].result = url
                    queue[i].status = "\(o.width)×\(o.height) · \(formatDuration(o.seconds))"
                    switch output {
                    case "replace" where queue[i].asset != nil:
                        let note = Core.engine.replacementNote(model: o.model, scale: s.scale, inWidth: info.width, inHeight: info.height)
                        do {
                            let r = try await PhotoLibrary.replace(assetID: queue[i].asset!, with: url, note: note)
                            queue[i].status += r == .edited || r == .swapped ? " · replaced" : " · added"
                        } catch {
                            queue[i].status += " · not replaced"
                            app.message = describe(error)
                        }
                    case "none":
                        break
                    default:
                        try? await PhotoIO.saveToPhotos(url)
                    }
                    let entry = HistoryEntry(id: "", createdAt: 0, model: o.model, scale: Float(scale), inWidth: info.width, inHeight: info.height,
                                             outWidth: o.width, outHeight: o.height, seconds: o.seconds, fidelityDb: o.fidelityDb, resultPath: "", thumbPath: "")
                    _ = try? Core.history.add(entry: entry, resultFile: o.outputPath)
                } else {
                    queue[i].status = "Failed"
                }
                finished += 1
            }
            BatchActivity.end(activity, done: finished, total: queue.count)
            UIApplication.shared.isIdleTimerDisabled = false
            app.log("batch", ["count": finished])
            running = false
        }
    }
}

extension BatchScreen {
    /// Apply one recipe to every queued photo (Rust batch_edit).
    func startEdits() {
        let op: String?
        if recipe == "copied" {
            op = app.copiedEdits
        } else {
            op = Core.presets.list().first { $0.id == recipe }?.opJson
        }
        guard let op else { app.message = "Choose the edits to apply (copy edits in the editor, or pick a preset)."; return }
        let todo = queue.indices.filter { queue[$0].result == nil && queue[$0].source != nil }
        guard !todo.isEmpty else { return }
        running = true
        token = CancelToken()
        let inputs = todo.map { queue[$0].source!.path }
        let outputs = todo.map { _ in Core.newFile("jpg").path }
        for i in todo { queue[i].status = "Waiting" }
        let listener = BatchRelay { idx, _, out, err in
            Task { @MainActor in
                let qi = todo[Int(idx)]
                if err.isEmpty {
                    queue[qi].result = URL(fileURLWithPath: out)
                    queue[qi].status = "Edited"
                    queue[qi].thumb = (try? Core.engine.loadPreview(path: out, maxSide: 160))?.uiImage
                } else {
                    queue[qi].status = "Failed: \(err)"
                }
            }
        }
        let tok = token
        let (outMode, assets) = (output, todo.map { queue[$0].asset })
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            let r = await Task.detached(priority: .userInitiated) {
                Result { try Core.engine.batchEdit(inputs: inputs, outputs: outputs, opJson: op, listener: listener, cancel: tok) }
            }.value
            UIApplication.shared.isIdleTimerDisabled = false
            switch r {
            case .success(let sum):
                for (k, out) in outputs.enumerated() where FileManager.default.fileExists(atPath: out) {
                    let url = URL(fileURLWithPath: out)
                    if outMode == "replace", let a = assets[k] {
                        _ = try? await PhotoLibrary.replace(assetID: a, with: url, note: "{\"app\":\"PixelBoost\",\"op\":\"batch_edit\"}")
                    } else if outMode == "new" {
                        try? await PhotoIO.saveToPhotos(url)
                    }
                }
                if sum.done > 0 { Feedback.kept(sum.runId, how: "batch_\(outMode)") }
                app.flash("\(sum.done) edited" + (sum.failed > 0 ? ", \(sum.failed) failed" : ""))
            case .failure(let e):
                app.message = describe(e)
            }
            running = false
        }
    }
}

final class BatchRelay: BatchListener, @unchecked Sendable {
    private let f: (UInt32, UInt32, String, String) -> Void
    init(_ f: @escaping (UInt32, UInt32, String, String) -> Void) { self.f = f }
    func onItem(index: UInt32, total: UInt32, outputPath: String, error: String) { f(index, total, outputPath, error) }
}

enum BatchActivity {
    static func start(total: Int) -> Activity<BatchUpscaleActivityAttributes>? {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else { return nil }
        let state = BatchUpscaleActivityAttributes.ContentState(completedCount: 0, totalCount: total, currentFileName: nil)
        return try? Activity.request(attributes: BatchUpscaleActivityAttributes(startedAt: Date()),
                                     content: .init(state: state, staleDate: nil))
    }

    static func update(_ a: Activity<BatchUpscaleActivityAttributes>?, done: Int, total: Int, name: String) {
        guard let a else { return }
        let state = BatchUpscaleActivityAttributes.ContentState(completedCount: done, totalCount: total, currentFileName: name)
        Task { await a.update(.init(state: state, staleDate: nil)) }
    }

    static func end(_ a: Activity<BatchUpscaleActivityAttributes>?, done: Int, total: Int) {
        guard let a else { return }
        let state = BatchUpscaleActivityAttributes.ContentState(completedCount: done, totalCount: total, currentFileName: nil)
        Task { await a.end(.init(state: state, staleDate: nil), dismissalPolicy: .default) }
    }
}
