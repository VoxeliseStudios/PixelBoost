import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Upscale a video (or a Live Photo's motion exported as video): pick it,
/// choose the network, run frame by frame with flicker control.
struct VideoSheet: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var item: PhotosPickerItem?
    @State private var source: URL?
    @State private var showFiles = false
    @State private var model = "fast"
    @State private var scale = 2.0
    @State private var progress: Double?
    @State private var result: VideoPump.Result?
    @State private var error: String?
    @State private var cancel = false
    @State private var background: BackgroundRun?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    PhotosPicker(selection: $item, matching: .videos, photoLibrary: .shared()) {
                        Label(source == nil ? "Choose a video" : "Choose another video", systemImage: "video")
                    }
                    Button { showFiles = true } label: { Label("From Files", systemImage: "folder") }
                    if let source { Text(source.lastPathComponent).font(.caption).foregroundStyle(Theme.dim) }
                }
                Section {
                    Picker("Network", selection: $model) {
                        Text("Fast").tag("fast")
                        Text("Anime / line art").tag("lineart")
                        Text("Photo (slow)").tag("photo")
                    }
                    Picker("Size", selection: $scale) {
                        Text("2×").tag(2.0)
                        Text("4×").tag(4.0)
                    }
                    .pickerStyle(.segmented)
                } footer: {
                    Text("Each frame is upscaled on this iPhone; still parts of the picture are kept steady between frames so the new detail doesn't shimmer. The sound is kept.")
                }
                if let progress {
                    Section {
                        ProgressView(value: progress) { Text("\(Int(progress * 100))%") }
                        Button("Stop", role: .destructive) { cancel = true }
                    }
                }
                if let r = result {
                    Section("Done · \(r.width) × \(r.height) · \(r.frames) frames") {
                        Button { save(r) } label: { Label("Save to Photos", systemImage: "square.and.arrow.down") }
                        ShareLink(item: r.url) { Label("Share", systemImage: "square.and.arrow.up") }
                    }
                }
                if let error { Text(error).foregroundStyle(.red).font(.caption) }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.canvas)
            .navigationTitle("Upscale video")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { cancel = true; dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Start") { start() }.disabled(source == nil || progress != nil) }
            }
            .fileImporter(isPresented: $showFiles, allowedContentTypes: [.movie]) { r in
                if case .success(let u) = r { source = copyIn(u) }
            }
            .onChange(of: item) { _, new in
                guard let new else { return }
                Task {
                    if let m = try? await new.loadTransferable(type: MovieFile.self) { source = m.url }
                    item = nil
                }
            }
        }
    }

    private func copyIn(_ u: URL) -> URL? {
        let scoped = u.startAccessingSecurityScopedResource()
        defer { if scoped { u.stopAccessingSecurityScopedResource() } }
        let dest = Core.newFile(u.pathExtension.isEmpty ? "mov" : u.pathExtension)
        return (try? FileManager.default.copyItem(at: u, to: dest)) != nil ? dest : nil
    }

    private func start() {
        guard let src = source else { return }
        let s = UpscaleSettings(model: model, scale: Float(scale), fidelity: "natural", detail: 0.2, denoise: 0, ensemble: 1, tile: 192,
                                maxOutputMp: 0, realism: 1, adaptive: false, prescale: 1, dehalo: 0)
        progress = 0
        result = nil
        error = nil
        cancel = false
        UIApplication.shared.isIdleTimerDisabled = true
        background = BackgroundRun.begin(title: "Upscaling a video", subtitle: src.lastPathComponent) { cancel = true }
        let flag = CancelFlag()
        Task {
            let watcher = Task { while !Task.isCancelled { if cancel { flag.set() }; try? await Task.sleep(nanoseconds: 200_000_000) } }
            do {
                let r = try await VideoPump.upscale(src, settings: s, progress: { p in Task { @MainActor in progress = p; background?.update(p) } },
                                                    cancelled: { flag.isSet })
                result = r
                background?.finish(success: true)
            } catch {
                self.error = describe(error)
                background?.finish(success: false)
            }
            watcher.cancel()
            background = nil
            progress = nil
            UIApplication.shared.isIdleTimerDisabled = false
            Feedback.flushSoon()
        }
    }

    private func save(_ r: VideoPump.Result) {
        Feedback.kept(r.runId, how: "save_video")
        Task {
            do { try await VideoPump.saveToPhotos(r.url); app.flash("Saved to Photos") } catch { self.error = describe(error) }
        }
    }
}

final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    func set() { lock.lock(); v = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return v }
}

/// A picked video, copied into the app's work folder.
struct MovieFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { m in SentTransferredFile(m.url) } importing: { received in
            let dest = Core.newFile(received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return MovieFile(url: dest)
        }
    }
}
