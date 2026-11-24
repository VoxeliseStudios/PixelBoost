import SwiftUI

/// Upscales made on this iPhone, and the batch runner.
struct LibraryScreen: View {
    @EnvironmentObject var app: AppModel
    @State private var showBatch = false
    @State private var history: [HistoryEntry] = []

    var body: some View {
        VStack(spacing: 0) {
            StudioHeader(title: "Library", showPhotoMenu: false)
            ScrollView {
                VStack(spacing: 12) {
                    HStack {
                        Text(history.isEmpty ? "" : "\(history.count) upscale\(history.count == 1 ? "" : "s")")
                            .font(.footnote.weight(.medium)).foregroundStyle(Theme.dim)
                        Spacer()
                        Button { showBatch = true } label: {
                            Label("Batch", systemImage: "square.stack.3d.up").font(.footnote.weight(.semibold))
                                .padding(.horizontal, 12).padding(.vertical, 8).background(Theme.card2, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                    historyGrid
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 110)
            }
            .refreshable { history = Core.history.list() }
        }
        .onAppear { history = Core.history.list() }
        .sheet(isPresented: $showBatch, onDismiss: { history = Core.history.list() }) {
            BatchScreen().presentationBackground(Theme.ink)
        }
    }

    @ViewBuilder
    private var historyGrid: some View {
        if history.isEmpty {
            VStack(spacing: 10) {
                Image(systemName: "photo.stack").font(.system(size: 44, weight: .light)).foregroundStyle(Theme.energy)
                Text("No upscales yet").font(.headline)
                Text("Upscales made on this iPhone show up here. Use Batch to upscale or edit many photos in one go.")
                    .font(.footnote).foregroundStyle(Theme.dim).multilineTextAlignment(.center)
            }
            .padding(.vertical, 50).padding(.horizontal, 20)
        }
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
            ForEach(history, id: \.id) { e in
                VStack(alignment: .leading, spacing: 4) {
                    AsyncThumb(path: e.thumbPath)
                        .frame(height: 130).frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    Text("\(e.outWidth) × \(e.outHeight)").font(.caption.monospacedDigit())
                    Text("\(UpscaleChoice.name(e.model)) · \(formatDuration(e.seconds))").font(.caption2).foregroundStyle(Theme.dim)
                }
                .contextMenu {
                    ShareLink(item: URL(fileURLWithPath: e.resultPath))
                    Button {
                        Task { try? await PhotoIO.saveToPhotos(URL(fileURLWithPath: e.resultPath)); app.flash("Saved to Photos") }
                    } label: { Label("Save to Photos", systemImage: "square.and.arrow.down") }
                    Button {
                        let copy = Core.newFile(URL(fileURLWithPath: e.resultPath).pathExtension)
                        try? FileManager.default.copyItem(at: URL(fileURLWithPath: e.resultPath), to: copy)
                        app.setPhoto(copy)
                        app.mode = .edit
                    } label: { Label("Edit", systemImage: "slider.horizontal.3") }
                    Button(role: .destructive) { try? Core.history.delete(id: e.id); history = Core.history.list() } label: { Label("Delete", systemImage: "trash") }
                }
            }
        }
    }
}

struct AsyncThumb: View {
    let path: String
    @State private var img: UIImage?
    var body: some View {
        ZStack {
            Theme.card2
            if let img { Image(uiImage: img).resizable().scaledToFill() }
        }
        .task(id: path) { img = UIImage(contentsOfFile: path) }
    }
}
