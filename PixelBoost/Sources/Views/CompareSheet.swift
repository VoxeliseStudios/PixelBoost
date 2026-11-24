import SwiftUI
import UIKit

/// Pixel loupe: real pixels of the result under the finger, drawn by Rust
/// (nearest neighbour, so pixels stay square).
struct PixelLoupe: View {
    let path: String
    let width: Int
    let height: Int
    /// Normalised position on the picture.
    let at: CGPoint
    @State private var image: UIImage?
    private let span = 24, zoom = 6

    var body: some View {
        Group {
            if let image {
                Image(uiImage: image).interpolation(.none).resizable()
            } else {
                Color.black
            }
        }
        .frame(width: CGFloat(span * zoom), height: CGFloat(span * zoom))
        .clipShape(Circle())
        .overlay(Circle().stroke(Theme.amber, lineWidth: 2))
        .overlay(Rectangle().stroke(.white.opacity(0.8), lineWidth: 1).frame(width: CGFloat(zoom), height: CGFloat(zoom)))
        .shadow(radius: 8)
        .task(id: key) { await load() }
    }

    private var key: String {
        let x = Int(at.x * CGFloat(width)), y = Int(at.y * CGFloat(height))
        return "\(x / 2)|\(y / 2)"
    }

    private func load() async {
        let x = max(0, Int(at.x * CGFloat(width)) - span / 2), y = max(0, Int(at.y * CGFloat(height)) - span / 2)
        let (p, s, z) = (path, span, zoom)
        let img = await Task.detached(priority: .userInitiated) {
            (try? Core.engine.loupe(path: p, x: UInt32(x), y: UInt32(y), w: UInt32(s), h: UInt32(s), zoom: UInt32(z)))?.uiImage
        }.value
        image = img
    }
}

/// Runs the same crop of the photo through several models and variants;
/// tap the one that looks best (that choice is the most valuable signal for
/// tuning Auto). Hold a tile to inspect its pixels.
struct CompareSheet: View {
    let source: URL
    let scale: Float
    @Environment(\.dismiss) private var dismiss
    @State private var result: CompareResult?
    @State private var progress = (0, 0)
    @State private var error: String?
    @State private var picked: String?
    @State private var loupe: (String, CGPoint)?
    @State private var token = CancelToken()

    private let candidates = ["photo", "photo:r0.5", "photo+adaptive", "clean", "anime", "fast", "lineart"]

    var body: some View {
        NavigationStack {
            ScrollView {
                if let r = result {
                    Text("Auto would pick \(UpscaleChoice.name(r.autoPick)). Tap the one you like best.")
                        .font(.footnote).foregroundStyle(Theme.dim).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal)
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        ForEach(r.candidates, id: \.id) { c in tile(c, run: r.runId, all: r.candidates.map(\.id)) }
                    }
                    .padding()
                } else if let error {
                    Text(error).foregroundStyle(.red).padding()
                } else {
                    VStack(spacing: 10) {
                        ProgressView(value: Double(progress.0), total: Double(max(progress.1, 1)))
                        Text("Running \(progress.0 + 1) of \(candidates.count)…").font(.caption).foregroundStyle(Theme.dim)
                    }
                    .padding(40)
                }
            }
            .background(Theme.canvas)
            .navigationTitle("Compare models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { token.cancel(); dismiss() } } }
            .task { await run() }
        }
    }

    private func tile(_ c: CompareCandidate, run: String, all: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let img = c.image.uiImage {
                GeometryReader { g in
                    Image(uiImage: img).resizable().interpolation(.high).scaledToFill()
                        .frame(width: g.size.width, height: g.size.width).clipped()
                        .overlay {
                            if let l = loupe, l.0 == c.id {
                                LoupeView(image: c.image, at: l.1).position(x: l.1.x * g.size.width, y: max(l.1.y * g.size.width - 60, 50))
                            }
                        }
                        .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                            loupe = (c.id, CGPoint(x: v.location.x / g.size.width, y: v.location.y / g.size.width))
                        }.onEnded { v in
                            loupe = nil
                            if abs(v.translation.width) < 6, abs(v.translation.height) < 6 { pick(c.id, run: run, all: all) }
                        })
                }
                .aspectRatio(1, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(picked == c.id ? Theme.amber : Theme.line, lineWidth: picked == c.id ? 3 : 1))
            }
            Text(c.label).font(.caption.weight(.semibold)).lineLimit(1)
            Text(String(format: "%.1f dB · detail %.2f · %.1fs", c.fidelityDb, c.detail, c.ms / 1000)).font(.caption2.monospacedDigit()).foregroundStyle(Theme.dim)
        }
    }

    private func pick(_ id: String, run: String, all: [String]) {
        picked = id
        Feedback.send(run, "compare_pick", 1, ["picked": id, "candidates": all.joined(separator: ",")])
    }

    private func run() async {
        let (src, s, cand, tok) = (source.path, scale, candidates, token)
        let listener = CompareProgress { d, t in Task { @MainActor in progress = (Int(d), Int(t)) } }
        let r = await Task.detached(priority: .userInitiated) {
            Result { try Core.engine.compareModels(inputPath: src, crop: [], size: 160, candidates: cand, scale: s, listener: listener, cancel: tok) }
        }.value
        switch r {
        case .success(let v): result = v
        case .failure(let e): error = describe(e)
        }
    }
}

/// Loupe over an in-memory result (compare tiles).
struct LoupeView: View {
    let image: RgbaImage
    let at: CGPoint
    var body: some View {
        let span = 16, zoom = 6
        let x = max(0, min(Int(image.width) - span, Int(at.x * CGFloat(image.width)) - span / 2))
        let y = max(0, min(Int(image.height) - span, Int(at.y * CGFloat(image.height)) - span / 2))
        let crop = image.crop(x: x, y: y, w: span, h: span)
        return Image(uiImage: crop?.uiImage ?? UIImage()).interpolation(.none).resizable()
            .frame(width: CGFloat(span * zoom), height: CGFloat(span * zoom))
            .clipShape(Circle()).overlay(Circle().stroke(Theme.amber, lineWidth: 2)).shadow(radius: 6)
    }
}

final class CompareProgress: CompareListener, @unchecked Sendable {
    private let f: (UInt32, UInt32) -> Void
    init(_ f: @escaping (UInt32, UInt32) -> Void) { self.f = f }
    func onCandidate(done: UInt32, total: UInt32, id: String) { f(done, total) }
}

extension RgbaImage {
    /// Byte copy of a rectangle (no pixel processing).
    func crop(x: Int, y: Int, w: Int, h: Int) -> RgbaImage? {
        let W = Int(width), H = Int(height)
        guard x >= 0, y >= 0, w > 0, h > 0, x + w <= W, y + h <= H else { return nil }
        var out = Data(capacity: w * h * 4)
        for row in y..<(y + h) {
            out.append(rgba.subdata(in: (row * W + x) * 4..<(row * W + x + w) * 4))
        }
        return RgbaImage(width: UInt32(w), height: UInt32(h), rgba: out)
    }
}
