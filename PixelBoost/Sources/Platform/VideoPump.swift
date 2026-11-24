import AVFoundation
import Photos
import CoreVideo
import Foundation

/// Video in / out with Apple's codecs (AVFoundation — the only H.264/HEVC
/// codecs on iOS). Every frame's pixels go to the Rust `VideoUpscaler`
/// (upscale + temporal stabilisation); this only moves buffers between the
/// decoder, Rust and the encoder, then puts the original sound back.
enum VideoPump {
    struct Result { let url: URL; let width: Int; let height: Int; let frames: Int; let runId: String }

    enum PumpError: LocalizedError {
        case noVideo, reader(String), writer(String), cancelled
        var errorDescription: String? {
            switch self {
            case .noVideo: return "That file has no video track."
            case .reader(let m): return "Couldn't read the video (\(m))."
            case .writer(let m): return "Couldn't write the video (\(m))."
            case .cancelled: return "Cancelled"
            }
        }
    }

    static func upscale(_ input: URL, settings: UpscaleSettings, progress: @escaping @Sendable (Double) -> Void,
                        cancelled: @escaping @Sendable () -> Bool) async throws -> Result {
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw PumpError.noVideo }
        let duration = try await asset.load(.duration).seconds
        let fps = try await track.load(.nominalFrameRate)
        let transform = try await track.load(.preferredTransform)
        let expected = max(1, Int(Double(fps) * duration))

        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading() else { throw PumpError.reader(reader.error?.localizedDescription ?? "?") }

        let up = VideoUpscaler(core: Core.engine, settings: settings)
        let silent = Core.newFile("mov")
        var writer: AVAssetWriter?
        var input: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?
        var size = (0, 0)
        var n = 0
        while let sample = out.copyNextSampleBuffer() {
            if cancelled() { reader.cancelReading(); writer?.cancelWriting(); throw PumpError.cancelled }
            guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), stride = CVPixelBufferGetBytesPerRow(pb)
            let data = Data(bytes: CVPixelBufferGetBaseAddress(pb)!, count: stride * h)
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            let frame = try up.pushBgra(width: UInt32(w), height: UInt32(h), stride: UInt32(stride), data: data)
            if writer == nil {
                size = (Int(frame.width), Int(frame.height))
                let wr = try AVAssetWriter(outputURL: silent, fileType: .mov)
                let inp = AVAssetWriterInput(mediaType: .video, outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: size.0, AVVideoHeightKey: size.1,
                ])
                inp.transform = transform
                inp.expectsMediaDataInRealTime = false
                let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: size.0, kCVPixelBufferHeightKey as String: size.1,
                ])
                wr.add(inp)
                guard wr.startWriting() else { throw PumpError.writer(wr.error?.localizedDescription ?? "?") }
                wr.startSession(atSourceTime: pts)
                (writer, input, adaptor) = (wr, inp, ad)
            }
            guard let input, let adaptor, let pool = adaptor.pixelBufferPool else { throw PumpError.writer("no buffer pool") }
            var ob: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &ob)
            guard let ob else { throw PumpError.writer("no buffer") }
            CVPixelBufferLockBaseAddress(ob, [])
            let dst = CVPixelBufferGetBaseAddress(ob)!, dstStride = CVPixelBufferGetBytesPerRow(ob), row = size.0 * 4
            frame.rgba.withUnsafeBytes { src in
                for y in 0..<size.1 { memcpy(dst.advanced(by: y * dstStride), src.baseAddress!.advanced(by: y * row), row) }
            }
            CVPixelBufferUnlockBaseAddress(ob, [])
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 5_000_000) }
            adaptor.append(ob, withPresentationTime: pts)
            n += 1
            progress(min(0.99, Double(n) / Double(expected)))
        }
        up.finish()
        guard let writer, let input else { throw PumpError.reader("no frames") }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw PumpError.writer(writer.error?.localizedDescription ?? "?") }
        let final = try await addSound(video: silent, from: asset)
        progress(1)
        return Result(url: final, width: size.0, height: size.1, frames: n, runId: up.runId())
    }

    /// Put the original sound track(s) back (no re-encode).
    private static func addSound(video: URL, from source: AVURLAsset) async throws -> URL {
        let audio = try await source.loadTracks(withMediaType: .audio)
        guard !audio.isEmpty else { return video }
        let comp = AVMutableComposition()
        let v = AVURLAsset(url: video)
        guard let vt = try await v.loadTracks(withMediaType: .video).first else { return video }
        let range = CMTimeRange(start: .zero, duration: try await v.load(.duration))
        let cv = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        try cv?.insertTimeRange(range, of: vt, at: .zero)
        cv?.preferredTransform = try await vt.load(.preferredTransform)
        for a in audio {
            let ca = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
            try? ca?.insertTimeRange(range, of: a, at: .zero)
        }
        let out = Core.newFile("mov")
        guard let ex = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetPassthrough) else { return video }
        if #available(iOS 18.0, *) {
            try await ex.export(to: out, as: .mov)
        } else {
            ex.outputURL = out
            ex.outputFileType = .mov
            await ex.export()
            if ex.status != .completed { return video }
        }
        return out
    }

    static func saveToPhotos(_ url: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { throw PhotoIO.ImportError.noPhotoAccess }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: url, options: nil)
        }
    }
}
