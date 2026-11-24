import Foundation

/// Subject cutout — PixelBoost's own segmentation networks in Rust
/// (U²-Net-p bundled; IS-Net "HQ" once downloaded). This only picks the
/// output file.
enum SubjectMask {
    /// Mask of the main subject, or only the subject under `point`
    /// (normalised, top-left origin) when given.
    static func make(for imageURL: URL, at point: CGPoint? = nil, hq: Bool = true) throws -> SubjectMaskResult {
        let out = Core.newFile("png")
        return try Core.engine.subjectMask(inputPath: imageURL.path, outputPath: out.path, hq: hq,
                                           pointX: Float(point?.x ?? -1), pointY: Float(point?.y ?? -1))
    }
}
