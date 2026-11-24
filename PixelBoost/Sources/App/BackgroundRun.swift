import BackgroundTasks
import Foundation

/// Keeps a long on-device job running when the user leaves the app
/// (iOS 26 continued processing: the system shows its progress, the user
/// can stop it there). Older iOS: the job pauses the GPU and continues on
/// the CPU for as long as iOS allows, as before.
@MainActor
final class BackgroundRun {
    /// The system granted GPU time to the running background job.
    static private(set) var gpuGranted = false

    private var task: AnyObject?
    /// GPU time was asked for (and is supported on this device).
    private var wantsGpu = false
    private var fraction: Double = 0
    private var finished = false
    private let onExpire: () -> Void

    private init(onExpire: @escaping () -> Void) { self.onExpire = onExpire }

    /// Start a continued-processing request for a job the user just began.
    /// `onExpire` cancels the job if the system (or the user) stops it.
    static func begin(title: String, subtitle: String, onExpire: @escaping () -> Void) -> BackgroundRun? {
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) {
            let run = BackgroundRun(onExpire: onExpire)
            let id = "com.pixelboost.upscale.\(UUID().uuidString)"
            let ok = BGTaskScheduler.shared.register(forTaskWithIdentifier: id, using: .main) { t in
                guard let t = t as? BGContinuedProcessingTask else { t.setTaskCompleted(success: false); return }
                MainActor.assumeIsolated {
                    run.task = t
                    BackgroundRun.gpuGranted = run.wantsGpu
                    t.progress.totalUnitCount = 1000
                    t.progress.completedUnitCount = Int64(run.fraction * 1000)
                    t.expirationHandler = { Task { @MainActor in run.expire() } }
                    if run.finished { t.setTaskCompleted(success: true) }
                }
            }
            guard ok else { return nil }
            let req = BGContinuedProcessingTaskRequest(identifier: id, title: title, subtitle: subtitle)
            req.strategy = .fail
            if BGTaskScheduler.supportedResources.contains(.gpu) {
                req.requiredResources = .gpu
                run.wantsGpu = true
            }
            do { try BGTaskScheduler.shared.submit(req) } catch { return nil }
            return run
        }
        #endif
        return nil
    }

    func update(_ f: Double, subtitle: String? = nil) {
        fraction = min(max(f, 0), 1)
        #if compiler(>=6.2)
        if #available(iOS 26.0, *), let t = task as? BGContinuedProcessingTask {
            t.progress.completedUnitCount = Int64(fraction * 1000)
            if let subtitle { t.updateTitle(t.title, subtitle: subtitle) }
        }
        #endif
    }

    func finish(success: Bool) {
        finished = true
        BackgroundRun.gpuGranted = false
        #if compiler(>=6.2)
        if #available(iOS 26.0, *), let t = task as? BGContinuedProcessingTask {
            t.progress.completedUnitCount = t.progress.totalUnitCount
            t.setTaskCompleted(success: success)
        }
        #endif
        task = nil
    }

    private func expire() {
        guard !finished else { return }
        onExpire()
        finish(success: false)
    }
}
