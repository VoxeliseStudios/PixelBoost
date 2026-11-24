import Foundation
import UIKit

/// Telemetry glue. The Rust engine records every measurement itself
/// (Core.telemetry is its sink and upload queue); this file only supplies
/// what only the platform knows — thermal state, Low Power Mode, battery,
/// memory — and the user's reactions to results, which judge the engine's
/// choices: kept / discarded / rating / re-run / Auto overridden / compare
/// pick / cutout repainted.
@MainActor
enum Feedback {
    static func start() {
        updateContext()
        let nc = NotificationCenter.default
        for n in [ProcessInfo.thermalStateDidChangeNotification, Notification.Name.NSProcessInfoPowerStateDidChange,
                  UIDevice.batteryLevelDidChangeNotification] {
            nc.addObserver(forName: n, object: nil, queue: .main) { _ in Task { @MainActor in updateContext() } }
        }
        UIDevice.current.isBatteryMonitoringEnabled = true
        flushSoon()
    }

    static func updateContext() {
        let t: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: t = "nominal"
        case .fair: t = "fair"
        case .serious: t = "serious"
        case .critical: t = "critical"
        @unknown default: t = "unknown"
        }
        Core.telemetry.setContext(thermal: t, lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled,
                                  battery: UIDevice.current.batteryLevel,
                                  memoryMb: UInt32(ProcessInfo.processInfo.physicalMemory / 1_048_576))
    }

    /// Upload what's queued (off the main thread; failures keep the queue).
    static func flushSoon() {
        guard Prefs.telemetry else { return }
        Task.detached(priority: .utility) { _ = try? Core.telemetry.flush(server: Core.server) }
    }

    static func send(_ run: String?, _ kind: String, _ value: Double = 1, _ detail: [String: Any] = [:]) {
        guard let run, !run.isEmpty else { return }
        let d = (try? JSONSerialization.data(withJSONObject: detail)).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        Core.telemetry.feedback(run: run, kind: kind, value: value, detailJson: d)
    }

    static func kept(_ run: String?, how: String) { send(run, "kept", 1, ["how": how]) }
    static func discarded(_ run: String?) { send(run, "discarded", 1) }
    static func rerun(_ run: String?) { send(run, "rerun", 1) }
    static func rate(_ run: String?, good: Bool) { send(run, "rating", good ? 1 : -1) }

    /// The user chose something else than what Auto would have picked.
    static func overrides(_ run: String?, auto a: UpscaleSettings?, used u: UpscaleSettings) {
        guard let a else { return }
        if a.model != u.model { send(run, "override", 1, ["field": "model", "auto": a.model, "user": u.model]) }
        if a.fidelity != u.fidelity { send(run, "override", 1, ["field": "fidelity", "auto": a.fidelity, "user": u.fidelity]) }
        if a.ensemble != u.ensemble { send(run, "override", 1, ["field": "quality", "auto": "\(a.ensemble)", "user": "\(u.ensemble)"]) }
        if abs(a.scale - u.scale) > 0.01 { send(run, "override", 1, ["field": "scale", "auto": "\(a.scale)", "user": "\(u.scale)"]) }
        if abs(a.detail - u.detail) > 0.05 { send(run, "override", 1, ["field": "detail", "auto": String(format: "%.2f", a.detail), "user": String(format: "%.2f", u.detail)]) }
        if a.adaptive != u.adaptive { send(run, "override", 1, ["field": "adaptive", "auto": "\(a.adaptive)", "user": "\(u.adaptive)"]) }
    }

    /// "Send this example": explicit, per result.
    static func sendExample(run: String, reason: String, source: URL, result: URL, note: String, done: @escaping (String) -> Void) {
        Task.detached(priority: .utility) {
            let r = Result { try Core.telemetry.sendSample(server: Core.server, run: run, reason: reason, sourcePath: source.path, resultPath: result.path, note: note) }
            await MainActor.run {
                switch r {
                case .success: done("Thanks — example sent")
                case .failure(let e): done(describe(e))
                }
            }
        }
    }
}
