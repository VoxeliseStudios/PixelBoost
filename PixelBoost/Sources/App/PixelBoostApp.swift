import SwiftUI

@main
struct PixelBoostApp: App {
    @StateObject private var app = AppModel()
    @Environment(\.scenePhase) private var scenePhase

    init() {
        Task { @MainActor in Feedback.start() }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(app)
                .preferredColorScheme(.dark)
                .tint(Theme.accent)
                .onOpenURL { url in app.importFile(url) }
        }
        .onChange(of: scenePhase) { _, phase in
            // iOS forbids GPU work in the background: running upscales
            // carry on on the CPU until the app is back.
            // (A continued-processing job the system gave GPU time keeps it.)
            Core.engine.setGpuPaused(paused: phase != .active && !BackgroundRun.gpuGranted)
            if phase == .background { Feedback.flushSoon() }
            if phase == .active { app.pickUpSharedPhoto() }
        }
    }
}
