# server/ — to do

- [x] Deployed on Mel 2026-11-08 (see ../RUST-REMAKE.md for rollback).

## 5.7 (2026-11-24)
- [x] Telemetry tables + endpoints (`/log/telemetry*`), worker telemetry pull, `/models/file/{name}`
      (needs `MODEL_FILES_DIR` on Mel with native-x2.pbnn + cutout-isnet.pbnn).
- [x] Deploy to Mel (see RUST-REMAKE.md 5.7).

## 2.1.0 (2026-11-24): cloud jobs retired

- [x] `/jobs*` → 410 Gone with an "update the app" message; worker
      telemetry pull, JOBS_UPSTREAM, WORKER_KEY and reqwest removed.
- [x] Telemetry insert keeps the sender's device_id (gpu events'
      `tags.device` used to overwrite it with the GPU name).
- [ ] Once no 5.7 apps call `/jobs` (check action_log `cloud_send`), the
      410 route can go too.

