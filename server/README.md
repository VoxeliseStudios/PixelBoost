# upscaler-bridge

PixelBoost's optional server: debug logging, 24-hour temporary image
storage (imports/exports), custom presets, device settings backup and the
model registry. Upscaling itself never leaves the device.

Written in Rust (`rust/crates/pb-server`, axum + sqlx). It replaced the
FastAPI version with the **same routes, payloads and Postgres schema**
(`rust/crates/pb-server/src/schema.sql`, applied idempotently at start), so
older app builds keep working.

## Endpoints

- `GET /health`
- Logging: `POST /log/upscale`, `GET|DELETE /log/history`, `DELETE /log/history/{id}`,
  `POST /log/action`, `POST /log/actions` (≤200), `GET /log/action-history`,
  `POST /log/snapshot`, `GET /log/snapshots`, `GET /log/stats`,
  `POST /log/comparison`, `POST /log/comparison/{id}/pick`, `GET /log/comparisons`,
  `GET /log/overview?days=`
- Storage (multipart `device_id`, `file`, `ttl_hours`, `label`, `is_auto`):
  `POST|GET|DELETE /import`, `GET|DELETE /import/{id}`, same for `/export`,
  `GET /storage/usage`
- `POST|GET /presets`, `DELETE /presets/{id}`, `PUT|GET /device-settings`, `GET /models`
- Telemetry (5.7): `POST /log/telemetry` (batch ≤2000 events), `GET /log/telemetry`
  (export: system / name / run / device_id / hours / after_id), `GET /log/telemetry/summary?days=`,
  `GET /log/telemetry/feedback?by=model`, `POST /log/telemetry/sample` (opt-in crops),
  `GET /log/telemetry/samples`, `GET /log/telemetry/sample/{id}/source|result`.
  Events older than 180 days and samples older than 30 are deleted hourly.
- `/jobs*` answers 410 Gone: PixelBoost Cloud (server-side upscaling) was removed in 5.8.
- `GET /models/file/{name}`: optional model downloads from `MODEL_FILES_DIR` (Range supported).

All but `/health` require `Authorization: Bearer $UPSCALER_BRIDGE_API_KEY`
when that variable is set.

## Config

`DATABASE_URL`, or `DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME`;
`HOST` (default 127.0.0.1), `PORT` (default 8003), `DB_POOL_MAX_SIZE` (4).

## Build / deploy

```
docker build -f server/Dockerfile -t pixelboost-upscaler-bridge .
```

On Mel it runs as compose service `upscaler-bridge` (container
`upscaler_bridge`, host network, port 8004, behind the Cloudflare tunnel).
