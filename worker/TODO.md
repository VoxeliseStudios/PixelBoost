# worker/ — to do

Server-side upscaling (pixelboost-worker). Deploy: `docker compose up -d --build`
here; Mel's upscaler-bridge needs JOBS_UPSTREAM=http://192.168.1.67:8095 and
the same WORKER_KEY (Mel ~/bots/Music/.env: PIXELBOOST_WORKER_KEY).

- [x] Background check + local notification in the app (BGAppRefresh)
- [ ] Real push notification when a job finishes (needs APNs credentials).
- [ ] Depends on this PC keeping 192.168.1.67 (DHCP reservation recommended).

## Done (2026-11-15) — security pass (see ~/Projects/SECURITY-AUDIT.md)
- Host network instead of a Docker `ports:` publish (that bypassed ufw and exposed :8095 to the whole
  LAN). ufw: `allow from 192.168.1.87 to any port 8095` (Mel's upscaler-bridge only).
- Worker key compared in constant time.

