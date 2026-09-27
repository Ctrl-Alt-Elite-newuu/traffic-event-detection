# Deployment

The team server already runs Caddy for other sites, so we reuse it:

```
https://ctrl-alt-elite.sukoon.uz ─► existing Caddy (HTTPS, other sites untouched)
                                    ├── /api/*  ─► demo container on 127.0.0.1:8001 (FastAPI, CPU)
                                    └── else    ─► static website in /srv/site
```

## One-time server setup
1. DNS: `A` record `ctrl-alt-elite.sukoon.uz` → server IP.
2. Docker, rsync and a dedicated user for our deploys (`traffic`), allowed to run Docker:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo apt-get install -y rsync
   sudo adduser --disabled-password --gecos "" traffic && sudo usermod -aG docker traffic
   sudo mkdir -p /srv/site /opt/traffic /home/traffic/.ssh
   sudo chown -R traffic:traffic /srv/site /opt/traffic /home/traffic/.ssh
   ```
   Add the GitHub Actions public key to `/home/traffic/.ssh/authorized_keys`.
3. Add the contents of `caddy-site.caddy` to `/etc/caddy/Caddyfile`, then
   `sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy`.

## GitHub configuration
Secrets (both repositories): `SSH_HOST`, `SSH_USER` (`traffic`), `SSH_PORT` (`22`), `SSH_PRIVATE_KEY`.

Variables:

| Variable | Backend | Website |
|---|---|---|
| `DEPLOY_ENABLED` | `true` | `true` |
| `DOMAIN` | `ctrl-alt-elite.sukoon.uz` | — |
| `DEPLOY_DIR` | `/opt/traffic` | — |
| `SITE_DIR` | — | `/srv/site` |

The demo image is published to GHCR as `ghcr.io/ctrl-alt-elite-newuu/traffic-event-detection-demo`
and stays private: each deploy logs the server in with the workflow run's short-lived token and logs
out right after pulling.

## Branch flow
All work is pushed to `dev`; `main` only changes through a pull request from `dev`.

| Event | CI checks | Demo image | Deploy to server |
|---|---|---|---|
| push to `dev` | run | built, not published | skipped |
| pull request into `main` | run | built, not published | skipped |
| merge into `main` | run | built and published to GHCR | runs (when `DEPLOY_ENABLED` is `true`) |
