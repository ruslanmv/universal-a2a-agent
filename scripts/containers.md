## Quick start

```bash
# 1) Build a local image (multi-arch optional)
./scripts/build-containers.sh

# 2) Bring it up with Compose
./scripts/dev-up.sh

# 3) Check health
curl -s http://localhost:8000/healthz | python -m json.tool

# 4) Exercise endpoints
curl -s http://localhost:8000/a2a \
  -H 'Content-Type: application/json' \
  -d '{"method":"message/send","params":{"message":{"role":"user","messageId":"m1","parts":[{"type":"text","text":"ping"}]}}}' \
  | python -m json.tool

# 5) Tail logs
./scripts/dev-logs.sh

# 6) Tear down
./scripts/dev-down.sh
```
