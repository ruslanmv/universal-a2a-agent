#!/usr/bin/env bash
# A2A server validation — PyPI package only
#
# This script uses the Python package's verifier (a2a_universal.verification)
# to validate one A2A server (or a list in bulk). No curl/jq is used.
#
# Usage:
#   scripts/a2a_verify.sh -u http://13.48.133.166:8080
#   scripts/a2a_verify.sh -u http://host:8080 -t 10        # custom timeout (sec)
#   scripts/a2a_verify.sh -u http://host:8080 -k            # disable TLS verification (diagnostics)
#   scripts/a2a_verify.sh -b urls.txt                       # bulk mode (one URL per line)
#   scripts/a2a_verify.sh -u http://host:8080 -o report.json
#
# Exit codes:
#   0  = status ok/degraded
#   1  = missing dependency (python3)
#   2  = verifier reported failure
#   3  = bad args / usage
#   4  = bulk file not found/empty
#   127= package not importable (pip install a2a-universal)

set -Eeuo pipefail

URL=""
TIMEOUT="8"
TLS_VERIFY="true"   # set false with -k
BULK_FILE=""
OUTFILE=""

usage() {
  cat <<USAGE
A2A server validation — PyPI package only

Options:
  -u <url>       Base URL to validate (e.g., http://host:8080)
  -t <seconds>   Timeout in seconds (default: 8)
  -k             Insecure mode — disable TLS verification (diagnostics only)
  -b <file>      Bulk mode: file with one URL per line (comments and blanks ignored)
  -o <file>      Save JSON report to file
  -h             Show this help

Examples:
  $0 -u http://13.48.133.166:8080
  $0 -b urls.txt -t 12 -o reports.json
USAGE
}

while getopts ":u:t:kb:o:h" opt; do
  case "$opt" in
    u) URL="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    k) TLS_VERIFY="false" ;;
    b) BULK_FILE="$OPTARG" ;;
    o) OUTFILE="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; usage; exit 3 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 3 ;;
  esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
pass() { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
info() { echo -e "${CYAN}›${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found"
  exit 1
fi

# Validate inputs
if [[ -n "$BULK_FILE" ]]; then
  if [[ ! -f "$BULK_FILE" ]]; then
    fail "Bulk file not found: $BULK_FILE"
    exit 4
  fi
  # Ensure at least one non-empty, non-comment line exists
  if ! grep -E -v '^(#|\s*$)' "$BULK_FILE" >/dev/null; then
    fail "Bulk file has no URLs: $BULK_FILE"
    exit 4
  fi
elif [[ -z "$URL" ]]; then
  usage
  exit 3
fi

# Announce
if [[ -n "$BULK_FILE" ]]; then
  info "Running PyPI verifier in BULK against: $(wc -l < <(grep -E -v '^(#|\s*$)' "$BULK_FILE")) URL(s)"
else
  info "Running PyPI verifier against: ${URL}"
fi

# Python verifier (single or bulk)
A2A_VERIFY_URL="$URL" \
A2A_VERIFY_TIMEOUT="$TIMEOUT" \
A2A_VERIFY_TLS="$TLS_VERIFY" \
A2A_VERIFY_BULK_FILE="$BULK_FILE" \
A2A_VERIFY_OUTPUT="$OUTFILE" \
python3 - <<'PY'
import asyncio, json, os, sys
from urllib.parse import urlparse

try:
    from a2a_universal.verification import verify_a2a, verify_a2a_bulk
except Exception as e:
    print(f"SKIP: a2a_universal not importable: {e}")
    sys.exit(127)

TIMEOUT = float(os.environ.get("A2A_VERIFY_TIMEOUT", "8") or 8)
TLS = (os.environ.get("A2A_VERIFY_TLS", "true").lower() != "false")
SINGLE_URL = os.environ.get("A2A_VERIFY_URL", "").strip()
BULK_FILE = os.environ.get("A2A_VERIFY_BULK_FILE", "").strip()
OUTFILE = os.environ.get("A2A_VERIFY_OUTPUT", "").strip()

async def run_single(url: str) -> int:
    report = await verify_a2a(url, timeout_sec=TIMEOUT, verify_tls=TLS)
    print(json.dumps(report, indent=2))

    # Helpful hint: detect Agent Card advertising localhost/other host
    try:
        base_host = urlparse(url).netloc
        rpc_host = urlparse((report.get("rpc_url") or "")).netloc
        card_url = (report.get("card", {}) or {}).get("url")
        if rpc_host and base_host and rpc_host != base_host:
            print(
                f"HINT: Agent Card RPC host '{rpc_host}' differs from base host '{base_host}'.\n"
                f"      The server likely started with an incorrect PUBLIC_URL.\n"
                f"      Card advertises: {card_url}",
                file=sys.stderr,
            )
    except Exception:
        pass

    status = (report or {}).get('status')
    return 0 if status in ("ok", "degraded") else 2

async def run_bulk(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        urls = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith('#')]
    reports = await verify_a2a_bulk(urls, concurrency=int(os.environ.get("VERIFY_CONCURRENCY", "50")), timeout_sec=TIMEOUT, verify_tls=TLS)
    print(json.dumps({"count": len(reports), "results": reports}, indent=2))
    # success if all are ok/degraded
    bad = [r for r in reports if r.get('status') not in ("ok", "degraded")]
    return 0 if not bad else 2

async def main() -> int:
    if BULK_FILE:
        rc = await run_bulk(BULK_FILE)
    else:
        rc = await run_single(SINGLE_URL or "http://127.0.0.1:8080")
    if OUTFILE:
        try:
            with open(OUTFILE, "w", encoding="utf-8") as f:
                # Write whatever was printed last by re-running minimal logic to fetch data
                if BULK_FILE:
                    with open(BULK_FILE, "r", encoding="utf-8") as fh:
                        urls = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith('#')]
                    data = await verify_a2a_bulk(urls)
                    json.dump({"count": len(data), "results": data}, f, indent=2)
                else:
                    data = await verify_a2a(SINGLE_URL or "http://127.0.0.1:8080")
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"WARN: failed to write output file: {e}", file=sys.stderr)
    return rc

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
PY
RC=$?

if [[ "$RC" -eq 0 ]]; then
  pass "Verifier status ok/degraded"
  exit 0
elif [[ "$RC" -eq 2 ]]; then
  fail "Verifier reported failure"
  exit 2
elif [[ "$RC" -eq 127 ]]; then
  warn "a2a_universal is not importable. Install with: pip install a2a-universal"
  exit 127
else
  fail "Verifier exited with unexpected code: $RC"
  exit "$RC"
fi
