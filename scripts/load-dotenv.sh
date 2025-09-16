#!/usr/bin/env bash
# load-dotenv.sh — source-friendly loader for .env using python-dotenv
# Usage:
#   source ./scripts/load-dotenv.sh [--file .env] [--no-overwrite]
# Or print export lines (for Make / eval):
#   ./scripts/load-dotenv.sh --file .env --print
#   eval "$(./scripts/load-dotenv.sh --print)"
set -euo pipefail

# --- Configuration and Argument Parsing ---
FILE=".env"
PRINT=0
NO_OVERWRITE=0
MASK=0

# If executed (not sourced), default to printing export lines
if [[ "${BASH_SOURCE[0]-$0}" == "$0" ]]; then
  PRINT=1
fi
SOURCED=1
[[ "${BASH_SOURCE[0]-$0}" == "$0" ]] && SOURCED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file) FILE="$2"; shift 2 ;;
    -p|--print) PRINT=1; shift ;;
    --no-overwrite) NO_OVERWRITE=1; shift ;;
    --mask) MASK=1; shift ;;
    *) echo "load-dotenv: unknown option $1" >&2; [[ $SOURCED -eq 1 ]] && return 2 || exit 2 ;;
  esac
done

if [[ ! -f "$FILE" ]]; then
  echo "load-dotenv: $FILE not found" >&2
  [[ $SOURCED -eq 1 ]] && return 0 || exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Main Logic ---

# Check if python-dotenv is available
if ! "$PYTHON_BIN" -c "import dotenv" >/dev/null 2>&1; then
  # FIXED: Replaced inconsistent fallback with a unified, robust shell-based parser.
  # This block now correctly handles sourcing, printing, --no-overwrite, and --mask without python-dotenv.
  echo "load-dotenv: python-dotenv not found, using shell fallback parser." >&2

  # Use awk to parse the .env file into a series of `export KEY='VALUE'` commands.
  # This handles comments, whitespace, and basic value quoting.
  RAW_OUT=$(awk -F= '
    # Ignore comments and empty lines
    /^[[:space:]]*#/ || /^[[:space:]]*$/ {next}
    # Process valid lines
    {
      # Extract key, which is everything before the first "="
      key=substr($0, 1, index($0, "=") - 1);
      # Trim trailing whitespace from key
      sub(/[[:space:]]*$/, "", key);
      sub(/^[[:space:]]*/, "", key);
      # Extract value, which is everything after the first "="
      val=substr($0, index($0, "=") + 1);
      # Escape single quotes in the value for safe evaluation
      gsub(/'\''/, "'\\''", val);
      printf("export %s='\''%s'\''\n", key, val);
    }' "$FILE")

  # If --no-overwrite is set, filter out variables that are already defined.
  if (( NO_OVERWRITE )); then
    FILTERED_OUT=""
    while IFS= read -r line; do
      # Use bash regex to extract the variable key from the `export KEY='...'` line
      if [[ "$line" =~ ^export[[:space:]]+([A-Za-z_][A-Za-z_0-9]*) ]]; then
        key="${BASH_REMATCH[1]}"
        # The ${!key+x} syntax checks if a variable is set. If not, we keep the line.
        if [ -z "${!key+x}" ]; then
          FILTERED_OUT+="${line}"$'\n'
        fi
      fi
    done <<< "$RAW_OUT"
    OUT="$FILTERED_OUT"
  else
    OUT="$RAW_OUT"
  fi

  # Now, either print the result or evaluate it in the current shell.
  if (( PRINT )); then
    if (( MASK )); then
      # Replace values with ***
      printf '%s\n' "$OUT" | sed -E "s/^(export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=).*/\1'***'/"
    else
      printf '%s\n' "$OUT"
    fi
  else
    # Apply to current shell by evaluating the generated export commands.
    eval "$OUT"
  fi
  # We are done with the fallback, so return/exit successfully.
  [[ $SOURCED -eq 1 ]] && return 0 || exit 0
fi

# If we are here, python-dotenv IS available. Use the robust Python parser.
OUT="$("$PYTHON_BIN" - "$FILE" "$NO_OVERWRITE" <<'PY'
from dotenv import dotenv_values
import os, shlex, sys
path = sys.argv[1]
no_overwrite = sys.argv[2] == "1"
vals = dotenv_values(path)  # robust parsing (quotes, comments, expansion)
lines = []
for k, v in vals.items():
    if v is None:
        v = ""
    if no_overwrite and k in os.environ:
        continue
    # shlex.quote ensures the value is safe for shell evaluation
    lines.append(f"export {k}={shlex.quote(str(v))}")
print("\n".join(lines))
PY
)"

# Final processing of the output from the Python script.
if (( PRINT )); then
  if (( MASK )); then
    # Replace values with ***
    printf '%s\n' "$OUT" | sed -E "s/^(export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=).*/\1'***'/"
  else
    printf '%s\n' "$OUT"
  fi
else
  # Apply to current shell (only works when this script is *sourced*)
  eval "$OUT"
fi