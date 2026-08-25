#!/usr/bin/env bash
#
# Launch the console on the workstation's physical display from an SSH session.
#
#   ./scripts/run-on-console.sh          # Vite dev server + Electron window
#   ./scripts/run-on-console.sh built    # serve the production build instead
#
# An SSH shell has no DISPLAY, so Electron has nowhere to draw. This points it
# at the seat's X server and hands it the credentials to open a window there.
# The window appears on the machine's own monitor, not in the SSH session.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-dev}"
export DISPLAY="${DISPLAY_OVERRIDE:-:1}"
export XAUTHORITY="${XAUTHORITY_OVERRIDE:-/run/user/$(id -u)/gdm/Xauthority}"

# Some tool environments export this to make the Electron binary behave as a
# plain Node interpreter. If it survives into the launch, `require('electron')`
# returns a path string instead of the API and the app dies on startup.
unset ELECTRON_RUN_AS_NODE

if [[ ! -e "/tmp/.X11-unix/X${DISPLAY#:}" ]]; then
  echo "No X server on ${DISPLAY}. Sockets present:" >&2
  ls /tmp/.X11-unix/ >&2 || true
  exit 1
fi

if [[ ! -r "$XAUTHORITY" ]]; then
  echo "Cannot read ${XAUTHORITY}." >&2
  echo "Log in on the machine's own screen once, or set XAUTHORITY_OVERRIDE." >&2
  exit 1
fi

echo "display   ${DISPLAY}"
echo "xauth     ${XAUTHORITY}"
echo "mode      ${MODE}"
echo

case "$MODE" in
  dev)
    echo "Starting Vite and opening the Electron window on ${DISPLAY}."
    echo "Ctrl-C here stops both."
    exec npm run dev
    ;;
  built)
    npm run build

    # Chromium's SUID sandbox helper has to be owned by root with mode 4755.
    # A plain `npm install` cannot set that, so on most workstations it is not
    # configured and Electron refuses to start rather than run unsandboxed.
    #
    # The renderer's own sandbox (`sandbox: true` in the BrowserWindow) is
    # unaffected either way; this is the outer process-level layer. Say plainly
    # when it is being dropped rather than passing the flag silently.
    SANDBOX_BIN="node_modules/electron/dist/chrome-sandbox"
    FLAGS=()
    if [[ -u "$SANDBOX_BIN" && "$(stat -c '%U' "$SANDBOX_BIN")" == "root" ]]; then
      echo "chrome-sandbox is configured; running with the sandbox enabled."
    else
      echo "chrome-sandbox is not setuid-root — starting with --no-sandbox."
      echo "To enable it:  sudo chown root:root ${SANDBOX_BIN} && sudo chmod 4755 ${SANDBOX_BIN}"
      FLAGS+=(--no-sandbox)
    fi

    echo "Opening the built app on ${DISPLAY}."
    exec ./node_modules/electron/dist/electron "${FLAGS[@]}" .
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use 'dev' or 'built'." >&2
    exit 2
    ;;
esac
