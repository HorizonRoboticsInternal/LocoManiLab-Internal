#!/usr/bin/env bash
# Bring up the G1's onboard camera publisher for a collection session.
#
# The robot ships with `composed_camera_server.service`, the sonic-era publisher.
# It and OrinVideoSender both want port 5555, and the service is `enabled`, so it
# comes back on every Orin reboot -- meaning "I stopped it last week" is not a
# state you can rely on. This stops it, then starts OrinVideoSender in its place.
#
# Both are idempotent: stopping an already-stopped service is a no-op, and an
# OrinVideoSender that is already listening is left alone rather than restarted
# (restarting would drop the headset's video link mid-session).
#
# Run from the collection PC; everything here happens over ssh on the Orin.
#
#   scripts/g1/start_orin_camera.sh --host user@robot-host
#                                   [--mode listen|headless|composed]
#                                   --listen robot-headset-address:13579
#                                   [--zmq 'tcp://*:5555'] [--sink-port 9999]
#                                   [--status] [--stop]
#
# Two different programs can publish the robot's camera on port 5555, and only
# one of them can have the ZED at a time -- so picking one means stopping the
# other, which is what --mode does:
#
#   listen    (default) OrinVideoSender waiting for the headset. Serves the
#             headset's video AND the zmq feed, but only once the headset
#             connects; see --mode headless.
#   headless  OrinVideoSender in its direct (--send) mode, so the capture loop
#             starts immediately with no headset involved. See below.
#   composed  composed_camera_server.service, the sonic-era publisher that ships
#             on the robot. It captures on its own with no headset and no sink,
#             and it opens the ZED BY SERIAL NUMBER (`--ego-view-device-id` in
#             the unit) -- swap the camera without editing the unit and it will
#             sit in a retry loop reporting CAMERA NOT DETECTED while holding
#             /dev/video0, which starves every other process too.
#
# --headless: publish frames WITHOUT a headset. In the default (--listen) mode
# OrinVideoSender only opens the ZED once the headset connects and asks for a
# stream -- until then the zmq port accepts subscribers and stays silent, which
# looks exactly like a dead camera server. --headless instead runs the sender in
# its direct mode (--send), which starts the capture loop at startup. That mode
# insists on a TCP peer for the H.264 stream, so this puts a discard sink on the
# Orin's loopback and points the sender at it; the encoded video goes nowhere and
# the zmq feed -- which is a separate fan-out of the same grabbed frame -- runs.
# Cost: 13579 is not served, so the operator gets no robot-eye view in the
# headset. Re-run without --headless to go back.
#
# The Orin's sudo needs a password for `systemctl stop`. Pass it in the
# environment -- never on the command line (it would land in shell history and
# in every `ps` listing on this machine):
#
#   export G1_SSH_PASSWORD=...
#   scripts/g1/start_orin_camera.sh --host user@robot-host --mode headless
#
# Without it the script still runs: it reports whether the old service is in the
# way and lets the caller decide, rather than failing the whole launch.
set -uo pipefail

G1_HOST="${G1_HOST:-}"
# Where OrinVideoSender listens for the headset's control connection. This is the
# ORIN's address on the headset's network, which can differ from the robot link
# the collection PC uses. Configure it for the selected robot.
ORIN_LISTEN="${ORIN_LISTEN:-}"
ORIN_ZMQ="${ORIN_ZMQ:-tcp://*:5555}"
ORIN_DIR="${ORIN_DIR:-~/XRoboToolkit-Orin-Video-Sender}"
LEGACY_SERVICE="${LEGACY_SERVICE:-composed_camera_server.service}"
SSH_TIMEOUT="${SSH_TIMEOUT:-15}"
# Loopback port for the --headless discard sink. Never leaves the Orin, so it
# only has to avoid colliding with something else on the robot's localhost.
SINK_PORT="${SINK_PORT:-9999}"
SINK_SCRIPT=/tmp/locomanilab_video_sink.py
# listen | headless | composed
CAMERA_MODE="${CAMERA_MODE:-listen}"

MODE=start
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host|--listen|--zmq|--dir|--mode|--sink-port)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[orin-camera] missing value for $1" >&2
        exit 2
      fi ;;
  esac
  case "$1" in
    --host)   G1_HOST="$2"; shift 2 ;;
    --listen) ORIN_LISTEN="$2"; shift 2 ;;
    --zmq)    ORIN_ZMQ="$2"; shift 2 ;;
    --dir)    ORIN_DIR="$2"; shift 2 ;;
    --headless) CAMERA_MODE=headless; shift ;;
    --mode)
      case "$2" in
        listen|headless|composed) CAMERA_MODE="$2" ;;
        *) echo "[orin-camera] --mode must be listen|headless|composed" >&2; exit 2 ;;
      esac
      shift 2 ;;
    --sink-port) SINK_PORT="$2"; shift 2 ;;
    --status) MODE=status; shift ;;
    --stop)   MODE=stop; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[orin-camera] unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { echo "[orin-camera] $*"; }

if [[ -z "$G1_HOST" ]]; then
  say "set --host user@robot-host or G1_HOST before connecting to a robot."
  exit 2
fi
if [[ "$MODE" == start && "$CAMERA_MODE" == listen && -z "$ORIN_LISTEN" ]]; then
  say "listen mode needs --listen address:port or ORIN_LISTEN on the headset network."
  exit 2
fi

# BatchMode: fail fast instead of hanging on a password prompt in a tmux pane
# nobody is watching. Key-based auth to the Orin is a prerequisite.
orin() {
  ssh -o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT" \
      -o StrictHostKeyChecking=accept-new "$G1_HOST" "$@"
}

if ! orin true 2>/dev/null; then
  say "cannot reach $G1_HOST over ssh (key auth required)."
  say "the camera is the robot's, so collection will have no images until this works."
  exit 1
fi

# Match the executable name only (pgrep -x against ./OrinVideoSender's comm),
# never the full command line: `pgrep -f OrinVideoSender` also matches the ssh
# command that carries the pattern, so it reports "running" against itself.
sender_running() { orin 'pgrep -x OrinVideoSender >/dev/null' 2>/dev/null; }

# The sender's own argv says which mode it is in; `pgrep -x -a` prints it without
# the self-match `pgrep -f` would suffer.
sender_argv() { orin 'pgrep -x -a OrinVideoSender | head -1' 2>/dev/null || true; }

sink_running() { orin "ss -ltn 2>/dev/null | grep -q '127.0.0.1:$SINK_PORT'"; }

# Who actually holds the camera port, as "<comm> <pid>" (empty if nobody).
#
# Asked of the port rather than of systemd on purpose: `systemctl is-active` only
# knows about the instance systemd itself started, and this script can launch the
# unit's ExecStart directly when there is no sudo password. Trusting is-active
# there reported "inactive" for a publisher that was serving fine, and a second
# --mode composed then started a DUPLICATE that could only fail (measured).
port_owner() {
  local port="${ORIN_ZMQ##*:}"
  # NOT `grep -w ':$port'`: the match starts at ':' but is preceded by a digit of
  # the address, so -w rejects it and every port looks free.
  orin "ss -ltnp 2>/dev/null | grep -E ':$port[[:space:]]'" 2>/dev/null \
    | sed -n 's/.*users:(("\([^"]*\)",pid=\([0-9]*\).*/\1 \2/p' | head -1
}

report_status() {
  local svc sender
  svc="$(orin "systemctl is-active $LEGACY_SERVICE" 2>/dev/null || true)"
  sender="$(sender_argv)"
  say "$LEGACY_SERVICE: ${svc:-unknown}"
  if [[ -n "$sender" ]]; then
    if [[ "$sender" == *--send* ]]; then
      say "OrinVideoSender: running (HEADLESS/--send) -- $sender"
    else
      say "OrinVideoSender: running (--listen; silent until the headset connects) -- $sender"
    fi
  else
    say "OrinVideoSender: not running"
  fi
  if sink_running; then
    say "discard sink: listening on 127.0.0.1:$SINK_PORT"
  fi
  local owner
  owner="$(port_owner)"
  if [[ -n "$owner" ]]; then
    say "camera port ${ORIN_ZMQ##*:}: held by $owner"
  else
    say "camera port ${ORIN_ZMQ##*:}: nobody is publishing"
  fi
}

# Whatever currently holds the ZED, other than our own sender. Found through the
# device node rather than a process name: `pkill -f <pattern>` over ssh matches
# the very command line carrying the pattern, and the remote shell kills itself.
camera_holders() {
  orin "fuser /dev/video0 2>/dev/null" | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
}

# --- Free the ZED --------------------------------------------------------
# Only one process can have the camera. The legacy service is the usual squatter,
# and it does not even need to succeed to block everyone else: with a serial that
# does not match the attached camera it sits in a retry loop holding /dev/video0,
# and every other process -- ours included -- then gets `count=0` out of the ZED
# SDK's device enumeration and reports CAMERA NOT DETECTED. Measured 2026-08-27.
release_camera() {
  local legacy stopped=0 pid comm
  legacy="$(orin "systemctl is-active $LEGACY_SERVICE" 2>/dev/null || true)"
  if [[ "$legacy" == "active" ]]; then
    if [[ -n "${G1_SSH_PASSWORD:-}" ]]; then
      say "stopping $LEGACY_SERVICE (it holds the camera and port 5555)"
      if printf '%s\n' "$G1_SSH_PASSWORD" \
           | orin "sudo -S -p '' systemctl stop $LEGACY_SERVICE" 2>/dev/null; then
        say "stopped."; stopped=1
      else
        say "WARNING: could not stop it (wrong G1_SSH_PASSWORD?)."
      fi
    else
      say "$LEGACY_SERVICE is active and holds the camera, but G1_SSH_PASSWORD is unset."
    fi
  else
    say "$LEGACY_SERVICE: ${legacy:-inactive} (nothing to stop)"
  fi

  # Fallback, and the reason `systemctl stop` is not the only path: the service
  # runs as the login user, so its main process can simply be signalled. systemd
  # counts SIGTERM as a clean exit, so `Restart=on-failure` does not resurrect it
  # -- which is how this works with no sudo password at all (measured).
  if [[ $stopped -eq 0 ]]; then
    for pid in $(camera_holders); do
      comm="$(orin "cat /proc/$pid/comm 2>/dev/null" || true)"
      [[ "$comm" == "OrinVideoSender" ]] && continue
      say "releasing the ZED from pid $pid ($comm)"
      orin "kill $pid" >/dev/null 2>&1 || say "  could not signal $pid (not ours? needs sudo)"
    done
  fi
}

start_sink() {
  if sink_running; then
    say "discard sink already listening on 127.0.0.1:$SINK_PORT"
    return 0
  fi
  say "starting the discard sink on 127.0.0.1:$SINK_PORT"
  # Written fresh each time: /tmp does not survive a reboot, and this keeps the
  # robot free of anything to install or keep in sync.
  orin "cat > $SINK_SCRIPT" <<'SINK'
import socket, sys
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", int(sys.argv[1])))
srv.listen(1)
print("sink listening", flush=True)
while True:
    conn, addr = srv.accept()
    print(f"accepted {addr}", flush=True)
    try:
        while conn.recv(1 << 16):
            pass
    except OSError as exc:
        print(f"recv error: {exc}", flush=True)
    finally:
        conn.close()
        print("peer closed", flush=True)
SINK
  orin "setsid nohup python3 $SINK_SCRIPT $SINK_PORT > /tmp/locomanilab_video_sink.log 2>&1 < /dev/null & disown" \
    >/dev/null 2>&1 || true
  sleep 1
  if sink_running; then
    return 0
  fi
  say "ERROR: the discard sink did not come up; --headless cannot stream."
  orin 'tail -5 /tmp/locomanilab_video_sink.log' 2>/dev/null || true
  return 1
}

case "$MODE" in
  status) report_status; exit 0 ;;
  stop)
    say "stopping OrinVideoSender on $G1_HOST"
    orin 'pkill -x OrinVideoSender' >/dev/null 2>&1 || true
    # The sink outlives the sender otherwise, holding a port for nothing. Killed
    # by the port it holds, not by `pkill -f <path>`: that pattern is inside the
    # ssh command line too, so the remote shell would kill itself.
    orin "fuser -k $SINK_PORT/tcp" >/dev/null 2>&1 || true
    sleep 2
    report_status
    exit 0
    ;;
esac

# --- The sonic-era publisher -----------------------------------------------
# The unit's own ExecStart is the source of truth for the interpreter, the module
# and above all the ZED serial number -- re-deriving any of that here would mean
# two places to edit when the camera is swapped.
composed_argv() {
  orin "systemctl show -p ExecStart --value $LEGACY_SERVICE 2>/dev/null" \
    | sed -n 's/.*argv\[\]=\(.*\) ; ignore_errors.*/\1/p'
}

composed_health() {
  local log
  log="$(orin "journalctl -u $LEGACY_SERVICE -n 40 --no-pager 2>/dev/null; tail -40 /tmp/composed_camera.log 2>/dev/null" || true)"
  if grep -q "Camera successfully opened" <<<"$log"; then
    say "  ZED opened -- publishing with no headset involved."
    grep -o "S/N [0-9]*" <<<"$log" | tail -1 | while read -r sn; do say "  $sn"; done
    return 0
  fi
  if grep -q "CAMERA NOT DETECTED" <<<"$log"; then
    say "  WARNING: CAMERA NOT DETECTED. This publisher opens the ZED BY SERIAL:"
    say "    $(composed_argv)"
    say "  If the attached camera is a different one, the serial in the unit is"
    say "  stale -- and note it holds /dev/video0 while retrying, which starves"
    say "  every other camera process too. Check the attached camera with:"
    say "    <venv>/bin/python -c 'import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])'"
    say "  (that returns 0 devices while this publisher holds them -- stop it first)"
    return 1
  fi
  say "  (no verdict yet in the log; check journalctl -u $LEGACY_SERVICE on the Orin)"
  return 0
}

stop_sender() {
  if [[ -n "$(sender_argv)" ]]; then
    say "stopping OrinVideoSender (it and $LEGACY_SERVICE cannot share the ZED)"
    orin 'pkill -x OrinVideoSender' >/dev/null 2>&1 || true
  fi
  orin "fuser -k $SINK_PORT/tcp" >/dev/null 2>&1 || true
  sleep 2
}

if [[ "$CAMERA_MODE" == "composed" ]]; then
  stop_sender
  state="$(orin "systemctl is-active $LEGACY_SERVICE" 2>/dev/null || true)"
  owner="$(port_owner)"
  if [[ "$state" == "active" || ( -n "$owner" && "$owner" != OrinVideoSender* ) ]]; then
    say "already publishing: ${owner:-$LEGACY_SERVICE (systemd: $state)}"
  else
    started=0
    if [[ -n "${G1_SSH_PASSWORD:-}" ]]; then
      say "starting $LEGACY_SERVICE"
      if printf '%s\n' "$G1_SSH_PASSWORD" \
           | orin "sudo -S -p '' systemctl start $LEGACY_SERVICE" 2>/dev/null; then
        say "started."; started=1
      else
        say "WARNING: systemctl start failed (wrong G1_SSH_PASSWORD?)."
      fi
    fi
    if [[ $started -eq 0 ]]; then
      # No sudo needed: the unit runs as the login user, so running its own
      # ExecStart directly gives a process identical to the one systemd starts.
      argv="$(composed_argv)"
      if [[ -z "$argv" ]]; then
        say "ERROR: could not read ExecStart of $LEGACY_SERVICE; start it on the Orin yourself."
        exit 1
      fi
      say "launching the unit's own command as the login user (no sudo required):"
      say "  $argv"
      orin "setsid nohup $argv > /tmp/composed_camera.log 2>&1 < /dev/null & disown" \
        >/dev/null 2>&1 || true
    fi
    sleep 5
  fi
  say "$LEGACY_SERVICE is the publisher now."
  say "  Port comes from the unit's own --port (not this script's --zmq)."
  composed_health || true
  say "  No headset video: 13579 is not served in this mode."
  exit 0
fi

# --- 1. Free the camera and port 5555 ---------------------------------------
release_camera

# --- 2. Start OrinVideoSender ----------------------------------------------
# A sender that is already up is left alone -- restarting it would drop the
# headset's video link mid-session -- but only if it is in the mode being asked
# for. Silently keeping a --listen sender when headless was asked for would hand
# back exactly the silent port headless mode exists to avoid.
running_argv="$(sender_argv)"
if [[ -n "$running_argv" ]]; then
  if [[ "$CAMERA_MODE" == "headless" && "$running_argv" != *--send* ]]; then
    say "OrinVideoSender is in --listen mode but headless was asked for; restarting it"
    orin 'pkill -x OrinVideoSender' >/dev/null 2>&1 || true
    sleep 2
  elif [[ "$CAMERA_MODE" == "listen" && "$running_argv" == *--send* ]]; then
    say "OrinVideoSender is headless (--send) but the headset mode was asked for; restarting it"
    orin 'pkill -x OrinVideoSender' >/dev/null 2>&1 || true
    orin "fuser -k $SINK_PORT/tcp" >/dev/null 2>&1 || true
    sleep 2
  else
    say "OrinVideoSender already running -- leaving it alone"
    say "  (restarting would drop the headset's video link; use --stop to force)"
    report_status
    exit 0
  fi
fi

if [[ "$CAMERA_MODE" == "headless" ]]; then
  start_sink || exit 1
  say "starting OrinVideoSender (headless: --send 127.0.0.1:$SINK_PORT, zmq=$ORIN_ZMQ)"
  launch="./OrinVideoSender --send --server 127.0.0.1 --port '$SINK_PORT'"
else
  say "starting OrinVideoSender (listen=$ORIN_LISTEN zmq=$ORIN_ZMQ)"
  launch="./OrinVideoSender --listen '$ORIN_LISTEN'"
fi

# setsid + </dev/null: without both, the process is killed when this ssh channel
# closes -- `nohup ... &` alone is not enough here (measured).
# OrinVideoSender is installed separately on the robot. Its publisher option is
# still named --soma-zmq; this launcher exposes it through the public --zmq flag.
orin "cd $ORIN_DIR && setsid nohup $launch \
        --soma-zmq '$ORIN_ZMQ' > /tmp/orin_sender.log 2>&1 < /dev/null & disown" \
  >/dev/null 2>&1 || true

for _ in 1 2 3 4 5 6 7 8 9 10; do
  sender_running && break
  sleep 1
done

if ! sender_running; then
  say "ERROR: OrinVideoSender did not come up. Last log lines:"
  orin 'tail -15 /tmp/orin_sender.log' 2>/dev/null || say "  (no log)"
  exit 1
fi

say "OrinVideoSender is up. Publishing 'ego_view' on ${ORIN_ZMQ/\*/$(echo "$G1_HOST" | cut -d@ -f2)}"
if [[ "$CAMERA_MODE" == "headless" ]]; then
  # The capture loop opens the ZED after the sink connects; give it a moment so
  # a camera it cannot open is reported here rather than as a silent port.
  sleep 4
  if orin 'grep -q "Camera successfully opened" /tmp/orin_sender.log' 2>/dev/null; then
    say "  ZED opened -- frames are flowing with no headset attached."
  else
    say "  WARNING: the ZED did not open. Last log lines:"
    orin 'grep -v "Logging level" /tmp/orin_sender.log | tail -8' 2>/dev/null || true
    say "  CAMERA NOT DETECTED here usually means something else still holds"
    say "  /dev/video0 -- check with 'fuser -v /dev/video0' on the Orin."
  fi
  say "  No headset video: 13579 is not served in this mode (use --mode listen)."
else
  say "  It captures only while the headset is connected -- until then 5555 is silent."
fi
say "  Log on the Orin: /tmp/orin_sender.log"
