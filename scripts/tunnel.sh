#!/usr/bin/env bash
# tunnel.sh — keep one SSH tunnel up, reconnecting forever (autossh-style). The RL batch (scripts/rl_batch.py) runs one
# per server machine on the RL machine, so the runs reach the probe server and the judges at 127.0.0.1.
#
#   setsid nohup bash scripts/tunnel.sh STATE_DIR NAME > /dev/null 2>&1 < /dev/null &
#
# STATE_DIR/NAME.conf is the tunnel, as shell assignments the loop reads again before every connection, so the driver
# can point it at another machine (a replaced server) by rewriting it and killing the live ssh:
#   TARGET=root@ssh5.vast.ai  PORT=31234  KEY=/mnt/ram/rl-batch/tunnel_key  KNOWN_HOSTS=NAME.<instance>.known_hosts
#   FORWARDS='8178:127.0.0.1:8178'   (one -L spec each, separated by spaces)
# (a relative KNOWN_HOSTS is in STATE_DIR: one per instance, since a vast.ai proxy port is reused by later instances)
# The loop writes STATE_DIR/NAME.pid (its own pid), NAME.ssh.pid (the live ssh's), NAME.state (connected or down, since
# when) and NAME.log (one line per event, ssh's own errors too).
#
# Keepalive: ServerAliveInterval 10 × ServerAliveCountMax 3, so a dead link ends ssh within ~30 s; the forwarded
# connections then close, and the clients' retries (the probe monitor's, the judges') reconnect through the next ssh.
# ExitOnForwardFailure: an ssh that can't open its ports exits instead of idling. Between attempts it waits 2 s, then
# twice as long each time, at most 30 s, back to 2 s after a connection that lasted a minute. SIGTERM stops the loop
# and its ssh.
set -u
dir=$1
name=$2
conf="$dir/$name.conf"
logf="$dir/$name.log"
child=""
mkdir -p "$dir"
echo $$ > "$dir/$name.pid"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$logf"; }
stop() {
  say "tunnel loop stopped (signal)"
  [ -n "$child" ] && kill "$child" 2>/dev/null
  rm -f "$dir/$name.ssh.pid" "$dir/$name.pid"
  echo "stopped since $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dir/$name.state"
  exit 0
}
trap stop TERM INT HUP
say "tunnel loop started (pid $$)"
delay=2
while :; do
  if [ ! -r "$conf" ]; then
    say "no $conf: stopping"
    rm -f "$dir/$name.pid"
    exit 1
  fi
  TARGET="" PORT="" KEY="" KNOWN_HOSTS="" FORWARDS=""
  # shellcheck source=/dev/null
  . "$conf"
  case "$KNOWN_HOSTS" in /*) ;; *) KNOWN_HOSTS="$dir/${KNOWN_HOSTS:-$name.known_hosts}" ;; esac
  forwards=()
  for f in $FORWARDS; do forwards+=(-L "$f"); done
  t0=$(date +%s)
  say "connecting: ssh -p $PORT $TARGET, forwarding $FORWARDS"
  ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" -o IdentitiesOnly=yes -o "IdentityFile=$KEY" -o LogLevel=ERROR \
    -p "$PORT" "${forwards[@]}" "$TARGET" < /dev/null 2>> "$logf" &
  child=$!
  echo "$child" > "$dir/$name.ssh.pid"
  echo "ssh running since $(date -u +%Y-%m-%dT%H:%M:%SZ) (pid $child)" > "$dir/$name.state"  # the ports tell if it is up
  wait "$child"
  rc=$?
  child=""
  rm -f "$dir/$name.ssh.pid"
  lasted=$(( $(date +%s) - t0 ))
  [ "$lasted" -ge 60 ] && delay=2
  echo "down since $(date -u +%Y-%m-%dT%H:%M:%SZ) (ssh exited $rc after ${lasted}s)" > "$dir/$name.state"
  say "ssh exited $rc after ${lasted}s; reconnecting in ${delay}s"
  sleep "$delay"
  delay=$(( delay * 2 > 30 ? 30 : delay * 2 ))
done
