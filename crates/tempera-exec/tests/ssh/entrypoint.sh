#!/bin/sh
set -eu
: "${AUTHORIZED_KEY:?AUTHORIZED_KEY must hold the test public key}"
install -d -m 700 -o tempera -g tempera /home/tempera/.ssh
printf '%s\n' "$AUTHORIZED_KEY" > /home/tempera/.ssh/authorized_keys
chown tempera:tempera /home/tempera/.ssh/authorized_keys
chmod 600 /home/tempera/.ssh/authorized_keys
exec /usr/sbin/sshd -D -e
