#!/usr/bin/env bash
# Bundle training outputs BEFORE you terminate the pod -- the container disk is wiped
# on terminate, so your checkpoints vanish if you don't pull them off first.
# Creates one tarball and prints the scp command to run from your laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="augur-results-$STAMP.tar.gz"

if [ -d checkpoints ]; then
  tar -czf "$ARCHIVE" outputs/ checkpoints/
else
  tar -czf "$ARCHIVE" outputs/
fi

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo ">> Bundled results: $(pwd)/$ARCHIVE ($SIZE)"
echo ">> From your LAPTOP, pull it (SSH details are in the pod's Connect tab):"
echo "     scp -P <PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>:$(pwd)/$ARCHIVE ."
echo ">> Do this BEFORE terminating the pod."
