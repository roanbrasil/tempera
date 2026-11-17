#!/usr/bin/env bash
# Syncs docs/wiki -> GitHub wiki.
# Prerequisite (once): create the first page at https://github.com/roanbrasil/tempera/wiki
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
git clone -q https://github.com/roanbrasil/tempera.wiki.git "$TMP"
find "$TMP" -maxdepth 1 -name '*.md' -delete
cp "$ROOT"/docs/wiki/*.md "$TMP"/
cd "$TMP"
sed -i -E 's/\]\(([A-Za-z-]+)\.md\)/](\1)/g; s#\]\(\.\./adr/([^)]+)\)#](https://github.com/roanbrasil/tempera/blob/main/docs/adr/\1)#g' *.md
{
  echo "**Têmpera**"
  for f in Home Vision-and-Problem User-Journey Architecture Data-Audit Algorithm-Catalog \
           Execution-and-Machines Cost-and-Estimation Contracts Security Tech-Stack Roadmap Glossary Contributing; do
    echo "- [$(head -1 "$f.md" | sed 's/^# //')]($f)"
  done
  echo "- [ADRs](https://github.com/roanbrasil/tempera/tree/main/docs/adr)"
} > _Sidebar.md
git add -A
git -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" \
  commit -q -m "sync: docs/wiki" || { echo "nothing to sync"; exit 0; }
git push -q
echo "wiki synced"
