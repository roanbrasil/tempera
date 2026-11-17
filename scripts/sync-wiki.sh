#!/usr/bin/env bash
# Sincroniza docs/wiki -> wiki do GitHub.
# Pré-requisito (uma vez): criar a primeira página em https://github.com/roanbrasil/tempera/wiki
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
  for f in Home Visao-e-Problema Jornada-do-Usuario Arquitetura Auditoria-de-Dados Catalogo-de-Algoritmos \
           Execucao-e-Maquinas Custo-e-Estimativa Contratos Seguranca Stack-Tecnologica Roadmap Glossario Contribuindo; do
    echo "- [$(head -1 "$f.md" | sed 's/^# //')]($f)"
  done
  echo "- [ADRs](https://github.com/roanbrasil/tempera/tree/main/docs/adr)"
} > _Sidebar.md
git add -A
git -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" \
  commit -q -m "sync: docs/wiki" || { echo "nada a sincronizar"; exit 0; }
git push -q
echo "wiki sincronizada"
