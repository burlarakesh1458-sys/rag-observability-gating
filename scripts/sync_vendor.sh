#!/usr/bin/env bash
# Re-vendor the DroneDocs RAG package from a sibling checkout (git-subtree style copy).
# Usage: scripts/sync_vendor.sh ../01-production-rag
# Copies only the package, configs, golden eval set and raw corpus (no tests, no index, no reports).
set -euo pipefail
SRC="$(cd "${1:-../01-production-rag}" && pwd)"
DST="$(cd "$(dirname "$0")/.." && pwd)/vendor/dronedocs-rag"
INDEX_BACKUP=""
if [ -d "$DST/data/index" ]; then INDEX_BACKUP="$(mktemp -d)"; mv "$DST/data/index" "$INDEX_BACKUP/"; fi
rm -rf "$DST" && mkdir -p "$DST/data"
cp -r "$SRC/src" "$SRC/configs" "$SRC/eval" "$SRC/pyproject.toml" "$SRC/requirements.txt" "$SRC/LICENSE" "$DST/"
cp -r "$SRC/data/raw" "$SRC/data/SOURCES.md" "$DST/data/"
find "$DST" -name "__pycache__" -type d -prune -exec rm -rf {} + ; rm -rf "$DST"/src/*.egg-info
if [ -n "$INDEX_BACKUP" ]; then mv "$INDEX_BACKUP/index" "$DST/data/index"; rmdir "$INDEX_BACKUP"; fi
COMMIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo "uncommitted")
printf '# Vendored: dronedocs-rag\n\nCopied from `01-production-rag` at commit `%s` by `scripts/sync_vendor.sh`.\nDo not edit here: change the upstream repo and re-sync. Only the package, configs, golden eval set\nand raw corpus are vendored (no tests, no index, no reports).\n' "$COMMIT" > "$DST/VENDORED.md"
echo "vendored $SRC@$COMMIT -> $DST"
