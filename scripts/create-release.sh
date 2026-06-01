#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" || ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Usage: scripts/create-release.sh 0.1.4"
  exit 1
fi

TAG="v$VERSION"
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "[ERROR] Expected main branch, got $BRANCH"
  exit 1
fi

git fetch origin main --tags
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "[ERROR] Local main is not aligned with origin/main"
  exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1 || git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
  echo "[ERROR] Tag already exists: $TAG"
  exit 1
fi

echo "==> Recent main workflow runs"
gh run list --branch main --limit 10
echo ""
read -r -p "Create and push $TAG, then create GitHub release? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

notes_file="$(mktemp)"
python - "$VERSION" > "$notes_file" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

version = sys.argv[1]
text = Path("CHANGELOG.md").read_text(encoding="utf-8")
start = text.index(f"## {version}")
end = text.find("\n## ", start + 1)
print((text[start:] if end == -1 else text[start:end]).strip())
PY

git tag "$TAG"
git push origin "$TAG"
gh release create "$TAG" --title "$TAG" --notes-file "$notes_file"
echo "Created release $TAG"
