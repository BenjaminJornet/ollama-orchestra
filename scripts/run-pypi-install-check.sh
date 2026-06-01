#!/usr/bin/env bash
set -euo pipefail

gh workflow run install.yml --ref main
sleep 5
run_id="$(gh run list --workflow "Install from PyPI" --limit 1 --json databaseId --jq '.[0].databaseId')"
if [[ -z "$run_id" || "$run_id" == "null" ]]; then
  echo "[ERROR] Could not find Install from PyPI run"
  exit 1
fi
gh run watch "$run_id" --exit-status
