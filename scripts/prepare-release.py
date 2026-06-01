#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def assert_on_main() -> None:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "main":
        print(f"[ERROR] Expected branch 'main', got {branch!r}")
        sys.exit(1)


def assert_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        print("[ERROR] Working tree is not clean")
        print(status)
        sys.exit(1)


def assert_tag_absent(version: str) -> None:
    tag = f"v{version}"
    local = subprocess.run(
        ["git", "tag", "--list", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local:
        print(f"[ERROR] Local tag already exists: {tag}")
        sys.exit(1)


def update_pyproject(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{version}"', text, count=1)
    if count != 1:
        print("[ERROR] Could not update pyproject.toml version")
        sys.exit(1)
    PYPROJECT.write_text(updated, encoding="utf-8")


def update_changelog(version: str) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = "## Unreleased\n"
    if marker not in text:
        print("[ERROR] CHANGELOG.md is missing '## Unreleased'")
        sys.exit(1)
    if f"## {version}\n" in text:
        print(f"[ERROR] CHANGELOG.md already has section {version}")
        sys.exit(1)
    CHANGELOG.write_text(text.replace(marker, f"{marker}\n## {version}\n", 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a patch release.")
    parser.add_argument("version", help="Version like 0.1.4")
    parser.add_argument("--skip-clean-check", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        print("[ERROR] Version must look like 0.1.4")
        sys.exit(1)

    assert_on_main()
    if not args.skip_clean_check:
        assert_clean_worktree()
    assert_tag_absent(args.version)
    update_pyproject(args.version)
    update_changelog(args.version)
    run(["uv", "lock"])
    run(["bash", "scripts/validate-release.sh"])

    print("Release files prepared. Review git diff, then commit and push main.")


if __name__ == "__main__":
    main()
