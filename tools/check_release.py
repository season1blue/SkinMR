#!/usr/bin/env python3
"""Validate the allowlisted upload tree, not private local runtime outputs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", "runs", "data", "datasets", "models", "checkpoints", "private", "__pycache__", ".venv", ".venv-skincd", ".venv-skinir", "venv"}
FORBIDDEN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".dcm", ".dicom", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".log", ".pid", ".pem", ".key", ".p12", ".zip", ".gz"}
SECRET_PATTERNS = [re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
                   re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
                   re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
                   re.compile(r"\bAKIA[0-9A-Z]{16}\b")]
MACHINE_PATH = re.compile(r"/(?:root/experiment-models|data/(?:phase7|ssz|mini)|Aaron_Disk|merge/SkinVL)")


def main():
    manifest_path = ROOT / "provenance/release_files.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing final release allowlist")
    hashes = json.loads(manifest_path.read_text())
    errors = []
    for relative, expected in hashes.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            errors.append(f"Missing or symbolic-link file: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"Hash mismatch: {relative}")
    count = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in IGNORED_DIRS or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            errors.append(f"Symlink must not be published: {relative}")
            continue
        if not path.is_file():
            continue
        count += 1
        if str(relative) not in hashes and path != manifest_path:
            errors.append(f"Unlisted upload file: {relative}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS or path.stat().st_size >= 50_000_000:
            errors.append(f"Forbidden format or large file: {relative}")
        if path.suffix == ".csv" and str(relative) != "results/summary.csv":
            errors.append(f"Unapproved CSV: {relative}")
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Possible credential: {relative}")
        if MACHINE_PATH.search(text):
            errors.append(f"Machine-specific path: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as exc:
                errors.append(f"Syntax error: {relative}:{exc.lineno}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"PASS: {count} allowlisted text files; hashes, Python syntax, formats and selected sensitive patterns checked.")
    print("No inference run. Manual privacy and license review are still required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
