"""Vendor analysis code and derived results into this public repository.

The working copies live outside this repo. This script copies from them using an
explicit ALLOWLIST, so a file is published only if something here names it. That
is deliberate: the source trees sit next to directories holding client material,
outreach plans and pricing, and an exclude-list would publish anything anyone
forgot to exclude.

Three guards run on every file before it is written:

  1. Allowlist       nothing copies unless a rule names it.
  2. Size ceiling    refuses anything over MAX_BYTES. Raw third-party data pulls
                     are large, and the ceiling catches them even if a glob is
                     written too loosely.
  3. Secret scan     refuses anything matching a credential pattern.

Raw data is never vendored. Yahoo Finance terms do not permit redistributing
their price data, and the ENTSO-E pulls are both large and better re-fetched
than mirrored. Every analysis ships the script that fetches its own inputs.

Usage:
    python scripts/sync.py --check     report what would change, write nothing
    python scripts/sync.py             perform the copy
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO.parent

MAX_BYTES = 2_000_000

SECRET_PATTERNS = [
    # Named credential assignment. `token` is the alternation that matters: the
    # one real credential these analyses use is ENTSOE_API_TOKEN, which contains
    # neither "key" nor "secret". The value may be quoted or bare.
    re.compile(rb"(?i)(api[_-]?key|secret|passwd|password|bearer|token|credential)\s*[:=]\s*['\"]?[A-Za-z0-9/+_-]{16,}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"(?i)aws_secret_access_key\s*[:=]"),
    # Well-known token shapes, for a bare credential with no telltale name.
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),
    # Boundary guard: without it this matches inside prose like "risk-backtest".
    re.compile(rb"(?<![A-Za-z])sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}"),
    # Bare UUID: the shape of an ENTSO-E token with no telltale name next to
    # it, e.g. passed positionally or pasted into a comment. Nothing published
    # here legitimately contains a UUID, so a false trip only costs a look.
    re.compile(rb"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
]

# (source root, destination under analyses/, [glob rules relative to source root])
#
# Rules are globs. Directories are walked; __pycache__ and dotfiles are always
# skipped. Keep every rule as narrow as the analysis allows.
MANIFEST: list[tuple[str, str, list[str]]] = [
    (
        "calibration-audit",
        "german-grid-forecast-calibration",
        [
            "src/*.py",
            # No report/draft.md: working drafts carry reviewer notes written
            # for an internal audience. The published PDF is authored in this
            # repository from the live article.
            # No parquet. Derived results are rebuilt from the pull scripts;
            # see the analysis DATA.md. Shipping them made the repo a partial
            # mirror of ENTSO-E for no reproducibility gain.
        ],
    ),
    (
        "var-backtest/backtest-uq",
        "var-backtest-tail-risk",
        [
            "CHANGELOG.md",
            "LICENSE",
            "pyproject.toml",
            "src/**/*.py",
            "scripts/*.py",
            "tests/**/*.py",
            # Aggregate results and the null-sensitivity evidence behind the
            # article. details.csv is row-level and derived from Yahoo prices,
            # so it is rebuilt rather than shipped.
            "output/summary.csv",
            "experiments/factorial/*/as_null_sensitivity.csv",
        ],
    ),
    (
        "aero/audit",
        "tool-wear-calibration",
        [
            "src/*.py",
            # No report/*.md: working drafts carry reviewer notes written for
            # an internal audience. The published PDF is authored in this
            # repository from the live article.
            # artifacts/raw is the 98 MB NASA mat-file set; artifacts/interim is
            # rebuilt by the pipeline. Figure data only.
            "artifacts/figures/*.json",
        ],
    ),
]

SKIP_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".venv"}

# Per-analysis README.md files are authored in this repository and never
# vendored. The upstream ones reference internal roadmap documents, local
# Windows paths and outreach sequencing, none of which belongs in public.

# Files authored directly in this repository rather than vendored. A file in a
# destination tree that neither a manifest rule nor one of these accounts for
# is stale: usually something vendored under a rule that was later narrowed or
# removed, which would otherwise stay published forever with no signal.
# fnmatch semantics, so `*` crosses path separators.
AUTHORED_HERE = ["README.md", "DATA.md", "notebooks/*", "report/*.pdf"]


def scan(path: Path) -> str | None:
    """Return a refusal reason, or None if the file may be published."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        return f"over size ceiling ({size:,} > {MAX_BYTES:,} bytes)"
    # Binary files are scanned too: the patterns are byte regexes, a PDF or
    # parquet can embed strings, and a false trip only costs an inspection.
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return f"unreadable: {exc}"
    for pat in SECRET_PATTERNS:
        if pat.search(blob):
            return f"matches credential pattern {pat.pattern[:38]!r}"
    return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def tracked_under(dest_root: Path) -> list[Path]:
    """Git-tracked files under dest_root; empty if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", str(dest_root)],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        print(f"  warning: git ls-files failed; stale check skipped for {dest_root.name}")
        return []
    return [REPO / p.decode("utf-8") for p in out.split(b"\x00") if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    copied = skipped = unchanged = 0
    refusals: list[str] = []
    missing: list[str] = []
    stale: list[str] = []

    for src_rel, dest_rel, rules in MANIFEST:
        src_root = WORK / src_rel
        dest_root = REPO / "analyses" / dest_rel
        if not src_root.is_dir():
            missing.append(f"{src_rel} (source tree not found)")
            continue
        expected: set[Path] = set()

        for rule in rules:
            matches = sorted(src_root.glob(rule))
            if not matches:
                missing.append(f"{src_rel}/{rule}")
                continue
            for src in matches:
                expected.add(dest_root / src.relative_to(src_root))
                if src.is_symlink() or not src.is_file():
                    continue
                if SKIP_PARTS & set(src.parts):
                    continue
                # Dotfiles never publish: `src/*.py` would otherwise happily
                # match a stray `.secrets.py`.
                if any(part.startswith(".") for part in src.relative_to(src_root).parts):
                    continue
                reason = scan(src)
                if reason:
                    refusals.append(f"{src.relative_to(WORK)} -> {reason}")
                    skipped += 1
                    continue
                dest = dest_root / src.relative_to(src_root)
                if dest.exists() and digest(dest) == digest(src):
                    unchanged += 1
                    continue
                if not args.check:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                copied += 1
                print(f"  {'would copy' if args.check else 'copied'}  {dest.relative_to(REPO)}")

        # Stale check: every *published* file in the destination must be
        # accounted for by a rule match or an authored-here pattern. Published
        # means git-tracked — the working copy also holds gitignored local
        # rebuilds (caches, parquet pulls), which are nobody's business here.
        for pub in tracked_under(dest_root):
            rel = pub.relative_to(dest_root).as_posix()
            if pub in expected or any(fnmatch.fnmatch(rel, g) for g in AUTHORED_HERE):
                continue
            stale.append(f"{pub.relative_to(REPO)} (no manifest rule accounts for it)")

    print(f"\n{'would copy' if args.check else 'copied'}: {copied} | unchanged: {unchanged} | refused: {skipped}")
    if refusals:
        print("\nREFUSED (guard tripped, nothing written):")
        for r in refusals:
            print(f"  - {r}")
    if missing:
        print("\nNO MATCH (rule matched nothing; source may have moved):")
        for m in missing:
            print(f"  - {m}")
    if stale:
        print("\nSTALE (published, but no rule names it any more — delete or re-add a rule):")
        for s in stale:
            print(f"  - {s}")
    # A tripped guard is a real signal, not noise. Fail loudly.
    return 1 if refusals or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
