"""Refuse to publish internal references that leaked in through a sync.

Vendored source carries docstrings written for an internal audience. Most are
harmless; some name roadmap documents, local paths or outreach sequencing. The
sync guards catch secrets and bulk data, not vocabulary, so this runs after it.

Usage:  python scripts/audit.py   (exit 1 on any finding)
"""
from __future__ import annotations
import re, sys
from pathlib import Path

# Findings quote the offending line; a Windows cp1252 console must not be able
# to crash the report mid-print.
sys.stdout.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync import SECRET_PATTERNS  # one pattern list, no drift between the two gates

REPO = Path(__file__).resolve().parent.parent
PATTERNS = [
    (re.compile(r"(?i)\bworkstream\b"), "internal roadmap reference"),
    (re.compile(r"(?i)\boutreach\b"), "commercial planning term"),
    (re.compile(r"(?i)\bprospect(s|ing)?\b"), "commercial planning term"),
    (re.compile(r"(?i)\bnot for publication\b"), "internal reviewer note"),
    (re.compile(r"(?i)\breviewer notes?\b"), "internal reviewer note"),
    (re.compile(r"(?i)\bpersona\b"), "internal publication-process term"),
    (re.compile(r"(?i)\bfunnel\b"), "commercial planning term"),
    (re.compile(r"(?i)\bcomms[ -]?pack\b"), "commercial planning document"),
    (re.compile(r"\bKW-\d{4}-"), "internal article-id scheme"),
    # Separator optional and either slash direction: notebook JSON carries
    # doubled backslashes, shell output sometimes forward slashes.
    (re.compile(r"(?i)[a-z]:[\\/]*work\b"), "local filesystem path"),
    (re.compile(r"(?i)[a-z]:[\\/]*users\b|\\{1,2}users\\"), "local user-profile path"),
    (re.compile(r"(?i)\bday rate\b|\binvoice\b|\bretainer\b"), "pricing term"),
    (re.compile(r"(?i)\bthornbury\b"), "fictional client name from an internal exercise"),
]
# Credential shapes come from sync.py's SECRET_PATTERNS. sync scans vendored
# files; notebooks and README/DATA files are authored in this repo and get
# their only credential check here, so the two gates must test the same shapes.
PATTERNS += [(re.compile(p.pattern.decode("ascii")), "credential shape") for p in SECRET_PATTERNS]
ALLOW = {Path("scripts/sync.py"), Path("scripts/audit.py")}

def candidates() -> list[Path]:
    """What git would publish: tracked plus untracked-but-not-ignored files.

    Local caches and rebuilt data are gitignored and never leave this machine;
    scanning them only produces noise. Falls back to a full walk without git.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True,
        ).stdout
        return [REPO / p.decode("utf-8") for p in out.split(b"\x00") if p]
    except (OSError, subprocess.CalledProcessError):
        return list(REPO.rglob("*"))


findings = []
for p in candidates():
    if not p.is_file() or ".git" in p.parts:
        continue
    if p.suffix not in {".py", ".md", ".txt", ".toml", ".csv", ".yaml", ".yml", ".ipynb", ".json",
                        ".html", ".cfg", ".ini", ".sh", ".ps1", ""}:
        continue
    rel = p.relative_to(REPO)
    if rel in ALLOW:
        continue
    try:
        raw = p.read_bytes()
    except OSError:
        continue
    text = raw.decode("utf-8", errors="ignore")
    if b"\x00" in raw:
        # A UTF-16 file (typical of Windows `>` redirection) decodes under lossy
        # UTF-8 into NUL-interleaved text that no pattern can match. Scan the
        # UTF-16 reading as well.
        text += "\n" + raw.decode("utf-16", errors="ignore")
    for i, line in enumerate(text.splitlines(), 1):
        for pat, why in PATTERNS:
            if pat.search(line):
                findings.append(f"{rel}:{i}  {why}\n      {line.strip()[:96]}")

if findings:
    print(f"AUDIT FAILED: {len(findings)} internal reference(s) found\n")
    for f in findings:
        print("  " + f)
    sys.exit(1)
print("audit clean: no internal references found")
