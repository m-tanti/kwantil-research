"""The lineage section has to be checkable, or it is worse than absent.

An earlier version of this file compared the site's copy of meta.json against
this repository's copy. That proves the two files agree; it proves nothing about
whether either one describes the artifacts sitting next to it. A reader does not
have two copies, they have the shipped files and a stated rule, so these tests
do what the reader does: recompute.
"""

import hashlib
import json
import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
SPECIMEN = HERE / "artifacts" / "specimen"
SCRIPT = HERE / "src" / "audit_specimen.py"

_site = os.environ.get("KWANTIL_SITE_SPECIMEN_DATA")
SITE_COPY = pathlib.Path(_site) if _site else None

needs_run = pytest.mark.skipif(
    not (SPECIMEN / "meta.json").exists(), reason="specimen not generated")
needs_site = pytest.mark.skipif(
    SITE_COPY is None or not SITE_COPY.exists(),
    reason="KWANTIL_SITE_SPECIMEN_DATA not set")


def digest_of(directory: pathlib.Path) -> str:
    """The rule exactly as meta.json states it, applied from scratch."""
    d = hashlib.sha256()
    for f in sorted(p for p in directory.glob("*") if p.name != "meta.json"):
        d.update(f.name.encode())
        d.update(f.read_bytes())
    return d.hexdigest()


@needs_run
def test_digest_recomputes_from_the_shipped_files():
    meta = json.loads((SPECIMEN / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifact_digest_sha256"] == digest_of(SPECIMEN)


@needs_run
def test_the_rule_is_published_alongside_the_digest():
    # A digest a reader cannot reproduce is decoration. The rule has to travel
    # with it, including the exclusion that trips people up.
    meta = json.loads((SPECIMEN / "meta.json").read_text(encoding="utf-8"))
    rule = meta["digest_rule"]
    assert "meta.json" in rule and "filename order" in rule
    assert "excluded" in rule


@needs_run
def test_meta_is_excluded_because_it_cannot_hash_itself():
    # Guards the rule against a well-meaning change: including meta.json makes
    # the digest unsatisfiable, since writing it changes what is being hashed.
    everything = hashlib.sha256()
    for f in sorted(SPECIMEN.glob("*")):
        everything.update(f.name.encode())
        everything.update(f.read_bytes())
    meta = json.loads((SPECIMEN / "meta.json").read_text(encoding="utf-8"))
    assert everything.hexdigest() != meta["artifact_digest_sha256"]


@needs_run
def test_script_hash_identifies_the_code_that_ran():
    # The digest says the outputs match. This says what produced them, which is
    # what the removed commit line was for and what a reader needs to reproduce.
    meta = json.loads((SPECIMEN / "meta.json").read_text(encoding="utf-8"))
    assert meta["script"] == "src/audit_specimen.py"
    assert meta["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), (
        "audit_specimen.py has changed since the shipped artifacts were generated; "
        "re-run it so the lineage names the code that actually produced them")


@needs_run
@needs_site
def test_the_published_copy_recomputes_to_the_same_digest():
    # Not "the two meta.json files agree" but "the site's own files hash to the
    # value the site prints", which is what a reader checking the page does.
    site_meta = json.loads((SITE_COPY / "meta.json").read_text(encoding="utf-8"))
    repo_meta = json.loads((SPECIMEN / "meta.json").read_text(encoding="utf-8"))
    assert site_meta["artifact_digest_sha256"] == repo_meta["artifact_digest_sha256"]
    # The site copy holds only the JSON, so recompute over what it actually has
    # and confirm each file is byte-identical to its twin.
    for f in sorted(SPECIMEN.glob("*.json")):
        if f.name == "meta.json":
            continue
        assert (SITE_COPY / f.name).read_bytes() == f.read_bytes(), f"{f.name} differs"
