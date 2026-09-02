"""The site renders a copy of these artifacts; it must be the same copy.

A static deploy cannot read across repositories, so the specimen page carries
its own copy of artifacts/specimen. Keeping the two in step by hand is what let
the published lineage digest drift from the one in this repository, which is the
single thing a lineage row must never do. This test fails locally the moment
they diverge, and skips anywhere the site checkout is absent.
"""

import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
REPO_COPY = HERE / "artifacts" / "specimen"
SITE_COPY = pathlib.Path(r"c:/work/BLOG/apps/site/src/routes/specimen/data")


@pytest.mark.skipif(not SITE_COPY.exists(), reason="site checkout not present")
@pytest.mark.skipif(not (REPO_COPY / "meta.json").exists(), reason="specimen not generated")
def test_published_digest_matches_the_repository():
    repo = json.loads((REPO_COPY / "meta.json").read_text(encoding="utf-8"))
    site = json.loads((SITE_COPY / "meta.json").read_text(encoding="utf-8"))
    assert site["artifact_digest_sha256"] == repo["artifact_digest_sha256"], (
        "the specimen page is rendering a different run than this repository holds; "
        "re-run audit_specimen.py with --publish-to")


@pytest.mark.skipif(not SITE_COPY.exists(), reason="site checkout not present")
@pytest.mark.skipif(not (REPO_COPY / "meta.json").exists(), reason="specimen not generated")
def test_every_published_artifact_is_byte_identical():
    for f in sorted(REPO_COPY.glob("*.json")):
        twin = SITE_COPY / f.name
        assert twin.exists(), f"{f.name} is missing from the site copy"
        assert twin.read_bytes() == f.read_bytes(), f"{f.name} differs between repo and site"
