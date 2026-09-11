"""Links between README.md and RESULTS.md must resolve.

The two were one file until the write-up moved out, and every cross-reference between them
was rewritten by hand at that point. A dead `#anchor` renders as a link that silently does
nothing on GitHub, which is exactly the kind of breakage nobody notices.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "RESULTS.md", "PROTOCOL.md"]


def _slug(heading: str) -> str:
    """GitHub's anchor rule: lowercase, drop punctuation, each space becomes a hyphen.

    Runs of spaces are *not* collapsed -- "oracle -- is" leaves a double space once the dash
    goes, and so anchors as "oracle--is".
    """
    cleaned = re.sub(r"[^\w\s-]", "", heading.strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s", "-", cleaned)


def _anchors(path: Path) -> set[str]:
    out = set()
    for line in path.read_text().splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.add(_slug(m.group(2)))
    return out


def _links(path: Path):
    """Markdown links, with fenced code stripped so shell comments are not scanned."""
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)


@pytest.mark.parametrize("doc", DOCS)
def test_links_resolve(doc: str) -> None:
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not in this checkout")
    anchors = {d: _anchors(ROOT / d) for d in DOCS if (ROOT / d).exists()}
    broken = []
    for label, target in _links(path):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        if file_part and not (ROOT / file_part).exists():
            broken.append(f"[{label}] -> {target} (no such path)")
            continue
        if anchor:
            pool = anchors.get(file_part or doc)
            if pool is not None and anchor not in pool:
                broken.append(f"[{label}] -> {target} (no such heading)")
    assert not broken, f"{doc}: " + "; ".join(broken)


def test_readme_points_at_the_write_up_rather_than_containing_it() -> None:
    """The README is orientation and reproduction; the detail lives in RESULTS.md."""
    readme = (ROOT / "README.md").read_text()
    assert "RESULTS.md" in readme, "the README must point at the detailed write-up"
    assert len(readme.splitlines()) < 400, "the README has grown back into a lab notebook"
