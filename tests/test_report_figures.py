"""Nothing may reference a figure that was never built.

RESULTS.md embeds three figures. An `![...](figures/x.png)` pointing at a file that is not
there renders as a broken image on GitHub and nobody notices until a reader complains -- so
the reference and the file are checked against each other here.

The course report is written outside this repository. If a local copy happens to be present
its `\\includegraphics` targets are checked too, since that is exactly the breakage that
prompted this test: the report once cited three figures that did not exist and so could not
be compiled at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "RESULTS.md"]


@pytest.mark.parametrize("doc", DOCS)
def test_every_embedded_image_exists(doc: str) -> None:
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not in this checkout")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text())
    missing = [r for r in refs
               if not r.startswith(("http://", "https://")) and not (ROOT / r).exists()]
    assert not missing, f"{doc} embeds images that do not exist: {missing}"


def test_results_actually_shows_the_figures() -> None:
    """They are the one visual summary the repository carries; losing them is silent."""
    results = ROOT / "RESULTS.md"
    if not results.exists():
        pytest.skip("RESULTS.md not in this checkout")
    refs = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", results.read_text())
    assert len(refs) >= 3, f"expected the three summary figures, found {refs}"


def test_local_report_figures_if_the_report_is_present() -> None:
    """The report is gitignored; this only fires for whoever is building it."""
    tex = ROOT / "FINAL_REPORT.tex"
    if not tex.exists():
        pytest.skip("FINAL_REPORT.tex is not part of this repository")
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.read_text().splitlines())
    included = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body)
    assert included, "the report includes no figures at all"
    missing = [f for f in included if not (ROOT / f).exists()]
    assert not missing, f"the report references figures that do not exist: {missing}"
