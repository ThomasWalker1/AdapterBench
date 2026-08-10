"""Website build smoke tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"


def test_build_website_generates_index_and_codec_pages():
    script = ROOT / "scripts" / "build_website.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    index = (WEBSITE / "index.html").read_text()
    assert "AdapterBench" in index
    assert "leaderboards" in index
    assert "T2A — task-conditioned" in index
    assert (WEBSITE / "codecs" / "lora.html").exists()
    lora = (WEBSITE / "codecs" / "lora.html").read_text()
    assert "codec-link" not in lora  # detail page, not leaderboard row
    assert "T2A headline" in lora
    assert "D2A headline" in lora
    assert 'href="../index.html#leaderboards"' in lora
