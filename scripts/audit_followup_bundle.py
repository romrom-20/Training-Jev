"""Verify and audit the portable bundle without model downloads or large caches."""

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from analyze_relevance import analyze
from audit_fresh_templates import audit
from score_transport import analyze as transport


def check(bundle):
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest, name
    with tempfile.TemporaryDirectory(prefix="latent-decisions-audit-") as temporary:
        root = Path(temporary)
        for folder in ("remapping", "fresh"):
            for source in (bundle / folder).rglob("*"):
                if not source.is_file():
                    continue
                dest = root / source.relative_to(bundle)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".gz":
                    dest = dest.with_suffix("")
                    dest.write_bytes(gzip.decompress(source.read_bytes()))
                else:
                    shutil.copy2(source, dest)
        for model in ("qwen-0.5b", "qwen-1.5b"):
            result = analyze(root / "remapping" / model)
            print(model, result["audit"])
            expected = json.loads((root / "remapping" / model / "score-transport.json").read_text())
            recomputed = transport(root / "remapping" / model)
            assert expected == recomputed, "Score-transport reconstruction mismatch"
            audit(root / "remapping" / model, root / "fresh" / model)
    print("Portable bundle audit passed; no target weights or activations required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, nargs="?", default=Path("results/followup"))
    check(parser.parse_args().bundle)
