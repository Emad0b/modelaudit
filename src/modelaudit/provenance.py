import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def capture(output):
    package = Path(__file__).parent
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.glob("*.py"))
    }
    data = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={
            name: version(name) for name in ["numpy", "scipy", "scikit-learn"]
        },
        source_sha256=files,
        note="Seeded synthetic demo; no real-world performance claim",
    )
    (Path(output) / "provenance.json").write_text(json.dumps(data, indent=2))
