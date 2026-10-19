from .provenance import capture
import argparse
import json
from pathlib import Path
from .core import audit, fixture


def markdown(name, result):
    text = f"# {name}\n\nDecision: **{result['decision']}**\n\n"
    r = result["metrics"]
    text += f"AUC {r['auc']:.3f}, Brier {r['brier']:.3f}, ECE {r['ece']:.3f}, baseline Brier {r['baseline_brier']:.3f}.\n\n"
    text += "Findings:\n\n" + "".join(
        f"- {f['severity']} â€” {f['code']}: {f['detail']}\n"
        for f in result["findings"]
    )
    return text + "\n" + result["scope"] + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON bundle; see reports/healthy_input.json for schema",
    )
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bundles = (
        {"external": json.loads(args.input.read_text())}
        if args.input
        else {"healthy": fixture(), "flawed": fixture(flawed=True)}
    )
    for name, bundle in bundles.items():
        result = audit(bundle)
        (out / f"{name}_audit.json").write_text(json.dumps(result, indent=2))
        (out / f"{name}_input.json").write_text(
            json.dumps(bundle, separators=(",", ":"))
        )
        text = markdown(name, result)
        (out / f"{name}_memo.md").write_text(text)
        print(text)
    capture(out)


if __name__ == "__main__":
    main()
