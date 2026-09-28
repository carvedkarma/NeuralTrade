from __future__ import annotations

import argparse
from pathlib import Path

VERSION = "RITHAL_LAYER_TRUTH_V1_0_4_7_R2"

PATCHES = [
    (
        "exact manager trade and ticket path binding",
        """        cand_trade_id = extract_trade_id(mapping)\n        cand_ticket_id = extract_ticket_id(mapping)\n        cand_symbol = extract_symbol(mapping)\n""",
        """        cand_trade_id = extract_trade_id(mapping)\n        if not cand_trade_id and trade_id and trade_id in path:\n            cand_trade_id = trade_id\n        cand_ticket_id = extract_ticket_id(mapping)\n        if not cand_ticket_id and ticket_id and ticket_id in path:\n            cand_ticket_id = ticket_id\n        cand_symbol = extract_symbol(mapping)\n""",
        "if not cand_trade_id and trade_id and trade_id in path:",
    ),
    (
        "exact entry thesis path binding",
        """        if trade_id and mapped_trade == trade_id:\n            matches.append((extract_timestamp(mapping), mapping, path))\n        elif ticket_id and mapped_ticket == ticket_id:\n""",
        """        if trade_id and (mapped_trade == trade_id or trade_id in path):\n            matches.append((extract_timestamp(mapping), mapping, path))\n        elif ticket_id and (mapped_ticket == ticket_id or ticket_id in path):\n""",
        "mapped_trade == trade_id or trade_id in path",
    ),
    (
        "canonical regime projection",
        '                    "regime": regime_truth(thesis.get("snapshot") if isinstance(thesis, dict) else position),\n',
        '                    "regime": (((thesis.get("snapshot") or {}).get("regime")) if isinstance(thesis, dict) and isinstance(thesis.get("snapshot"), dict) and isinstance((thesis.get("snapshot") or {}).get("regime"), dict) else regime_truth(position)),\n',
        'else regime_truth(position)),',
    ),
    (
        "requested margin USD alias",
        '    requested_margin = safe_float(first_value(position, ("requested_margin", "margin", "margin_usd")))\n',
        '    requested_margin = safe_float(first_value(position, ("requested_margin", "requested_margin_usd", "margin", "margin_usd")))\n',
        '"requested_margin_usd", "margin"',
    ),
]


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    changed = False
    for name, old, new, marker in PATCHES:
        if marker in text:
            print(f"[{VERSION}] {name}: already applied")
            continue
        if old not in text:
            raise RuntimeError(f"{name}: patch anchor not found")
        text = text.replace(old, new, 1)
        changed = True
        print(f"[{VERSION}] {name}: applied")
    if changed:
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[{VERSION}] PATCH PASS: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("runtime")
    args = parser.parse_args()
    path = Path(args.runtime).resolve()
    if not path.is_file():
        raise SystemExit(f"runtime not found: {path}")
    patch_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
