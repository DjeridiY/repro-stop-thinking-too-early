"""Compare our runs with the paper and apply the preregistered rules (PREREGISTRATION.md).

    python -m repro.reach            # results/seed{0,1,2}
    python -m repro.reach --pilot    # results/pilot-800steps (seed 0 only)

Reach and its 95 % bootstrap interval come from the authors' own code (upstream/analysis/tab_reach.py).
"""
import argparse
import ast
import json
import zlib
from pathlib import Path

from . import paper
from .run import RESULTS, ROOT, SEEDS, tag

LOOPS = (1, 2, 3, 4, 6, 8)
N_PER_CELL = 150


def authors_ci():
    """ci(row, assume_one, B, seed) from the authors' tab_reach.py, loaded without running their table script."""
    tree = ast.parse((ROOT / "upstream/analysis/tab_reach.py").read_text())
    keep: list[ast.stmt] = [n for n in tree.body
            if (isinstance(n, ast.Import) or isinstance(n, ast.ImportFrom) and n.module != "_paths")
            or isinstance(n, ast.FunctionDef) and n.name in {"reach", "ci"}]
    ns: dict = {}
    exec(compile(ast.Module(keep, []), "tab_reach.py", "exec"), ns)
    return ns["ci"]


ci = authors_ci()


def measure(result_file: Path) -> dict:
    """{('map'|'frozen', loops): (point, lo, hi, flag)} plus 'min_T4' = lowest LoRA accuracy at 4 loops."""
    ev = json.loads(result_file.read_text())["eval"]
    out = {}
    for kind in ("map", "frozen"):
        for t in LOOPS:
            row = {int(d): (v[1], N_PER_CELL) for d, v in ev.get(f"T{t}_{kind}", {}).items()}
            if row:
                out[kind, t] = ci(row, True, seed=zlib.crc32(f"{result_file.name}|T{t}_{kind}".encode()))
    out["min_T4"] = min(v[1] for v in ev["T4_map"].values())
    return out


def overlap(a, b) -> bool:
    return a[1] <= b[2] and b[1] <= a[2]


def verdicts(m: dict, seed: int) -> dict:
    return {
        "C1 all lengths ≤ 24 at 4 loops": m["min_T4"] >= 0.8,
        "C2 frozen reach ≤ 2.5": m["frozen", 4][0] <= 3.0 and overlap(m["frozen", 4], paper.FROZEN[4]),
        "C3 reach after 2 and 3 loops": all(overlap(m["map", t], paper.LORA[seed][t]) for t in (2, 3)),
        "C4 reach kept at 6 and 8 loops": all(m["map", t][0] >= 24 for t in (6, 8)),
    }


def cell(v) -> str:
    ge = "≥" if len(v) > 3 and v[3] == "ge" else ""
    return f"{ge}{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    runs = {0: RESULTS / "pilot-800steps"} if a.pilot else {s: RESULTS / f"seed{s}" for s in SEEDS}
    measured = {s: measure(d / f"e71_ouro_{tag(s)}.json") for s, d in runs.items() if (d / f"e71_ouro_{tag(s)}.json").exists()}
    if not measured:
        print("no finished run yet")
        return
    print("| Reach (lines) | " + " | ".join(f"seed {s}: paper | seed {s}: ours" for s in measured) + " |")
    print("|---|" + "---|---|" * len(measured))
    for kind, label in (("frozen", "frozen"), ("map", "LoRA")):
        for t in LOOPS if kind == "map" else (1, 2, 3, 4):
            ref = lambda s: paper.FROZEN[t] if kind == "frozen" else paper.LORA[s][t]
            print(f"| {label}, {t} loop{'s' if t > 1 else ''} | " + " | ".join(f"{cell(ref(s))} | {cell(m[kind, t])}" for s, m in measured.items()) + " |")
    per_seed = {s: verdicts(m, s) for s, m in measured.items()}
    print("\n| Claim | " + " | ".join(f"seed {s}" for s in per_seed) + " | Verdict |")
    print("|---|" + "---|" * len(per_seed) + "---|")
    for claim in next(iter(per_seed.values())):
        ok = [v[claim] for v in per_seed.values()]
        verdict = ("reproduced" if all(ok) else "not reproduced" if not any(ok) else "partially") if len(ok) == len(SEEDS) else "pending"
        print(f"| {claim} | " + " | ".join("pass" if x else "fail" for x in ok) + f" | {verdict} |")


if __name__ == "__main__":
    main()
