# Does a 33K-parameter LoRA really stop transformers from "thinking too early"?

**A preregistered reproduction on a free Colab T4.**

[Jin, Deng & Wang (2026)](https://arxiv.org/abs/2609.36585) report that pretrained models lose track of a chain of references
(`B = K`, `D = B`, … `print(D)`) after 2-3 lines, and that a rank-8 LoRA on a single layer, with every weight frozen, pushes
Ouro-1.4B past 24 lines. This repo re-runs that experiment on hardware anyone can get for free, with the claims and pass/fail
rules written down **before** the runs.

> **Status:** proof of concept in progress. Pilot below; 3 seeds × 1,200 steps (the paper's recipe) are running.

## Paper vs. reproduction (pilot: seed 0, 800 of 1,200 steps)
Reach = longest chain answered with ≥ 80 % accuracy, in lines, with 95 % intervals.

| | Paper | Ours | |
|---|---|---|---|
| Frozen model, 1 loop | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | ✅ |
| Frozen model, 2 loops | 1.9 [1.7, 2.1] | 1.9 [1.7, 2.1] | ✅ |
| Frozen model, 3 loops | 2.3 [1.9, 2.6] | 2.3 [1.9, 2.6] | ✅ |
| Frozen model, 4 loops | 2.5 [2.3, 2.8] | 2.5 [2.2, 2.8] | ✅ |
| LoRA, 1 loop | 1.9 [1.7, 2.2] | 1.8 [1.6, 2.1] | ✅ |
| LoRA, 2 loops | 7.1 [6.7, 7.7] | 7.2 [6.8, 8.0] | ✅ |
| LoRA, 3 loops | 17.0 [16.2, 17.5] | 18.8 [17.9, 20.1] | ⚠️ higher |
| LoRA, 4 loops | ≥ 24 ¹ | ≥ 24 | ✅ |
| LoRA, 6 loops | ≥ 24 ¹ | ≥ 24 | ✅ |
| LoRA, 8 loops | ≥ 24 ¹ | 21.6 [18.4, 24.0] | ⚠️ lower |

¹ The paper reports 26.7 / 37.4 / 40.9 using programs longer than 24 lines; our evaluation stops at 24.

**Takeaway so far:** the frozen model matches the paper to the decimal, and the LoRA effect is clearly there (2.5 → ≥ 24 lines).
Two cells drift: faster growth at 3 loops, a small drop at 8. The full-length runs will tell whether that comes from the
shorter training.

| Preregistered claim | Pilot |
|---|---|
| C1 — every length ≤ 24 answered after 4 loops | pass |
| C2 — frozen reach ≤ 2.5 | pass |
| C3 — reach after 2 and 3 loops within the paper's intervals | fail (3 loops) |
| C4 — reach kept at 6 and 8 loops | fail (8 loops) |

## How
- Authors' code, untouched, as a pinned submodule (`upstream/`, commit `dd4fb45`).
- One change: float16 instead of bfloat16 (`patches/fp16.patch`), because a T4 has no fast bf16 (~20 h → ~3 h per run).
- Reach and intervals computed with the authors' own bootstrap (`upstream/analysis/tab_reach.py`).
- Claims and rules: [PREREGISTRATION.md](PREREGISTRATION.md). Every other difference: [DEVIATIONS.md](DEVIATIONS.md).

## Run it
    git clone --recursive https://github.com/DjeridiY/repro-stop-thinking-too-early
    cd repro-stop-thinking-too-early && pip install numpy
    python -m repro.run              # next unfinished seed on a free Colab T4 (needs google-colab-cli)
    python -m repro.reach            # paper vs. ours, claim by claim (--pilot for the pilot)

On your own GPU: `cd upstream && git apply ../patches/fp16.patch`, then the commands in the preregistration.

| Path | |
|---|---|
| `repro/run.py` | one seed per Colab session: setup, launch, keep-alive, fetch, release |
| `repro/reach.py` · `repro/paper.py` | comparison with the paper's Tables 11-12, preregistered verdicts |
| `repro/vm_setup.sh` | authors' environment on the VM |
| `results/` | logs, JSON, LoRA weights, environment per seed |

## License
Apache-2.0. `upstream/` keeps the authors' Apache-2.0 license; Ouro-1.4B is Apache-2.0.

Yanis Djeridi
