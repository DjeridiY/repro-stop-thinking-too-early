# Preregistration

Written on 2026-10-05, before any evaluation output of the full runs: seed 0 had started training at 11:05 UTC (no evaluation
yet), seeds 1 and 2 had not started. Not edited afterwards; any change is a new dated section at the end.

## Target
Jin, Deng & Wang (2026), *Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It*, arXiv:2609.36585.
Code: Lunamos/stop-thinking-too-early @ `dd4fb45` (submodule `upstream/`). Model: ByteDance/Ouro-1.4B @ `574fa66`.

Experiment: the every-loop rank-8 LoRA at layer 6 of Ouro-1.4B (paper Sec. 4, Fig. 3a, App. C, Tables 11-12).

    python -u e71_ouro_map.py --a 6 --tag o14_a6            # seed 0
    python -u e71_ouro_map.py --a 6 --seed 1 --tag o14_a6_s1 --Ts_eval 1,2,3,4,6,8
    python -u e71_ouro_map.py --a 6 --seed 2 --tag o14_a6_s2 --Ts_eval 1,2,3,4,6,8

Authors' recipe kept: 1,200 steps, batch 16, lengths 1-20, two chains, four training loops, text-KL weight 1, n = 150 per cell.

## Known deviation
float16 instead of bfloat16 (`patches/fp16.patch`, 3 lines). The T4 has no bf16 tensor cores: measured 61 s/step in bf16
vs 9.5 s/step in fp16, i.e. ~20 h vs ~3 h per run. No other change to code, data, hyperparameters or evaluation.

## Metric
Reach = longest chain length with two-chain choice accuracy >= 0.8, linearly interpolated, with 95% parametric-bootstrap
intervals, computed with the authors' own `upstream/analysis/tab_reach.py` (`python -m repro.reach`).
Lengths evaluated: 1-24 (script default), so reach is capped at 24 ("≥24").

## Claims and decision rules
| Id | Claim (paper) | Reproduced if |
|---|---|---|
| C1 | With the LoRA, after 4 loops, every tested length up to 24 is answered | choice ≥ 0.8 at every length ≤ 24 at T=4, in all 3 seeds |
| C2 | Frozen reach is at most 2.5 (Table 11: 0.0 / 1.9 / 2.3 / 2.5 at T=1-4) | our frozen reach at T=4 ≤ 3.0 and its 95% interval overlaps [2.3, 2.8] |
| C3 | LoRA reach after 2 and 3 loops (Table 11, seeds 0/1/2: T2 7.1/7.3/7.0, T3 17.0/20.0/17.9) | for each seed, our 95% interval overlaps the paper's interval for the same seed at T=2 and T=3 |
| C4 | Extra inference loops keep reach (Table 12: ≥24 at T=6 and T=8 for all three seeds) | reach ≥ 24 at T=6 and T=8, per seed |

Verdict per claim: **reproduced** (all seeds pass), **partially** (some seeds), **not reproduced** (none), **inconclusive**
(run failed or budget exhausted). Seeds are not expected to match the authors' bit for bit (different GPU and dtype), so
claims are judged on intervals, not point values.

## Already observed (pilot, disclosed)
A pilot with 800 steps (budget) was run on 2026-10-05: `results/pilot-800steps/`. It informed the fp16 decision and the run
time. Under the rules above, the pilot gives C1 pass, C2 pass, C3 fail (T=3: 18.8 [17.9, 20.1] vs 17.0 [16.2, 17.5]),
C4 fail (T=8: 21.6). C4 was added because of the pilot; it is the claim most at risk.

## Execution
Free Colab T4, one seed per session (`python -m repro.run`). Every reported number is recomputed from the raw JSON files.
