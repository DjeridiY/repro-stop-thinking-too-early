# Deviations from the authors' setup

| What | Authors | Here | Possible effect |
|---|---|---|---|
| dtype | bfloat16 | float16 (`patches/fp16.patch`) | smaller range (max 65504), no GradScaler; LoRA parameters stay fp32, logits cast to fp32 as in the original. Step-0 loss 4.075 (fp16) vs 4.077 (bf16) in the pilot |
| GPU | not stated (A100 for the Qwen runs) | Tesla T4 16 GB, sm75, driver 580.82 | kernels differ; seeds do not match bit for bit |
| Python / packages | 3.11, `requirements-looped.txt` | same, installed with `uv` (`pip_freeze_s*.txt`) | torch 2.8.0 wheel is cu128, authors state CUDA 12.6 |
| Ouro-1.4B revision | not stated | `574fa66` (`trust_remote_code`) | if the remote code changed since the paper, results may differ |
| WikiText-103 revision | not stated | latest on the Hub at run time | text-KL batches may differ |
| Lengths evaluated | Table 11 seed 0 at T≥4 also uses longer programs (`e71b_ouro_scaling.py`) | script default 1-24 only | our reach is capped at 24; compare seed 0 at T≥4 as "≥24" |
| Pilot only | 1,200 steps | 800 steps (budget) | `results/pilot-800steps/`, reported separately |
