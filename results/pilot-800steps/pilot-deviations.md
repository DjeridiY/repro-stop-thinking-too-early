# Écarts au code et à l'environnement des auteurs

Commande des auteurs (REPRODUCE.md) : `$LPY e71_ouro_map.py --a 6 --tag o14_a6` (défaut `--steps 1200`).
Commande lancée : `../.venv-loop/bin/python -u e71_ouro_map.py --a 6 --tag o14_a6 --steps 800`.

## 1. Environnement

| Quoi | Auteurs | Ici | Pourquoi | Effet possible |
|---|---|---|---|---|
| GPU | non précisé | Tesla T4 16 Go (sm75), pilote 580.82, CUDA 13.0 | VM imposée | voir 2 |
| Python | 3.11 | 3.11.16 (venv `uv venv --python 3.11`), pas le Python 3.13 de la VM | les versions épinglées (numpy 1.26.4, torch 2.8.0) visent 3.11 | aucun attendu |
| Paquets | requirements-looped.txt | requirements-looped.txt installé tel quel avec `uv pip install` ; torch/transformers 2.11/5.17 de la VM non utilisés | respect des versions des auteurs | aucun attendu |
| Wheel torch | CUDA 12.6 | torch 2.8.0+cu128 (wheel par défaut de PyPI) | wheel installé par `uv pip` sans index cu126 | noyaux CUDA différents, écarts numériques minimes possibles |
| Code du modèle Ouro (trust_remote_code) | révision non indiquée | snapshot `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` de ByteDance/Ouro-1.4B | dernière révision du hub au 2026-10-04 | si le modèle ou son code a changé depuis l'expérience des auteurs, les résultats peuvent différer |
| WikiText | révision non indiquée | snapshot `b08601e04326c79dfdd32d625aee71d232d685c3` de Salesforce/wikitext | idem | idem |
| Dossier `results/` | présent dans le dépôt des auteurs | créé à la main (`mkdir -p /content/work/results`) | `torch.save` écrit dans `results/` sans le créer | aucun |

## 2. dtype : bfloat16 remplacé par float16 (code modifié)

`scripts/e71_ouro_map.py` : `torch.bfloat16` -> `torch.float16` aux 3 endroits (chargement du modèle, deux `torch.autocast`).

Pourquoi : le T4 (sm75) n'a pas de tensor cores bf16 ; les matmuls bf16 sont émulés. Durées mesurées sur cette VM (même
commande, journal à chaque pas) :

| dtype | pas 0 (avec chauffe) | pas suivants | 1200 pas |
|---|---|---|---|
| bfloat16 | 63 s | 61 s | ~20 h |
| float16 | 10 s | ~9,5 s | ~3,2 h |

Effet possible : fp16 a 10 bits de mantisse (contre 7 en bf16) mais une plage de valeurs bien plus petite (max 65504). Des
activations du modèle gelé peuvent saturer ou perdre en précision ; le modèle a été entraîné et publié en bf16. Pas de
GradScaler : sous autocast fp16, des gradients de la carte (paramètres en fp32) peuvent tomber à zéro par sous-dépassement.
Les logits sont convertis en float32 avant softmax et entropie croisée, comme dans le code d'origine. Les pertes mesurées aux
6 premiers pas sont finies et proches de celles du run bf16 (pas 0 : 4,075 en fp16, 4,077 en bf16 ; pas 1 : 1,785 contre 1,813).

## 3. Nombre de pas d'entraînement : 1200 -> 800 (hyperparamètre réduit)

Pourquoi : budget de 3 h de GPU. Avec fp16 : 1200 pas x 9,5 s ≈ 3 h 10 pour l'entraînement seul. L'évaluation (6 valeurs de T
x 2 modes x 10 profondeurs x 150 items), estimée à partir du benchmark (2,75 s pour 25 items à d=20, T=4), prend ~15-20 min.
Environ 10 min de GPU ont déjà servi à l'installation et aux mesures. Désactiver le gradient checkpointing pour gagner du temps
n'est pas possible : les activations de 96 couches pour 16 x ~260 jetons dépassent 16 Go. Il reste ~2 h 50 ; 800 pas
(~2 h 07) + évaluation laissent ~15-25 min de marge.

Le taux d'apprentissage est constant (AdamW, pas de scheduler) et le générateur est tiré dans le même ordre : les 800 pas
lancés sont les 800 premiers pas du run de 1200 des auteurs (aux écarts numériques près). Le dernier pas journalisé est le 750.

Effet possible : la carte est moins entraînée ; l'exactitude avec la carte peut être plus basse que celle du papier,
en particulier aux grandes profondeurs. L'évaluation, les données, la graine et les autres hyperparamètres ne changent pas.

## 4. Benchmark de durée (hors run final)

Pour mesurer un pas, deux copies temporaires sur la VM uniquement (`bench_bf16.py`, `bench_fp16.py`, non commitées) : journal à
chaque pas (`if True:` à la place de `if step % 50 == 0:`), lancées avec `--steps 6` (fp16) ou `--steps 3` (bf16, arrêté
après 2 pas), `--Ts_eval 4 --depths_eval 20 --neval 25`. Elles écrivent dans `results/` sous les tags `bench16` et `bench_bf`,
qui ne servent pas au run final.
