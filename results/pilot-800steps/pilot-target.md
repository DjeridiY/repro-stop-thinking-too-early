# Cible épinglée (04/10/2026, avant lancement) — tenue à part du dossier de travail
Papier : arXiv 2609.36585, code Lunamos/stop-thinking-too-early@dd4fb45 (Apache-2.0).
Palier : code publié, poids de LoRA non publiés -> relancer le script des auteurs, corriger au plus près.
Commande auteurs : e71_ouro_map.py --a 6 --tag o14_a6 (1200 pas, bs 16, longueurs 1-20, 2 chaînes, 4 boucles, text_kl 1, seed 0, neval 150).
A1 (principale) : avec LoRA, après 4 boucles, choice accuracy >= 0,80 à toutes les longueurs testées <= 24 (papier : « every tested length through 24 »).
A2 : gelé, reach <= 2,5 (papier : « at most 2.5 »).
A3 : reach avec LoRA 1,9 / 7,1 / 17 après 1/2/3 boucles.
Reach = plus longue chaîne à choice >= 0,8, interpolée linéairement (analysis/fig_reach_loop.py).
Tolérance : A1 vérifiée si min_{d<=24} choice(T4) >= 0,75 (marge ~2 ET à n=150) ; A2 si reach gelé <= 3,5 ; A3 si T3 dans [12;22] et T2 dans [4;10].
Budget : 3 h de T4 gratuite. Recette réduite (moins de pas) => verdict « jouet » au mieux.
