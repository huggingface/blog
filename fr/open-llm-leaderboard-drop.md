---
title: "Open LLM Leaderboard : plongée dans le jeu de données DROP"
thumbnail: /blog/assets/evaluating-mmlu-leaderboard/thumbnail.png
authors:
- user: clefourrier
- user: cabreraalex
  guest: true
- user: stellaathena
  guest: true
- user: SaylorTwift
- user: thomwolf
translators:
- user: lbourdois
---

# *Open LLM Leaderboard* : plongée dans le jeu de données DROP
Récemment, trois [nouveaux](https://twitter.com/clefourrier/status/1722555555338956840) jeux d'évaluation (*benchmarks* en anglais) ont été ajouté à l’[*Open LLM Leaderboard*](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) : Winogrande, GSM8k et DROP, utilisant les implémentations originales reproduites dans la bibliothèque [EleutherAI Harness](https://github.com/EleutherAI/lm-evaluation-harness/). Un coup d'œil rapide sur les scores F1 obtenus sur DROP a révélé que quelque chose d'étrange se passait, avec l'écrasante majorité des modèles obtenant moins de 10 sur 100 ! Nous avons fait une plongée en profondeur pour comprendre ce qui se passait, venez avec nous pour voir ce que nous avons trouvé !

## Observations initiales 
DROP (*Discrete Reasoning Over Paragraphs*) est une évaluation où les modèles doivent extraire les informations pertinentes de paragraphes en anglais avant d’exécuter des étapes de raisonnement (par exemple, trier ou compter des éléments pour arriver à la bonne réponse, voir le tableau ci-dessous pour des exemples). Les métriques utilisées sont un [score F1](https://hf.co/spaces/evaluate-metric/f1) personnalisé et la [correspondance exacte](https://hf.co/spaces/evaluate-metric/exact_match).
<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_example.png" width="500" />
  <figcaption>Exemples de raisonnements et de paragraphes issus du papier introduisant DROP</figcaption>
</figure>
</div>

Nous l'avons ajouté à l’*Open LLM Leaderboard* il y a trois semaines et avons observé que les scores F1 des modèles pré-entraînés suivaient une tendance inattendue. Lorsque nous avons tracé les scores de DROP par rapport à la moyenne du classement (sur ARC, HellaSwag, TruthfulQA et MMLU), qui est une approximation raisonnable de la performance globale du modèle, nous nous attendions à ce qu'ils soient corrélés (les meilleurs modèles ayant une meilleure performance). Cependant, cela n'a été le cas que pour un petit nombre de modèles, et tous les autres avaient un score F1 très faible, inférieur à 10.

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_bimodal.png" width="500" />
  <figcaption>Deux tendances peuvent être observées dans les scores de DROP : certains suivent la moyenne (en diagonale), d’autres sont coincés autour de 5 (ligne vertical sur la droite du graphique)</figcaption>
</figure>
</div>

## Interrogations sur la normalisation
Lors de notre premier examen approfondi de ces comportements surprenants, nous avons observé que l'étape de normalisation ne fonctionnait peut-être pas comme prévu. Dans certains cas, elle ignorait les réponses numériques correctes lorsqu'elles étaient directement suivies d'un caractère d'espacement autre qu'un espace (un retour à la ligne, par exemple).
Regardons un exemple, où la génération est `10\n\nPassage: The 2011 census recorded a population of 1,001,360` et la réponse attendue `10`.

La normalisation se produit en plusieurs étapes, à la fois pour la génération et la réponse attendue :
1) **Découpage sur les séparateurs** `|`, `-` ou ` `  
Le début de la séquence de la génération `10\n\nPassage:` ne contient pas de tels séparateurs, et est donc considéré comme une unique entité après cette étape.
2) **Suppression de la ponctuation**  
  La première sous-unité de mot (*token* en anglais) devient alors `10\n\nPassage` (`:` est supprimé)
3) **Homogénisation des nombres**  
Toute chaîne pouvant être convertie en flottant est considérée comme un nombre et convertie en flottant, puis reconvertie en chaîne. `10\n\nPassage` reste le même comme il ne peut pas, est converti en flottant alors que la réponse attendue `10` devient `10.0`.
4) **Autres étapes**  
 De nombreuses autres étapes de normalisation s'ensuivent (suppression des articles, suppression d'autres espaces vides, etc.) et notre exemple original devient `10 passage 2011.0 census recorded population of 1001360.0`.

Cependant, le score global n'est pas calculé sur la chaîne, mais sur le « sac de mots » (BOW pour *bag of word*) extrait de la chaîne, ici `{'recorded', 'population', 'passage', 'census', '2011.0', '1001360.0', '10'}`, qui est comparé au BOW de la réponse attendue, également normalisé de la manière décrite ci-dessus, `{10.0}`. Comme vous pouvez le constater, ces deux valeurs ne se recoupent pas, bien que le modèle ait prédit la bonne valeur !

En résumé, si un nombre est suivi d'un espace autre qu'un simple espace, il ne passera pas la normalisation des nombres, et ne correspondra donc jamais à la référence s'il s'agit également d'un nombre ! Ce premier problème était susceptible de perturber considérablement les scores, mais il n'était manifestement pas le seul facteur à l'origine des scores si faibles obtenus sur DROP. Nous avons donc décidé d'approfondir la question.

## Plongée dans les résultats
Nos amis de [Zeno](https://zenoml.com) nous ont rejoints et [ont entrepris une exploration beaucoup plus approfondie](https://hub.zenoml.com/report/1255/DROP%20Benchmark%20Exploration) des résultats, en examinant 5 modèles représentatifs des problèmes que nous avions remarqués dans les scores DROP : falcon-180B et mistral-7B étaient moins performants que ce à quoi nous nous attendions, Yi-34B et tigerbot-70B avaient une très bonne performance par rapport à leurs scores moyens, et facebook/xglm-7.5B se situait entre les deux. 
Vous pouvez essayer d'analyser les résultats [dans le projet Zeno ici](https://hub.zenoml.com/project/2f5dec90-df5e-4e3e-a4d1-37faf814c5ae/OpenLLM%20Leaderboard%20DROP%20Comparison/explore?params=eyJtb2RlbCI6ImZhY2Vib29rX194Z2xtLTcuNUIiLCJtZXRyaWMiOnsiaWQiOjk1NjUsIm5hbWUiOiJmMSIsInR5cGUiOiJtZWFuIiwiY29sdW1ucyI6WyJmMSJdfSwiY29tcGFyaXNvbk1vZGVsIjoiVGlnZXJSZXNlYXJjaF9fdGlnZXJib3QtNzBiLWNoYXQiLCJjb21wYXJpc29uQ29sdW1uIjp7ImlkIjoiYzJmNTY1Y2EtYjJjZC00MDkwLWIwYzctYTNiNTNkZmViM2RiIiwibmFtZSI6ImVtIiwiY29sdW1uVHlwZSI6IkZFQVRVUkUiLCJkYXRhVHlwZSI6IkNPTlRJTlVPVVMiLCJtb2RlbCI6ImZhY2Vib29rX194Z2xtLTcuNUIifSwiY29tcGFyZVNvcnQiOltudWxsLHRydWVdLCJtZXRyaWNSYW5nZSI6W251bGwsbnVsbF0sInNlbGVjdGlvbnMiOnsic2xpY2VzIjpbXSwibWV0YWRhdGEiOnt9LCJ0YWdzIjpbXX19) si vous le souhaitez !

L'équipe de Zeno a trouvé deux caractéristiques encore plus inquiétantes :
1) Pas un seul modèle n'a obtenu un résultat correct avec des réponses en virgule flottante.
2) Les modèles de haute qualité qui génèrent des réponses longues ont en fait un score F1 plus faible.
   
À ce stade, nous pensions que les deux cas d'échec étaient en fait causés par le même facteur : l'utilisation de `.` comme sous-unité de mot (pour terminer les générations) :

1) Les réponses en virgule flottante sont systématiquement interrompues avant que leur génération ne soit terminée.
2) Les modèles de meilleure qualité, qui tentent de reproduire le format des instructions avec exemples, génèrent `Réponse\n\nGénération plausible pour la question suivante.` et s'arrêtent après la réponse réelle au premier « . », générant ainsi trop de mots et obtenant un mauvais F1.

Nous avons émis l'hypothèse que ces deux problèmes pourraient être résolus en utilisant `\n` au lieu de `.` comme mot de fin de génération.

## Modification de la sous-unité de mot de fin de génération
Nous avons essayé d'utiliser `\n` comme sous-unité de mot de fin de génération sur les résultats disponibles. Nous avons découpé la réponse générée sur le premier `\n` qu'elle contenait, s'il y en avait un, et nous avons recalculé les scores. 

> [!NOTE]
> Notez que ce n'est qu'une approximation du bon résultat car cela ne corrigera pas les réponses qui ont été coupées trop tôt sur `.` (par exemple les réponses en virgule flottante) mais cela ne donnera pas non plus un avantage injuste à un modèle, car ils ont tous été affectés par ce problème. 
C'est le mieux que nous puissions faire sans réexécuter les modèles (car nous voulions tenir la communauté informée le plus tôt possible).

Nous observons alors que le découpage sur `\n` s'avère très bien corrélé avec les autres scores et donc avec la performance globale.

<div align="center">
<figure class="image table text-center m-0 w-full">
  <img src="https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/open-llm-leaderboard/drop/drop_partial_fix.png" width="500" />
  <figcaption> Nous pouvons voir en orange que les scores calculés sur les nouvelles chaînes sont en bien meilleure corrélation avec la performance moyenne<figcaption>
</figure>
</div>

## Quelle est la suite des opérations ?
Un calcul rapide montre qu'il serait assez coûteux de refaire l'évaluation complète de tous les modèles (l'actualisation du classement a pris 8 ans de temps GPU, dont une grande partie a été prise par DROP). Nous avons alors estimé combien cela coûterait de ne refaire que les exemples qui échouent.  
Dans 10% des cas, la réponse attendue est un nombre flottant (par exemple `12.25`) et les prédictions du modèle commencent par le début correct (pour notre exemple, `12`) mais sont interrompues par un `.` ; ces prédictions auraient probablement été correctes si la génération s'était poursuivie. Nous aurions certainement besoin de les relancer !
Notre estimation ne tient pas compte des phrases générées qui se terminent par un nombre qui a pu être interrompu (40% des autres générations), ni des prédictions perturbées par leur normalisation.  
Pour obtenir des résultats corrects, nous devrions donc réexécuter plus de 50 % des exemples, ce qui représente une énorme quantité de temps GPU ! Nous devons être certains que l'implémentation que nous exécuterons sera correcte cette fois-ci.

Après en avoir discuté avec la fantastique équipe d'EleutherAI (sur [GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/978) et en interne), qui nous a guidé à travers le code et aidé dans nos investigations, il est devenu très clair que l'implémentation de LM Eval Harness suit très strictement le code officiel de DROP. Une nouvelle version de l'évaluation de ce jeu d'évaluation doit donc être développée !  
**Nous avons donc pris la décision de retirer DROP de l'*Open LLM Leaderboard* jusqu'à ce qu'une nouvelle version soit disponible.**  
L'une des conclusions de cette enquête est qu'il est utile que de nombreux yeux examinent un jeu d'évaluation en collaboration afin de détecter des erreurs qui n'avaient pas été détectées auparavant. Là encore, la puissance de la communauté et du développement en open-source s'illustre dans la mesure où elle permet d'enquêter de manière transparente sur la cause première d'un problème sur un jeu d'évaluation qui existe depuis quelques années.  
Nous espérons que les membres de la communauté intéressés joindront leurs forces à celles des universitaires travaillant sur l'évaluation de DROP pour corriger à la fois sa notation et sa normalisation. Nous aimerions qu'il redevienne utilisable, car le jeu de données lui-même est vraiment très intéressant et cool. Nous vous encourageons à nous faire part de vos commentaires sur la manière dont nous devrions évaluer DROP [dans cette *issue*](https://github.com/EleutherAI/lm-evaluation-harness/issues/1050).

## Remerciements
Nous remercions les nombreux membres de la communauté qui nous ont signalé des problèmes concernant les scores DROP, ainsi que les équipes d'EleutherAI Harness et de Zeno pour leur aide précieuse sur ce point.
