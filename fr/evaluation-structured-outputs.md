---
title: "Améliorer la stabilité des instructions grâce à la génération structurée"
thumbnail: /blog/assets/evaluating-mmlu-leaderboard/thumbnail.png
authors:
- user: willkurt
  guest: true
  org: dottxt
- user: remi
  guest: true
  org: dottxt
- user: clefourrier
translators:
- user: lbourdois
---

# Améliorer la stabilité des instructions grâce à la génération structurée

Récemment, l'équipe de recherche *Leaderboards and Evals* d'Hugging Face a mené de petites expériences qui ont mis en évidence l'instabilité de l'évaluation. Pour une tâche donnée, les résultats sont extrêmement sensibles à de minuscules changements dans le format de l'instruction (*prompt* en anglais) ! Or, ce n'est pas ce que nous voulons : un modèle recevant des instructions à contenu sémantique équivalent devrait produire des résultats similaires.

Nous en avons discuté avec nos amis de *Dottxt*, qui ont eu une idée : et s'il existait un moyen d'améliorer la stabilité entre les formats des instructions ? 

Investiguons tout ça !

## Contexte : Sensibilité de l'évaluation aux changements de format

Il est devenu de plus en plus clair que la performance des jeux d'évaluation (*benchmarks* en anglais) de LLM dépend étroitement, et de manière assez surprenante, du *format* de l'instruction lui-même, même si un certain nombre de méthodes ont été introduites au fil des ans pour réduire la variance des instructions. À titre d'illustration, lorsque nous évaluons des modèles avec plusieurs exemples (*few-shot* en anglais), nous fournissons des exemples de format au modèle pour forcer un gabarit spécifique en sortie ; lorsque nous comparons la log-vraisemblance des réponses plausibles au lieu de permettre une génération libre, nous essayons de contraindre l'espace de réponse.

L'équipe *Leaderboards and Evals* en a fait la démonstration en examinant 8 formats d'instruction différents pour une tâche bien connue, MMLU (en examinant 4 sous-sections de la tâche). Ces variations d'instruction ont été fournies à 5 modèles différents (choisis parce qu'ils étaient à l'état de l'art à l'époque pour leur taille, et qu'ils couvraient une variété d'unité de mot (*token* en anglais) et de langues). Les scores ont été calculés à l'aide d'une évaluation par log-probabilité, où la réponse la plus probable est considérée comme la bonne, une mesure classique pour les tâches à choix multiples. 

Examinons les différents formats plus en détail, en utilisant la première question de la sous-section `global_facts` de MMLU.

```

Question: “As of 2016, about what percentage of adults aged 18 years or older were overweight?”

Choices: [ "10%", "20%", "40%", "80%" ]

Correct choice: “40%”

# En français :
Question : "En 2016, quel était le pourcentage d'adultes âgés de 18 ans ou plus en surpoids ?"

Choix : [ "10%", "20%", "40%", "80%" ]

Choix correct : "40%"
```

<div>
<table><p>
  <tbody>
  <tr> <td colspan=3 text-align=center> Sans choix dans l'instruction </td></tr>
  <tr style=" vertical-align: top;">
    <td>As of 2016, about what percentage of adults aged 18 years or older were overweight?</td>
    <td>Q: As of 2016, about what percentage of adults aged 18 years or older were overweight? <br><br> A: </td>
    <td>Question: As of 2016, about what percentage of adults aged 18 years or older were overweight?<br><br> Answer: </td>
  </tr>
  <tr> <td colspan=3>  </td></tr>
  <tr> <td colspan=3> Avec des choix dans l'instruction </td></tr>
  <tr>
    <td>Question: As of 2016, about what percentage of adults aged 18 years or older were overweight?<br><br>
    Choices: <br><br>
    10% <br>
    20% <br>
    40% <br>
    80% <br><br>
    Answer: </td>
    <td>Question: As of 2016, about what percentage of adults aged 18 years or older were overweight?<br><br>
    Choices: <br><br>
    A. 10% <br>
    B. 20% <br>
    C. 40% <br>
    D. 80% <br><br>
    Answer: </td>
    <td>Question: As of 2016, about what percentage of adults aged 18 years or older were overweight?<br><br>
    Choices: <br><br>
    (A) 10% <br>
    (B) 20% <br>
    (C) 40% <br>
    (D) 80% <br><br>
    Answer: </td>
  </tr>
  <tr> 
    <td> Log-probabilités de 10%, 20%, 40%, 80% </td>
    <td> Log-probabilités de 10%, 20%, 40%, 80% vs A, B, C, D </td>
    <td> Log-probabilités de 10%, 20%, 40%, 80% vs (A), (B), (C), (D), </td>
  </tbody>
</table><p>
</div>

Les instructions contiennent soit uniquement la question, soit quelques balises pour indiquer que nous sommes dans un format question/réponse, et éventuellement tous les choix possibles. Dans tous les cas, les évaluations comparent uniquement la log-vraisemblance des choix possibles. Tous ces formats apparaissent dans la littérature d'évaluation et devraient contenir pratiquement la même quantité d'informations dans chaque ligne. Cependant, vous pouvez voir ci-dessous la grande variation des performances en fonction de ces changements théoriquement superficiels !

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-0.png)

Chaque modèle voit ses performances varier d'environ 10 points, à l'exception de l'exemple le plus extrême, Qwen1.5-7B, dont la performance chute de 51,2% pour son meilleur format d'instruction à 22,9% pour le moins bon alors que les deux contiennent essentiellement les mêmes informations.

Pris isolément, un changement de *score* n'est pas nécessairement très important tant que le *classement* est cohérent. Cependant, comme nous pouvons le voir dans le graphique suivant, il est affecté par ces changements :

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-1.png)

Aucun modèle n'est classé de manière constante d'une instruction à l'autre, même si la seule différence réside dans le format, et non dans l'information elle-même. Cela signifie que si les auteurs de Gemma-7b voulaient montrer que leur modèle est supérieur à Mistral-7B-v0.1, ils pourraient le faire simplement en choisissant la bonne instruction. 
Étant donné que presque personne n'indique sa configuration d'évaluation précise, c'est ce qui s'est produit historiquement dans les rapports, où les auteurs ont choisi d'indiquer la configuration la plus avantageuse pour leur modèle (c'est la raison pour laquelle vous verrez des nombres extrêmement bizarres dans certains papiers pour les instructions avec plusieurs exemples).

Cependant, ce n'est pas la seule source de variance dans les scores des modèles. 

Dans d'autres expériences, nous avons comparé les résultats obtenus par les mêmes modèles, instruits de la même façon, et auxquels ont été présentés les mêmes exemples; la seule différence entre les évaluations résidait dans l'ordre des exemples en question (par exmeple, instruction A/B/C/D/E vs instruction C/D/A/B/E). La figure suivante montre le delta des scores des modèles entre ces deux ordonnancements. Nous observons une différence de performance allant jusqu'à 3 points pour la même combinaison modèle/instruction !

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-2.png)

Si nous voulons être en mesure d'évaluer et de comparer correctement les différents modèles, nous devons trouver un moyen de surmonter cette difficulté.

L'article de Sclar et al. *[Quantifying Language Model's Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324)* donne également un bon aperçu de cette question, et les auteurs présentent [*FormatSpread*](https://github.com/msclar/formatspread), un outil logiciel qui évalue chaque modèle avec de multiples variations de formats puis calcule la variance des performances de ce modèle. De telles solutions nous permettent de déterminer avec plus d'assurance quels modèles sont meilleurs que d'autres, mais elles ont un coût de calcul élevé.

## Et si nous nous concentrions sur la sortie, et non sur l'entrée, pour rendre les résultats plus cohérents à travers ces petits changements de format ?

Bien que FormatSpread soit une excellente tentative pour rendre les classements plus justes et honnêtes, ce que nous voulons vraiment en tant qu'utilisateurs de LLM est la *stabilité des instructions*. En d'autres termes, nous aimerions trouver un moyen de réduire la variance entre les *instructions*.

À [.txt](http://dottxt.co/), nous nous concentrons sur l'amélioration et la meilleure compréhension de la *génération structurée*, c'est-à-dire lorsque la sortie d'un modèle est contrainte de suivre une structure spécifique. Notre bibliothèque, [Outlines](https://github.com/outlines-dev/outlines), nous permet de structurer la sortie d'un LLM à partir d'une expression régulière ou une grammaire non contextuelle (nous donnons des exemples ci-dessous). 

Nous nous sommes d'abord interessés à la generation structurée pour faciliter l'interaction programmatique avec les LLM, en garantissant des réponses en JSON bien formatées. Cependant, nous avons été continuellement surpris par les autres avantages de la génération structurée. 

En travaillant sur des recherches antérieures explorant les avantages de la génération structurée, nous avons démontré que [la génération structurée améliore systématiquement les performances des jeux d'évaluation](http://blog.dottxt.co/performance-gsm8k.html), et nous avons rencontré un cas particulier intéressant en explorant les instructions structurés en JSON.

Dans la plupart des cas, la modification du format d'instruction en JSON, même en utilisant la génération non structurée, conduit à une amélioration des performances du jeu d'évaluation pour presque tous les modèles. Cependant, ce n'est pas le cas pour MetaMath-Tulpar-7b-v2-Slerp, pour lequel nous avons constaté une diminution spectaculaire de la précision lors de l'utilisation d'instructions formatés en JSON. Plus surprenant encore, lorsque l'on utilise la *génération structurée* pour contraindre la sortie du modèle, la baisse de performance est négligeable ! 

Cela nous a amenés à nous demander si la génération structurée pouvait être exploitée pour assurer la *stabilité des instructions*.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-3.png)

### Note sur le dispositif expérimental : Focalisation sur n-exemples et leur ordonnancement

Alors que dans les expériences précédentes, l'équipe de recherche *Leaderboard and Evals* d'Hugging Face a exploré les changements de format d'instruction lui-même, pour les expériences suivantes, nous allons restreindre les changements. 

Pour cibler notre exploration de l'espace de l'instruction, nous allons chercher à faire varier seulement deux propriétés de l'instruction :
1. Varier le nombre d'exemples utilisés dans l'instruction (*n-exemples*).
2. Varier l'ordre de ces exemples (ordre spécifié par une *graine*)
Pour le point 2, avec *n-exemples* donnés, nous ne mélangeons que les mêmes *n* exemples. Cela signifie que tous les mélanges d'une instruction avec *1-exemple* sont identiques. Cela permet d'éviter de confondre le *format* d'une instruction avec l'*information* contenue. Il est clair qu'une instruction contenant *5-exemples* possède plus d'informations qu'une instruction avec *1-exemple*, mais tous les mélanges d'une instruction à *5-exemples* contiennent les mêmes exemples, mais dans un ordre différent.

## Exploration initiale : GSM8K instruction contenant 1 à 8 *exemples*

Afin d'approfondir cette question, nous avons voulu explorer le comportement de deux modèles très similaires mais forts dans l'espace des paramètres 7B : Mistral-7Bv0.1 et Zephyr-7B-beta. La raison de ce choix est d'étudier non seulement la variance des résultats individuels, mais aussi les *changements dans le classement relatif*. Nous utilisons la tâche GSM8K qui est un ensemble de problèmes mathématiques de niveau primaire.

Voici le format de base d'une instruction GSM8K avec la structure implicite mise en évidence.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-4.png)

Afin de générer systématiquement des réponses correctement structurées, nous créons une expression régulière qui correspond à la structure inhérente au format original de l'instruction. L'expression régulière suivante est utilisée dans Outlines pour définir la structure pour la génération :

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-5.png)

Nous pouvons voir dans la regex que nous autorisons le modèle à raisonner sur 200 à 700 caractères, puis il doit déclarer que « *The answer is* » et ensuite répondre avec un nombre de 10 chiffres maximum (qui ne peut pas commencer par 0).

Il convient de mentionner que la regex qui contrôle la structure est similaire mais pas identique à la regex utilisée pour analyser la réponse. Nous avons appris qu'il y a une nuance intéressante dans la définition de la structure car, comme l'instruction, elle peut avoir un impact sur les performances. Par exemple, remarquez la présence de `{200,700}` dans la regex. Cela signifie que le modèle dispose de 200 à 700 caractères pour « raisonner » avant de répondre. La modification de ces valeurs peut avoir un impact sur les performances et conduit à ce que nous appelons le « contrôle de la pensée », un domaine sur lequel nous espérons écrire plus en détail bientôt.

Notre première expérience a consisté à poursuivre l'exploration du jeu de données GSM8K et à itérer sur les instructions de 1 à 8 exemples. Les résultats, présentés ci-dessous, sont très convaincants.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-6.png)

Cette figure présente deux caractéristiques majeures : la variance des performances entre les configurations à *n-exemples* a été considérablement réduite et il n'y a eu aucun cas où le classement a changé (le Mistral est toujours en tête devant le Zéphyr). Il convient également de souligner que les performances structurées à *1-exemple* sont nettement meilleures que les performances non structurées à *1-exemple*, et qu'elles sont comparables aux performances à *5-exemples*. Cela nous amène à un autre domaine de recherche que nous appelons « efficacité de l'instruction ».

## Plongée dans les variations à *n-exemples* et de leur ordonnancement pour GPQA

Pour l'expérience suivante, nous avons voulu faire varier à la fois les *n-exemples* et leur ordre. L'ordre a été contrôlé en définissant la graine utilisée pour mélanger les exemples. Comme indiqué précédemment, seuls les *n-exemples* sont mélangés afin que les informations restent constantes d'une instruction à l'autre, ce qui signifie que toutes les instructions à *1-exemple* sont les mêmes d'une graine à l'autre. Voici un exemple de l'ordre des exemples pour un *4-exemples* :

| graine | ordre *4-exemples* |
| --- | --- |
| 42 | 2-1-3-0 |
| 1337 | 1-0-3-2 |
| 1981 | 3-2-0-1 |
| 1992 | 0-3-1-2 |
| 12345 | 1-0-2-3 |

En outre, afin d'explorer le degré de transférabilité de ces résultats, nous avons modifié la tâche en [*Graduate-Level Google-Proof Q&A Benchmark* (GPQA)](https://arxiv.org/abs/2311.12022). GPQA est une tâche d'évaluation difficile de connaissances à choix multiples. Le format de l'instruction et la structure mise en évidence sont présentés ci-dessous.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-7.png)

Pour l'expérience suivante, nous utilisons spécifiquement le sous-ensemble "diamond" qui représente des questions de haute qualité traitées et nettoyées. Sur les 198 questions de ce jeu de données, nous en avons réservé 8 pour l'instruction à *n-exemples* (bien que nous n'ayons jamais utilisé que les 5 premières), puis nous avons évalué les 190 questions restantes.

La grille ci-dessous représente la précision obtenue pour toutes les combinaisons possibles de graines et de *n*, pour les deux modèles, à la fois sans (à gauche) et avec (à droite) la génération structurée.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-9.png)

Une chose qui ressort immédiatement est que les résultats structurés ont tendance à être plus élevés que les résultats non structurés. La moyenne de chaque grille pour les résultats structurés et non structurés est indiquée ci-dessous :

**Moyenne des résultats sur toutes les graines et *n-exemples*** :

| modèle | non structuré | structuré |
| --- | --- | --- |
| Mistral-7B-v0.1 | 0,2360 | 0,2935 |
| Zephyr-7b-beta | 0,2387 | 0,3048 |

En outre, pour toutes les valeurs de la grille, nous constatons également une *réduction de la variance* lorsque nous comparons la génération structurée à la génération non structurée. 

**Moyenne des résultats sur toutes les graines et *n-exemples*** :
| modèle | non structuré | structuré |
| --- | --- | --- |
| Mistral-7B-v0.1 | 0,0213 | 0,0202 |
| Zephyr-7b-beta | 0,0273 | 0,0180 |

Cette réduction de la variance sur l'ensemble de la grille est similaire à la réduction de la variance que nous avons constatée en examinant uniquement les changements des *n-exemples* pour GSM8K.

Bien que l'augmentation de la performance attendue et la diminution de la variance soient des propriétés intéressantes, ce que nous voulons vraiment comprendre, c'est l'impact sur le classement. Dans le graphique suivant, nous examinons ces grilles afin de déterminer lequel des deux modèles serait déclaré vainqueur en utilisant
- A : Zephyr-7b-beta
- B : Mistral-7B-v0.1
- “-” : égalité

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-10.png)

Comme nous pouvons le voir sur ces images, il y a une amélioration majeure dans la constance de l'annonce d'un gagnant lorsque la génération structurée est appliquée. Ces résultats sont conformes à ceux que nous avons obtenus en utilisant GSM8K pour différents *n-exemples*.

## Conclusion et travaux futurs

Bien que ces résultats soient incroyablement prometteurs, nous devons encore les explorer sur un plus grand nombre de modèles et de tâches. Ce que nous avons vu jusqu'à présent, c'est que la génération structurée pourrait s'avérer être une partie essentielle de l'évaluation. Le fait d'avoir simultanément *augmenté* le score attendu et *diminué* la variance entre les changements d'instruction est un résultat très prometteur qui mérite d'être approfondi.
