---
title: "Améliorer la constance des prompts avec les générations structurées"
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

# Améliorer la constance des prompts avec les générations structurées

Récemment, l'équipe de recherche *Leaderboards and Evals* d'Hugging Face a mené de petites expériences qui ont mis en évidence l'inconstance de l'évaluation. Pour une tâche donnée, les résultats sont extrêmement sensibles à de minuscules changements dans le format du *prompt* ! Or, ce n'est pas ce que nous voulons : un modèle prompté avec la même quantité d'informations en entrée devrait produire des résultats similaires.

Nous en avons discuté avec nos amis de *Dottxt*, qui ont eu une idée : et s'il existait un moyen d'améliorer la constance entre les formats de *prompt* ? 

Creusons tout ça !

## Contexte : Sensibilité de l'évaluation aux changements de format

Il est devenu de plus en plus clair que la performance des *benchmarks* de LLM dépend étroitement, et de manière assez surprenante, du *format* du *prompt* lui-même, même si un certain nombre de méthodes ont été introduites au fil des ans pour réduire la variance liée au *prompt*. Par exemple, lorsque nous évaluons des modèles en *few-shot*, nous fournissons des exemples de format au modèle pour forcer un gabarit spécifique en sortie ; lorsque nous comparons la log-vraisemblance des réponses plausibles au lieu de permettre une génération libre, nous essayons de contraindre l'espace de réponse.

L'équipe *Leaderboards and Evals* en a fait la démonstration en examinant 8 formats de *prompt* différents pour une tâche bien connue, MMLU (en examinant 4 sous-ensembles de la tâche). Ces variations de *prompt* ont été fournies à 5 modèles différents (choisis parce qu'ils étaient SOTA à l'époque pour leur taille, et qu'ils couvraient une variété de *token* et de langues). Les scores ont été calculés à l'aide d'une évaluation par log-probabilité, où la réponse la plus probable est considérée comme la bonne, une mesure classique pour les tâches à choix multiples. 

Examinons les différents formats plus en détail, en utilisant la première question du sous-ensemble `global_facts` de MMLU.

```

Question: “As of 2016, about what percentage of adults aged 18 years or older were overweight?”

Choices: [ "10%", "20%", "40%", "80%" ]

Correct choice: “40%”

```

<div>
<table><p>
  <tbody>
  <tr> <td colspan=3 text-align=center> Sans choix dans le <i>prompt</i> </td></tr>
  <tr style=" vertical-align: top;">
    <td>As of 2016, about what percentage of adults aged 18 years or older were overweight?</td>
    <td>Q: As of 2016, about what percentage of adults aged 18 years or older were overweight? <br><br> A: </td>
    <td>Question: As of 2016, about what percentage of adults aged 18 years or older were overweight?<br><br> Answer: </td>
  </tr>
  <tr> <td colspan=3>  </td></tr>
  <tr> <td colspan=3> Avec des choix dans le <i>prompt</i> </td></tr>
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

Les *prompts* contiennent soit uniquement la question, soit quelques balises pour indiquer que nous sommes dans un format question/réponse, et éventuellement tous les choix possibles. Dans tous les cas, les évaluations comparent uniquement la log-vraisemblance des choix possibles. Tous ces formats apparaissent dans la littérature d'évaluation et devraient contenir pratiquement la même quantité d'informations dans chaque ligne. Cependant, vous pouvez voir ci-dessous la grande variation des performances en fonction de ces changements théoriquement superficiels !

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-0.png)

Chaque modèle voit ses performances varier d'environ 10 points, à l'exception de l'exemple le plus extrême, Qwen1.5-7B, qui chute jusqu'à une *accuracy* de 22,9 % avec la 7ème variation de *prompt* (principalement en raison d'un problème de *tokenizer*), alors qu'avec essentiellement les mêmes informations, il a pu atteindre une exactitude de 51,2 % avec un autre *prompt*.

Pris isolément, un changement de *score* n'est pas nécessairement très important tant que le *classement* est cohérent. Cependant, comme nous pouvons le voir dans le graphique suivant, il est affecté par ces changements :

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-1.png)

Aucun modèle n'est classé de manière constante d'un *prompt* à l'autre, même si la seule différence réside dans le format, et non dans l'information elle-même. Cela signifie que si les auteurs de Gemma-7b voulaient montrer que leur modèle est supérieur à Mistral-7B-v0.1, ils pourraient le faire simplement en choisissant le bon *prompt*. 
Étant donné que presque personne n'indique sa configuration d'évaluation précise, c'est ce qui s'est produit historiquement dans les rapports, où les auteurs ont choisi d'indiquer la configuration la plus avantageuse pour leur modèle (c'est la raison pour laquelle vous verrez des nombres extrêmement bizarres de « *few-shots* » dans certains papiers).

Cependant, ce n'est pas la seule source de variance dans les scores des modèles. 

Dans des expériences étendues, nous avons comparé l'évaluation de mêmes modèles, avec les mêmes formats de *prompt*, en utilisant exactement les mêmes exemples *few-shot* mélangés différemment avant le *prompt* (*prompt* A/B/C/D/E vs *prompt* C/D/A/B/E, par exemple). La figure suivante montre le delta des scores des modèles entre ces deux ordonnancements. Nous observons une différence de performance allant jusqu'à 3 points pour la même combinaison modèle/*prompt* !

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-2.png)

Si nous voulons être en mesure d'évaluer et de comparer correctement les différents modèles, nous devons trouver un moyen de surmonter cette difficulté.

L'article de Sclar et al. *[Quantifying Language Model's Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324)* donne également un bon aperçu de cette question, et les auteurs présentent [*FormatSpread*](https://github.com/msclar/formatspread), un outil logiciel qui évalue chaque modèle avec de multiples variations de formats puis calcule la variance des performances de ce modèle. De telles solutions nous permettent de déterminer avec plus d'assurance quels modèles sont meilleurs que d'autres, mais elles ont un coût de calcul élevé.

## Et si nous nous concentrions sur la sortie, et non sur l'entrée, pour rendre les résultats plus cohérents à travers ces petits changements de format ?

Bien que FormatSpread soit une excellente tentative pour rendre les classements plus justes et honnêtes, ce que nous voulons vraiment en tant qu'utilisateurs de LLM est la *constance des prompts*. En d'autres termes, nous aimerions trouver un moyen de réduire la variance entre les *prompts*.

À [.txt](http://dottxt.co/), nous nous concentrons sur l'amélioration et la meilleure compréhension de la *génération structurée*, c'est-à-dire lorsque la sortie d'un modèle est contrainte de suivre une structure spécifique. Notre bibliothèque, [Outlines](https://github.com/outlines-dev/outlines), nous permet de structurer la sortie d'un LLM en définissant une expression régulière ou une grammaire sans contexte (nous donnons des exemples ci-dessous). 

Notre cas d'utilisation initial pour la génération structurée était de faciliter l'interaction programmatique avec les LLM, en garantissant des réponses en JSON bien formatées. Cependant, nous avons été continuellement surpris par les autres avantages de la génération structurée que nous avons découverts. 

En travaillant sur des recherches antérieures explorant les avantages de la génération structurée, nous avons démontré que [la génération structurée améliore systématiquement les performances des *benchmarks*](http://blog.dottxt.co/performance-gsm8k.html), et nous avons rencontré un cas particulier intéressant en explorant les *prompts* structurés en JSON.

Dans la plupart des cas, la modification du format du *prompt* en JSON, même en utilisant la génération non structurée, conduit à une amélioration des performances du *benchmark* pour presque tous les modèles. Cependant, ce n'est pas le cas pour MetaMath-Tulpar-7b-v2-Slerp, pour lequel nous avons constaté une diminution spectaculaire de la précision lors de l'utilisation de *prompts* formatés en JSON. Plus surprenant encore, lorsque l'on utilise la *génération structurée* pour contraindre la sortie du modèle, la baisse de performance est négligeable ! 

Cela nous a amenés à nous demander si la génération structurée pouvait être exploitée pour assurer la *cohérence des prompts*.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-3.png)

### Note sur le dispositif expérimental : Focalisation sur le *n-shot* et leur ordonnancement

Alors que dans les expériences précédentes, l'équipe de recherche *Leaderboard and Evals* d'Hugging Face a exploré les changements de format du *prompt* lui-même, pour les expériences suivantes, nous allons restreindre les changements. 

Pour cibler notre exploration de l'espace du *prompt*, nous allons chercher à faire varier seulement deux propriétés du *prompt* :
1. Varier le nombre de *shots* ou exemples utilisés dans le *prompt* (*n-shot*).
2. Varier l'ordre de ces *shots* (ordre spécifié par une *graine*)
Pour le point 2, avec un *n-shot* donné, nous ne mélangeons que les mêmes *n* exemples. Cela signifie que tous les mélanges d'un *prompt* *1-shot* sont identiques. Cela permet d'éviter de confondre le *format* d'un *prompt* avec l'*information* qu'il contient. Il est clair qu'un *prompt* à *5-shots* contient plus d'informations qu'un *prompt* à *1-shot*, mais tous les mélanges d'un *prompt* à *5-shots* contiennent les mêmes exemples, mais dans un ordre différent.

## Exploration initiale : GSM8K *prompt* 1 à 8 *shots*

Afin d'approfondir cette question, nous avons voulu explorer le comportement de deux modèles très similaires mais forts dans l'espace des paramètres 7B : Mistral-7Bv0.1 et Zephyr-7B-beta. La raison de ce choix est d'étudier non seulement la variance des résultats individuels, mais aussi les *changements dans le classement relatif*. Nous utilisons la tâche GSM8K qui est un ensemble de problèmes mathématiques de niveau primaire.

Voici le format de base d'un *prompt* GSM8K avec la structure implicite mise en évidence.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-4.png)

Afin de générer systématiquement des réponses correctement structurées, nous créons une expression régulière qui correspond à la structure inhérente au format original du *prompt*. L'expression régulière suivante est utilisée dans Outlines pour définir la structure pour la génération :

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-5.png)

Nous pouvons voir dans la regex que nous autorisons le modèle à raisonner sur 200 à 700 caractères, puis il doit déclarer que « *The answer is* » et ensuite répondre avec un nombre de 10 chiffres maximum (qui ne peut pas commencer par 0).

Il convient de mentionner que la regex qui contrôle la structure est similaire mais pas identique à la regex utilisée pour analyser la réponse. Nous avons appris qu'il y a une nuance intéressante dans la définition de la structure car, comme le prompt, elle peut avoir un impact sur les performances. Par exemple, remarquez la présence de `{200,700}` dans la regex. Cela signifie que le modèle dispose de 200 à 700 caractères pour « raisonner » avant de répondre. La modification de ces valeurs peut avoir un impact sur les performances et conduit à ce que nous appelons le « contrôle de la pensée », un domaine sur lequel nous espérons écrire plus en détail bientôt.

Notre première expérience a consisté à poursuivre l'exploration du jeu de données GSM8K et à itérer sur les *prompts* de 1 à 8 *shots*. Les résultats, présentés ci-dessous, sont très convaincants.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-6.png)

Cette figure présente deux caractéristiques majeures : la variance des performances entre les configurations à *n-shots* a été considérablement réduite et il n'y a eu aucun cas où le classement a changé (le Mistral est toujours en tête devant le Zéphyr). Il convient également de souligner que les performances structurées à *1-shot* sont nettement meilleures que les performances non structurées à *1-shot*, et qu'elles sont comparables aux performances à *5-shots*. Cela nous amène à un autre domaine de recherche que nous appelons « efficacité du *prompt* ».

## Plongée dans les variations des *n-shots* et de leur ordre pour GPQA

Pour l'expérience suivante, nous avons voulu faire varier à la fois les *n-shots* et leur ordre. L'ordre a été contrôlé en définissant la graine utilisée pour mélanger les exemples. Comme indiqué précédemment, seuls les *n-shots* sont mélangés afin que les informations restent constantes d'un *prompt* à l'autre, ce qui signifie que tous les *prompts* à *1-shot* sont les mêmes d'une graine à l'autre. Voici un exemple de l'ordre des *shot* pour un *4-shots* :

| graine | ordre *4-shots* |
| --- | --- |
| 42 | 2-1-3-0 |
| 1337 | 1-0-3-2 |
| 1981 | 3-2-0-1 |
| 1992 | 0-3-1-2 |
| 12345 | 1-0-2-3 |

En outre, afin d'explorer le degré de transférabilité de ces résultats, nous avons modifié la tâche en [*Graduate-Level Google-Proof Q&A Benchmark* (GPQA)](https://arxiv.org/abs/2311.12022). GPQA est une tâche d'évaluation difficile de connaissances à choix multiples. Le format du *prompt* et la structure mise en évidence sont présentés ci-dessous.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-7.png)

Pour l'expérience suivante, nous utilisons spécifiquement le sous-ensemble "diamond" qui représente des questions de haute qualité traitées et nettoyées. Sur les 198 questions de ce jeu de données, nous en avons réservé 8 pour le *prompt* *n-shot* (bien que nous n'ayons jamais utilisé que les 5 premières), puis nous avons évalué les 190 questions restantes.

La grille ci-dessous représente la précision obtenue pour toutes les combinaisons possibles de graines et de *n*, pour les deux modèles, à la fois sans (à gauche) et avec (à droite) la génération structurée.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-9.png)

Une chose qui ressort immédiatement est que les résultats structurés ont tendance à être plus élevés que les résultats non structurés. La moyenne de chaque grille pour les résultats structurés et non structurés est indiquée ci-dessous :

**Moyenne des résultats sur toutes les graines et *n-shot*** :

| modèle | non structuré | structuré |
| --- | --- | --- |
| Mistral-7B-v0.1 | 0,2360 | 0,2935 |
| Zephyr-7b-beta | 0,2387 | 0,3048 |

En outre, pour toutes les valeurs de la grille, nous constatons également une *réduction de la variance* lorsque nous comparons la génération structurée à la génération non structurée. 

**Moyenne des résultats sur toutes les graines et *n-shot*** :
| modèle | non structuré | structuré |
| --- | --- | --- |
| Mistral-7B-v0.1 | 0,0213 | 0,0202 |
| Zephyr-7b-beta | 0,0273 | 0,0180 |

This reduction in variance across the grid is similar to the reduction in variance we saw when looking at just n-shot changes for GSM8K.

Cette réduction de la variance sur l'ensemble de la grille est similaire à la réduction de la variance que nous avons constatée en examinant uniquement les changements de *n-shot* pour  GSM8K.

Bien que l'augmentation de la performance attendue et la diminution de la variance soient des propriétés intéressantes, ce que nous voulons vraiment comprendre, c'est l'impact sur le classement. Dans le graphique suivant, nous examinons ces grilles afin de déterminer lequel des deux modèles serait déclaré vainqueur en utilisant
- A : Zephyr-7b-beta
- B : Mistral-7B-v0.1
- “-” : égalité

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-evaluation/dottxt-structured_output-ex-10.png)

Comme nous pouvons le voir sur ces images, il y a une amélioration majeure dans la constance de l'annonce d'un gagnant lorsque la génération structurée est appliquée. Ces résultats sont conformes à ceux que nous avons obtenus en utilisant GSM8K pour différents *n-shot*.

## Conclusion et travaux futurs

Bien que ces résultats soient incroyablement prometteurs, nous devons encore les explorer sur un plus grand nombre de modèles et de tâches. Ce que nous avons vu jusqu'à présent, c'est que la génération structurée pourrait s'avérer être une partie essentielle de l'évaluation. Le fait d'avoir simultanément *augmenté* le score attendu et *diminué* la variance entre les changements de *prompt* est un résultat très prometteur qui mérite d'être approfondi.
