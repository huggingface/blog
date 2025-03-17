---
title: "Que se passe-t-il avec l'Open LLM Leaderboard ?"
thumbnail: /blog/assets/evaluating-mmlu-leaderboard/thumbnail.png
authors:
- user: clefourrier
- user: SaylorTwift
- user: slippylolo
- user: thomwolf
translators:
- user: lbourdois
---

# Que se passe-t-il avec l'Open LLM Leaderboard ?

Une discussion intéressante a récemment eu lieu sur Twitter suite à la publication de [**Falcon 🦅**](https://hf.co/tiiuae/falcon-40b) et à son ajout à l'[*Open LLM Leaderboard*](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard), un classement public comparant les grands modèles de langage en libre accès.

La discussion a porté sur l'une des quatre évaluations affichées dans le classement : un *benchmark* pour mesurer [*Massive Multitask Language Understanding*](https://arxiv.org/abs/2009.03300) (communément abrégé en MMLU).

La communauté a été étonnée de constater que les résultats du modèle actuellement en tête du classement sur le jeu de données MMLU, le [**LLaMA 🦙**](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), étaient nettement inférieurs à ceux indiqués par les auteurs dans le [papier](https://arxiv.org/abs/2302.13971).
Nous avons donc décidé de nous plonger sur ce point pour comprendre ce qui se passait et comment y remédier 🕳🐇

Dans notre quête, nous avons discuté avec l'excellent [@javier-m](https://hf.co/javier-m) qui a collaboré aux évaluations de LLaMA et l'incroyable [@slippylolo](https://hf.co/slippylolo) de l'équipe Falcon. Ceci étant dit, toutes les erreurs observées ci-dessous doivent nous être attribuées plutôt qu'à eux !
Au cours de ce périple avec nous, vous en apprendrez davantage sur les façons d'évaluer un modèle à partir d'une seule évaluation et sur la question de savoir si vous devez croire ou non les chiffres que vous voyez en ligne et dans les papiers.

Vous êtes prêts ? Alors attachez votre ceinture, nous décollons 🚀

## Qu'est-ce que l’*Open LLM Leaderboard* ?

Tout d'abord, notez que l'[*Open LLM Leaderboard*](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) n'est en fait qu'un *wrapper* exécutant la bibliothèque d'évaluation open-source [*LM Evaluation Harness*](https://github.com/EleutherAI/lm-evaluation-harness) créée par le laboratoire de recherche en IA à but non lucratif [EleutherAI](https://www.eleuther.ai/) connu pour avoir créé [*The Pile*](https://pile.eleuther.ai/) et entraîné [*GPT-J*](https://hf.co/EleutherAI/gpt-j-6b), [*GPT-Neo-X 20B*](https://hf.co/EleutherAI/gpt-neox-20b), et [*Pythia*](https://github.com/EleutherAI/pythia). Une équipe avec de sérieuses références dans le domaine de l'IA !

Ce *wrapper* exécute des évaluations à l'aide d' *Eleuther AI harness* sur le cluster de calcul d'Hugging Face, stocke les résultats dans un jeu de données envoyé sur le Hub, puis qui sont finalement affichés dans un [*Space* dédié](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard).

Pour les modèles LLaMA, les résultats obtenus sur MMLU avec [*LM Evaluation Harness*](https://github.com/EleutherAI/lm-evaluation-harness) diffèrent considérablement des résultats obtenus dans le papier.
Pourquoi cela est-il le cas ?

## 1001 saveurs de MMLU

Il s'avère que l'équipe de LLaMA a adapté une autre implémentation disponible en ligne, à savoir le code d'évaluation proposé par l'équipe de UC Berkeley qui a conçu le *benchmark* MMLU disponible à l'adresse https://github.com/hendrycks/test et que nous appellerons ici l'**Implementation originale**.

En approfondissant la question, nous avons trouvé une autre implémentation intéressante pour MMLU : le code fourni par le [CRFM](https://crfm.stanford.edu/) de Stanford dans son *benchmark* d’évaluation très complet [*Holistic Evaluation of Language Models*](https://crfm.stanford.edu/helm/latest/) que nous appellerons ici l'implémentation **HELM**.

Les *benchmarks* EleutherAI Harness et Stanford HELM sont intéressants car ils rassemblent de nombreuses évaluations dans une seule base de code (y compris MMLU), et donnent ainsi une vue d'ensemble de la performance d'un modèle. C'est la raison pour laquelle l'*Open LLM Leaderboard* intègre de tels *benchmarks* « holistiques » au lieu d'utiliser des bases de code individuelles pour chaque évaluation.

Pour trancher la question, nous avons décidé d'exécuter ces trois implémentations possibles pour MMLU sur un ensemble de modèles afin de les classer en fonction de ces résultats :
- l'implémentation Harness ([commit e47e01b](https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4))
- l'implémentation HELM ([commit cab5d89](https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec))
- l'implémentation originale (compatible avec `transformers` grâce à l'incroyable [@olmer](https://hf.co/olmer) [ici](https://github.com/hendrycks/test/pull/13))

> [!NOTE]
> Notez que l'implémentation Harness a été récemment mise à jour. Plus d'informations à ce sujet à la fin de cet article.

Les résultats sont surprenants : 

![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-01-ter-01.png)

Vous trouverez tous les résultats de l'évaluation à la fin de l'article.

Ces trois implémentations du même *benchmark* donnent des résultats très différents et changent même l'ordre de classement des modèles sur le classement !

Essayons de comprendre d'où vient cette divergence 🕵️  
Mais tout d'abord, comprenons brièvement comment nous pouvons évaluer automatiquement les comportements dans les LLM modernes.

## Comment évaluer automatiquement un LLM d'aujourd'hui
MMLU consiste en des questions à choix multiples, donc un *benchmark* plutôt simple (par rapport aux questions ouvertes), mais comme nous le verrons, cela laisse encore beaucoup de marge pour des différences sur les détails de l'implémentation. Le *benchmark* consiste en des questions à quatre réponses possibles couvrant 57 domaines de connaissances générales regroupés en grandes catégories : « Sciences humaines », « Sciences sociales », « STIM  », etc.

Pour chaque question, une seule des réponses proposées est correcte. Voici un exemple :

```
Question: Glucose is transported into the muscle cells:


Choices:
A. via protein transporters called GLUT4.
B. only in the presence of insulin.
C. via hexokinase.
D. via monocarbylic acid transporters.


Correct answer: A
```

Remarque : vous pouvez explorer facilement ce jeu de données [via le visualiseur de jeu de données](https://hf.co/datasets/cais/mmlu/viewer/college_medicine/dev?row=0) du Hub.

Les grands modèles de langage sont des modèles simples dans le zoo des modèles d'IA. Ils prennent en entrée une *chaîne de texte* (appelée « *prompt* »), qui est découpée en *tokens* (mots, sous-mots ou caractères, selon la manière dont le modèle est construit) et introduite dans le modèle. À partir de cette entrée, ils génèrent une distribution de probabilité pour le *token* suivant, sur l'ensemble des *tokens* qu'ils connaissent (appelés « vocabulaire » du modèle). Vous pouvez donc obtenir le « degré de probabilité » qu'un *token* soit la suite du *prompt* d'entrée.

Nous pouvons utiliser ces probabilités pour choisir un *token*, par exemple le plus probable (ou nous pouvons introduire un léger bruit avec un échantillonnage pour éviter d'avoir des réponses « trop mécaniques »). L'ajout du *token* retenu au *prompt* et sa réintroduction dans le modèle permettent de générer un autre *token* et ainsi de suite jusqu'à ce que des phrases entières soient créées comme des suites du *prompt* d'entrée :

![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-01.png)

C'est de cette façon que ChatGPT ou Hugging Chat génèrent des réponses.
En résumé, nous disposons de deux moyens principaux pour obtenir des informations à partir d'un modèle afin de l'évaluer :
1. obtenir les **probabilités** que certains groupes de *tokens* spécifiques soient des suites du *prompt* puis **comparer ces probabilités ensemble** pour nos choix possibles prédéfinis ;
2. obtenir une **génération de texte** à partir du modèle (en sélectionnant les *tokens* de manière répétée comme nous l'avons vu) puis **comparer ces générations de texte** aux textes des différents choix possibles prédéfinis.

Fort de ces connaissances, nous allons nous plonger dans nos trois implémentations de MMLU, afin de découvrir les données d'entrée envoyées aux modèles, les résultats attendus et la manière dont ces résultats sont comparés.

## MMLU se décline sous toutes les formes et dans toutes les tailles : examen des prompts

Comparons un exemple de *prompt* envoyé aux modèles dans chaque implémentation pour le même jeu de données MMLU :
<div>
<table><p>
  <tbody>
 <tr style="text-align: left;">
  <td>Implementation originale <a href="https://github.com/hendrycks/test/pull/13">Ollmer PR</a></td>
  <td>HELM <a href="https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec">commit cab5d89</a> </td>
  <td>AI Harness <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4">commit e47e01b</a></td>
 </tr>
  <tr style=" vertical-align: top;">
    <td>The following are multiple choice questions (with answers) about  us foreign policy. <br>
How did the 2008 financial crisis affect America's international reputation? <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
    <td>The following are multiple choice questions (with answers) about us foreign policy. <br>
 <br>
Question: How did the 2008 financial crisis affect America's international reputation? <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
    <td>Question: How did the 2008 financial crisis affect America's international reputation? <br>
Choices: <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
  </tr>
  </tbody>
</table><p>
</div>

Les différences peuvent sembler minimes, les avez-vous toutes repérées ? Les voici :
- Première phrase, instruction et sujet :<br>Peu de différences. HELM ajoute un espace supplémentaire, et Harness n'inclut pas la ligne de sujet
- Question :<br>HELM et Harness ajoutent le préfixe « Question : »
- Choix :<br>Harness les fait précéder du mot-clé « Choix »

## Comment évaluer le modèle à partir de ces prompts ?

Commençons par la manière dont [l'implémentation originale de MMLU](https://github.com/hendrycks/test/pull/13) extrait les prédictions du modèle. Dans cette version, nous comparons les probabilités prédites par le modèle, pour les quatre réponses :
![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-02.png)

Cela peut être bénéfique pour le modèle dans certains cas, par exemple, comme vous pouvez le voir ici :

![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-03.png)

Dans ce cas, le modèle a obtenu un +1 pour avoir classé la bonne réponse en tête des 4 options. Mais si nous regardons le vocabulaire complet, il aurait plutôt généré un mot en dehors de nos quatre options : le mot « Zygote » (il s'agit plus d'un exemple que d'un cas d'utilisation réel 🙂).
Comment pouvons-nous nous assurer que le modèle commet le moins possible d'erreurs de ce type ?

Nous pouvons utiliser une approche « ***few-shots*** » dans laquelle nous fournissons au modèle un ou plusieurs exemples dans le *prompt*, avec leurs réponses attendues également. Voici à quoi cela ressemble :
![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-04.png)

Ici, le modèle dispose d'un exemple du comportement souhaité et est donc moins susceptible de prédire des réponses en dehors de la fourchette des réponses attendues.
Puisque cela améliore les performances, dans toutes nos évaluations, MMLU est évalué en 5 coups (en faisant précéder chaque *prompt* de 5 exemples). Notons que d'un *benchmark* à l'autre, bien que les mêmes 5 exemples soient utilisés, leur ordre d'introduction dans le modèle peut varier, ce qui est également une source possible de différence, que nous n'examinerons pas ici. Vous devez également faire attention à ne pas laisser filtrer certaines réponses dans les exemples *few-shot* que vous utilisez...

**HELM**  
Examinons à présent [l’implementation  d’HELM](https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec). Si le *prompt* avec *few-shot* est généralement similaire, la manière dont le modèle est évalué est très différente de l'implémentation originale que nous venons de voir. Nous utilisons les probabilités de sortie du prochain *token* du modèle pour sélectionner une génération de texte et nous la comparons au texte de la réponse attendue :
![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-05.png)

Dans ce cas, si notre *token* « Zygote » était au contraire le plus probable (comme nous l'avons vu plus haut), la réponse du modèle (« Zygote ») serait erronée et le modèle ne marquerait aucun point pour cette question :
![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-06.png)

**Harness**  
Finissons avec [l’implementation de janvier 2023 d’Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4) qui a été utilisé pour calculer les premiers résultats du classement. Comme nous allons le voir, nous avons ici une autre façon de calculer un score pour le modèle sur le même jeu de données d'évaluation.
Nous utilisons à nouveau les probabilités, mais cette fois-ci les probabilités de la séquence de réponses complète, avec la lettre suivie du texte de la réponse, par exemple « *C. The second pharyngeal arch* ». Pour calculer la probabilité d'une réponse complète, nous obtenons la probabilité pour chaque *token* (comme nous l'avons vu ci-dessus) et nous les regroupons. Pour des raisons de stabilité numérique, nous additionnons le logarithme des probabilités et nous pouvons décider (ou non) de calculer une normalisation dans laquelle nous divisons la somme par le nombre de *tokens* afin d'éviter d'avantager les réponses plus longues (plus d'informations à ce sujet ultérieurement). Voici à quoi cela ressemble :

![png](https://hf.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-07.png)

Pour résumer ce que nous avons vu jusqu'à présent, voici un tableau récapitulatif des réponses fournies et générées par le modèle :
<div>
<table><p>
  <tbody>
 <tr style="text-align: left;">
  <td>Implementation originale</td>
  <td>HELM</td>
  <td>AI Harness (janvier 2023)</td>
 </tr>
  <tr style=" vertical-align: top;">
    <td> Nous comparons les probabilités des réponses aux lettres suivantes :
</td>
    <td> Le modèle doit générer comme texte la lettre suivante comme réponse :
</td>
    <td> Nous comparons les probabilités des réponses complètes suivantes :
</td>
  </tr>
  <tr style=" vertical-align: top;">
    <td>  A <br>
 B <br>
 C <br>
 D
</td>
    <td>A
</td>
    <td> A. It damaged support for the US model of political economy and capitalism <br>
 B. It created anger at the United States for exaggerating the crisis <br>
 C. It increased support for American global leadership under President Obama <br>
 D. It reduced global use of the US dollar
</td>
  </tr>
  </tbody>
</table><p>
</div>

Maintenant que nous avons couvert toutes les implémentations, comparons les résultats obtenues via ces trois façons possibles d'évaluer les modèles :

|                                           | MMLU (HELM) | MMLU (Harness) | MMLU (Originale) |
|:------------------------------------------|------------:|---------------:|----------------:|
| llama-65b                     |       **0.637** |          0.488 |           **0.636** |
| tiiuae/falcon-40b                         |       0.571 |          **0.527** |           0.558 |
| llama-30b                     |       0.583 |          0.457 |           0.584 |
| EleutherAI/gpt-neox-20b                   |       0.256 |          0.333 |           0.262 |
| llama-13b                     |       0.471 |          0.377 |           0.47  |
| llama-7b                      |       0.339 |          0.342 |           0.351 |
| tiiuae/falcon-7b                          |       0.278 |          0.35  |           0.254 |
| togethercomputer/RedPajama-INCITE-7B-Base |       0.275 |          0.34  |           0.269 |

Nous pouvons constater que pour le même jeu de données, tant les scores absolus que les classements des modèles (voir la première figure) sont très sensibles à la méthode d'évaluation que nous décidons d'utiliser.

Disons que vous avez entraîné une reproduction parfaite du modèle LLaMA 65B et que vous l'avez évaluée avec Harness (score de 0,488). Vous le comparez maintenant au nombre publié (évalué sur l'implémentation originale de MMLU, donc avec un score de 0,637). Avec une telle différence de 25 points, vous vous dites probablement : « Oh mince, j'ai complètement raté mon entraînement 😱 ». Or, rien n'est plus faux, ces valeurs ne sont pas du tout comparables, même si elles portent toutes deux la mention « score MMLU » (et sont évaluées sur le même jeu de données MMLU).

Existe-t-il une « meilleure façon » d'évaluer un modèle parmi toutes celles que nous avons vues ? C'est une question délicate. Différents modèles peuvent se comporter différemment lorsqu'ils sont évalués d'une manière ou d'une autre, comme nous le voyons ci-dessus lorsque les classements changent. Pour rester aussi équitable que possible, on peut être tenté de choisir une implémentation pour laquelle le score moyen de tous les modèles testés est le plus élevé, de manière à « débloquer » le plus grand nombre possible de capacités des modèles. Dans notre cas, cela signifierait utiliser l'option de log-vraisemblance de l'implémentation originale. Mais comme nous l'avons vu plus haut, son utilisation donne également des indications au modèle en restreignant le champ des réponses possibles, et aide donc peut-être trop les modèles les moins puissants. En outre, la log-vraisemblance est facilement accessible pour les modèles open-source, mais n'est pas toujours indiquée pour les modèles API à source fermée.

Et vous, lecteur, qu'en pensez-vous ? Cet article de blog étant déjà long, il est temps d'ouvrir la discussion et de recueillir vos commentaires. Venez discuter de ce sujet dans le fil de discussion du *Space* de l'*Open LLM Leaderboard* : https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/82

## Conclusion

L'une des principales leçons à retenir est que les évaluations sont fortement liées à leur implémentation, jusque dans les moindres détails tels que les *prompts* et la *tokenisation*. La simple indication de « résultats MMLU » ne vous donne que peu ou pas d'informations sur la façon dont vous pouvez comparer ces chiffres à ceux d'autres bibliothèques que vous avez évaluées.

C'est pourquoi les *benchmarks* ouverts, normalisés et reproductibles tels que l'[EleutherAI Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness/) ou le [Stanford HELM](https://github.com/stanford-crfm/helm/) sont d'une valeur inestimable pour la communauté. Sans eux, la comparaison des résultats entre les modèles et les papiers serait impossible, ce qui étoufferait la recherche sur l'amélioration des LLM.

**Post scriptum** :  
Dans le cas de l'*Open LLM Leaderboard*, nous avons décidé de nous en tenir à l'utilisation de bibliothèques d'évaluation gérées par la communauté. Heureusement, pendant la rédaction de cet article de blog, la formidable communauté autour d'EleutherAI Harness, et en particulier [ollmer](https://github.com/EleutherAI/lm-evaluation-harness/issues/475), a réalisé un travail remarquable en mettant à jour l'évaluation de MMLU pour la rendre similaire à l'implémentation d'origine.

Nous sommes en train de mettre à jour le classement complet avec la version actualisée d'Harness, alors attendez-vous à voir des scores provenant d'Harness v2 dans les prochaines semaines !  
L'exécution de tous les modèles à nouveau prendra un certain temps, restez à l'écoute :hugs:

## Remerciements
Nous sommes très reconnaissants envers Xavier Martinet, Aurélien Rodriguez et Sharan Narang de l'équipe LLama pour leurs suggestions utiles dans cet article ainsi que pour avoir répondu à toutes nos questions.

## Hachages de reproductibilité
Voici les hachages de validation des différentes implémentations de code utilisées dans cet article de blog :
- Implementation d’EleutherAI LM Harness, commit e47e01b: https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4
- Implementation d’HELM, commit cab5d89: https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec
- Implementation originale de MMLU (compatible avec `transformers` par l'incroyable [@olmer](https://hf.co/olmer)) : https://github.com/hendrycks/test/pull/13
