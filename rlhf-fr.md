---
title: "Illustrating Reinforcement Learning from Human Feedback (RLHF)" 
thumbnail: /blog/assets/120_rlhf/thumbnail.png
authors:
- user: natolambert
- user: LouisCastricato
  guest: true
- user: lvwerra
- user: Dahoas
  guest: true
---

# Illustrer le "Reinforcement Learning from Human Feedback" (RLHF), ou l’apprentissage de renforcement avec des retours humains

<!-- {blog_metadata} -->
<!-- {authors} -->

_traduit de [l'original](https://huggingface.co/blog/rlhf) par son [Antoine E. Bachmann](mailto:antoine.bachmann@epfl.ch)_. 

Sur les quelques dernières années les modèles de langage ont montré
d'impressionnantes capacités à générer du texte divers et intriguant à
base d'instructions humaines. Mais, ce qui constitue un « bon » texte
est subjectif et dépendant du contexte, rendant la définition formelle
fort difficile. Il y a de nombreuses applications à la technologie :
l'écriture de fiction où nous voulons de la créativité, les textes
informatifs qui devraient être justes, ou des blocs de code qui
devraient fonctionner.

Écrire une fonction de coût (loss function) pour capturer ces attributs
semble infaisable, et la plupart des modèles de langage sont entraînés
avec une simple prédiction de coût du prochain token(jeton en français).
Pour compenser les limitations du coût furent crées des mesures qui
devraient mieux capturer les préférences humaines comme
[BLEU](https://en.wikipedia.org/wiki/BLEU) ou
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)). Et biens qu'elles
soient plus efficaces que la fonction de coût pour mesurer la
performance, ces mesures comparent le texte généré à une référence avec
des règles très simples, et se trouvent aussi limitées. L'idéal serait
d'utiliser les retours des humains comme mesure de performance, voire
même de s'en servir comme coût pour optimiser le modèle. C'est l'idée du
\"Reinforcement Learning from Human Feedback\" (RLHF) ; utiliser des
méthodes d'aprentissage de renforcement pour directement optimiser un
modèle de langage avec des retours humains. Le RLHF a permis aux modèles
de langage de commencer à aligner un modèle entraîné sur un corpus
général de données textuelles aux complexes valeurs humaines.

Le succès le plus récent de RLHF a été son usage dans
[ChatGPT](https://openai.com/blog/chatgpt/). Et vu les compétences
impressionnantes de chatGPT, nous lui avons demandé de nous explique le
RLHF :

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/chatgpt-explains.png" width="500" />
</p>

Il se débrouille étonnamment bien, mais ne
couvre pas tous les détails. Nous oui.

# RLHF: étape-par-étape

Le RLHF est un concept difficile car il implique un entraînement avec de
multiples modèles et plusieurs phases de déploiement. Dans ce post, nous
allons séparer le procédé en trois étapes fondamentales.

1. Pré-entraîner un modèle de langage(LM, langage model),
2. Connecter des données et entraîner un modèle de récompense(reward
model), et
3. Ajuster(fine tuning) le LM avec l'apprentissage de renforcement.

Pour commencer, regardons comment est entraîné un modèle de langage.

### Pré-entraîner un modèle de langage

Comme point de départ pour le RLHF, utilisez un modèle de langage qui a
déjà été entraîné de façon classique (voyez ce blog pour plus de
détails). OpenAi a usé d'une plus petite versionde GPT-3 pour son
premier modèle RLHF populaire,
[InstructGPT](https://openai.com/blog/instruction-following/). Anthropic
a usé de modèles à transformeurs allant de 10 millions à 52 milliards DE
paramètres pour cette tâche. Deepmind utilisèrent leur modèle à 280
milliards de paramètres, [Gopher](https://arxiv.org/abs/2112.11446).

Ce modèle initial peut être ajusté sur du texte ou des conditions
supplémentaires, mais ce n'est pas forcément le cas. Par exemple, OpenAI
on ajusté sur du texte humain qui était « Préférable » et Anthropic on
généré le leur en distillant un LM original sur des indices de contexte
pour leur criètres d' « adjuvant, honnête et innofensif ». Ce sont
toutes les deux des sources de données que je peux appeles chères et
augmentées, mais ce ne sont pas des techniques requires pour comprendre
le RLHF.

En général il n'y a pas de réponse claire sur le « meilleur » modèle
pour commencer RLHF. Ce sera un fil rouge de ce blog, les options de
design en RLHF de sont pas explorées exhaustivement.

Ensuite, avec un modèle de langage, nous devons générer des données pour
entraîner un modèle de récompense(reward model), qui est notre outil
pour intégrer des préférences humaines dans le modèle.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/pretraining.png" width="500" />
</p>

### Entraînement du modèle de récompense
    
Générer un modèle de récompense (RM ou reward model en anglais) calibré
avec des préférences humaines est là ou commence la recherche récente en
RLHF. L'objectif sous-jacent est d'obtenir un mpdèle ou système qui
prend en entrée une séquence de texte, et retourne une récompense
scalaire qui devrait numériquement représenter la préférence humaine. Le
système peut être un LM du début à la fin, ou un système modulaire, par
exemple le modèle classe les outputs, et la récompense dépend du
classement. L'output doit absolument être une **récompense scalaire**
pour pouvoir intégrer l'apprentissage de renforcement.

Les LM pour la construction de la récompense peuvent être d'autres LM
ajustés, ou des LM entraînés à partir du set de données préférentielles
uniquement. Par exemple, Anthropic utilise une méthode d'ajustement
spécialisée pour initialiser ces modèles après le pré-entraînement
(PMP,. Ou preference model pretraining) car ils ont trouvé la méthode
plus efficace que l'ajustement, mais aucune variante de stratégie n'est
considérée comme clairement supérieure aux autres.

Le set de données d'entraînement pour les paires prompt-génération pour
les RM sont générées en échantillonnant un set de prompts d'un dataset
prédéfini (Anthropic génère ses données primairement avec un outil de
chat sur Amazon Mechanical Turk est
[disponible](https://huggingface.co/datasets/Anthropic/hh-rlhf), et
OpenAI utilise des prompts soumis par les utilisateurs de l'API GPT).
Les prompts sont envoyés dans le LM initial pour générer du nouveau
texte.

Des annotateurs humaine sont utilisés pour classer les outputs du LM.
L'on pourrait supposer que les humains pourraient directement appliquer
un score scalaire, mais c'est difficile à accomplir en pratique. L
différence des valeurs humaines peut causer des déséquilibres et du
bruit dans ces scores. Le classement permet de créer quelque chose de
beaucoup plus régulier, et de facilement comparer différents modèles.

Il y a plusieurs méthodes de classement. L'une des efficaces et de
demander aux utilisateurs de comparer directement le texte sortie de
deux LM répondant au même prompt. En comparant ces outputs face-à-face,
un système d'ELO peut être utilise pour générer le calssement des
modèles et des outputs. Ces différentes méthodes sont ensuite
normalisées dans une récompense scalaire.

Un artefact intéressant de ce procédé est que les systèmes de RLHF qui
ont eu du succès, one utilisé des RM de taille variable par rapport à la
génération de texte. (OpenAI 175mia LM, 6mia RM, Anthropic utilise des
LM et RM de 10 à 52mia et DeepMind chinchilla de 70 mia pour LM et RM).
L'intuition voudrait que ces modèles de préférence aient besoin une
capacité similaire à celle d'un modèle pour pouvoir comprendre le texte.

A ce point dans le système RLHF, nous avons un modèle de langage qui
peut être utilisé pour générer du texte, et un modèle de préférence qui
peut prendre n'importe quel texte et lui assigner un score
d'appréciation humaine. Ensuite, nous utilisons l'apprentissage de
renforcement(Reinforcement learning ou RL) pour optimiser le modèle de
langage grâce au modèle de récompense.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png" width="600" />
</p>

### L'ajustement avec RL

Entraîner un modèle de langage avec de l'apprentissage de renforcement
était, pendant longtemps, pensé impossible pour des raisons techniques
et algorithmiques. Ce que plusieurs organisations on apparemment réussi
à faire fonctionner est l'ajustement des paramètres d'une copie du LM de
base avec un algorithme RL, Proximal Policy Organisation (PPO ou
organisation proximale des règles). Des paramètres du LM sont gelés, car
ajuster un modèle avec des milliards de paramètres est prohibitivement
coûteux. (pour en savoir plus, vous pouvez lire [Low-Rank
Adaptation](https://arxiv.org/abs/2106.09685) for LM ou le LM
[Sparrow](https://arxiv.org/abs/2209.14375) de deepmind). PPO existe
depuis longtemps et il y a
[quantité](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
de [guides](https://huggingface.co/blog/deep-rl-ppo) qui expliquent son
fonctionnement. La relative maturité de cette méthode en a fait un choix
favorable pour agrandis la nouvelle application de l'entraînement
distribué pour le RLHF. Il se retrouve que beaucoup des avancées
cruciales en RL pour le RLHF ont travaillé à mettre un grand modèle à
jour avec un algorithme familier. (Nous en reparlerons plus tard)

Commençons par formuler cette tâche d'ajustement comme une tâche de RL.
Premièrement, la police est un LM qui prend un prompt et retourne une
séquence de texte (ou des distributions de probabilités sur du texte).
L'espace d'action de cette police est tous les tokens correspondant au
vocabulaire de ce langage (dans l'ordre de cinquante mille), et l'espace
d'observation est la distribution des chaînes de tokens possibles en
input, ce qui est un assez grand ensemble (la dimension est
approximativement la taille du vocabulaire \^ la longueur des inputs).
La fonction de récompense est une combinaison du modèle de préférence et
une contrainte sur le changement de police.

La fonction de récompense est où le système combine tous les modèles
dont nous avons discuté en un procédé RLHF. Soit un prompt, x, du
dataset, deux textes, y1, y2, sont générés- un du modèle initial, et
l'autre de notre police ajustée. Le texte de la police actuelle est
passé au modèle de préférence, qui retourne un scalaire de «
préférabilité », $r_{\Theta}$. Ce texte est comparé au texte initial
pour calculer une pénalité sur la différence entre eux. Dans plusieurs
papers d'OpenAI, Anthropic, et DeepMind, cette pénalité est une version
à l'échelle de [la divergence Kullback-Leibler
(KL)](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) entre
ces séquences, $r_{KL}$. Cette divergence pénalise la police RL pour
éviter qu'elle bouge trop loin du modèle initial à chaque itération, ce
qui peut être utile pour maintenir la cohérence des outputs. Sans cette
pénalité, l'optimisation peux commencer à générer du texte illisible qui
trompe notre RM. En pratique, la divergence KL est approximée par
échantillonnage des deux distributions([expliqué par John
Schulman](http://joschu.net/blog/kl-approx.html)). Le produit final
envoyé au RL est : $r = r_{\Theta} - \lambda r_{KL}$.

Certains systèmes RLHF ajoutent des termes à la fonction de récompense.
Par exemple, OpenAI ont accompli plusieurs expériences sur InscructGPT,
en y ajoutant des gradients pré-entrainement (du set d'annotations
humaines) dans la règle de mise à jour de PPO. Il est probable que à la
mesure de nos avancées en RLHF que cette fonction continuera à évoluer.

Finalement, la règle d'update est la mise à jour venant de PPO qui
maximise les récompenses pour les données actuelles (PPO est sur la
police, ce qui veut dire que les paramètres sont seulement mis à jour
sur les données actuelles). PPO est un algorithme avec des régions de
confiance qui utilise des contraintes sur le gradient pour assurer que
l'étape de mise à jour ne déstabilise pas le procédé d'apprentissage.
DeepMind utilise une mise en place similaire pour Gopher, mais a choisi
[synchronous advantage author
critic](http://proceedings.mlr.press/v48/mniha16.html?ref=https://githubhelp.com)
(A2C) pour optimiser les gradient, ce qui est trèa différent, mais n'a
pas été reproduit en externe.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png" width="650" />
</p>

Optionnellement, RLHF peut continuer à
partir de cette étape en faisant des updates itératifs du RM et de la
police ensemble. À mesure que la police RL se met à jout, les
utilisateurs peuvent continuer à classer ces outputs contre les
anciennes versions du modèle. La plupart des publications n'ont pas
encore discuté de cette approche, car le déploiement requis pour
collecter ce genre de données ne marche que pour des agents de dialogue
avec une base utilisateur large et active. Anthropic discute de cette
approche sous le nom [Iterated Online
RLHF](https://arxiv.org/abs/2204.05862), ou les itérations de la police
sont incluses dans le classement ELO à travers les modèles, ceci induit
des dynamiques très complexes sur la police et l'évolution du modèle de
récompense, c'est une question de recherche encore très ouverte.

# Open source pour RLHF

Le premier [code](https://github.com/openai/lm-human-preferences) publié
pour faire du RLHF sur des LM est TensorFlow par OpenAi en 2019.

Aujourd'hui, il y a déjà plusieurs répositoires actifs pour RLHF sur
PyTorch qui en sont sortis. Les répositoires primaires sont Transfom
Transformers Reinforcement Learning
([TRL](https://github.com/lvwerra/trl)),
[TRLX](https://github.com/CarperAI/trlx) a commencé comme une forme de
TRL, et Reinforcement Learning for Language models
([RL4LMs](https://github.com/allenai/RL4LMs)).

TRL est conçu pour ajuster des Lms pré-entraînes dans l'écosystème
huggingface avec PPO. TRLX ext un for étendu de TRL construit par
[CarperAI](https://carper.ai/) pour soutenir des plus grand modèles en
et hors ligne en entraînement. Pour le moment, TRLX à une API capable de
construire des modèles prêts à la production avec RLHF, PPO et Implicit
Language Q-Learning aux échelles requises pour des LLM(large langage
models, ou grand modèles de langage, 33+mia paramètres). De fitures
versions de TRLX permettront 200mia et plus de paramètres. De ce fait,
TRLX est optimisé pour des ingénieurs qui ont l'habitude de telles
échelles.

[RL4LM](https://github.com/allenai/RL4LMs) offre des outils pour ajuster
et évaluer des LLM avec un large éventail d'algorithmes RL (PPO, NLPO,
A2C et TRPO), fonctions de récompense et mesures. De plus, la librarie
est facilement customisable, ce qui permet d'entraîner n'importe quel
MML basé sur les transofrmeurs et un modèle décodeur-encodeur. Il est
notablement bien testé et évalué sur de nombreuses tâches dans des
[travaux récents](https://arxiv.org/abs/2210.01241) comptant pour plus
de 2000 expériences qui surlignent plusieurs informations pratiques sur
la comparaison des budgets de données (démonstrations expertes vs
modelage de récompense), la gestion du contournement de récompense et
les instabilités d'entraînement, etc. RL4M a des plans pour inclure
l'entraînement distribué de plus gros modèles, et nouveaux algorithmes
de RL.

TRLX et RL4LM sont en développement très actif, donc attendez vous à de
nouvelles features dans un futur proche.

Il y a un grand
[dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) crée par
Anthropic disponible sur le Hub.

# Quelle est la prochaine étape pour RLHF?

Bien que ces techniques soient extrêmement prometteuses et impactantes,
et elles ont capté l'attention des plus grands laboratoires
d'intelligence artificielle. Mais il reste des limitations claires. Les
modèles, bien que plus performants, peuvent toujours produire des
contenus dangereux ou factuellement incorrects sand aucune certitude.
Cette imperfection représente un défi long-terme et motivation pour le
RLHF- opérer sur un problème fondamentalement humain veut dire qu'il n'y
aura jamais le ligne claire pour qu'un modèle soit « complet ».

Quand l'on déploie un système en utilisant RLHF, obtenir les préférences
humaines est coûteux et difficile vu toute la main d'œuvre humaine
requise. De plus, la performance est limitée par la qualité des
annotations humaines, qui sont en deux variétés : le texte généré par
des humains, comme celui utilisé pour ajuster InstructGPT, et les labels
de préférence humaine entre textes générés.

Générer du texte humain de qualité répondant à des prompts spécifiques
est très coûteux, car il est presque obligatoire d'engager plutôt que de
crowdsource. Heureusement, l'échelle des données utilisées dans
l'entraînement du RM pour la plupart des applications de RLHF (50k
échantillons labellisés) n'est pas si cher. Mais, cela reste un coût
plus élevé que ce que la plupart des laboratoires académiques peuvent se
permettre. Actuellement, il existe un seul grand set de données pour
RLHF sur un LM général
([d'Anthropic](https://huggingface.co/datasets/Anthropic/hh-rlhf)), et
quelques sets plus petits
([summarization](https://github.com/openai/summarize-from-feedback)
d'openAI). Le second défi pour RLHF est que les annotateurs humains
peuvent être en désaccord, ajoutant une variance potentielle assez
importante sur les données sans vérité objective.

Avec ces limitations, il y a encore d'immenses possibilités de design
inexplorées qui pourraient encore avancer RLHF. Plusieurs d'entre elles
tombent sous l'amélioration de RL. PPO est un algorithme relativement
vieux, mais il n'y a pas de raisons structurelles pour lesquelles un
autre algorithme ne pourrait pas offrir des avantages. Un grand coût du
feedback est que chaque texte généré doit être évalué sur le RM étant
donné qu'il agit comme une parite de l'environnement sur l'algorithme
RL. Pour éviter ces passes en avant sur un grand modèle, le RL offline
peut être utilisé comme optimisateur de police. Récemment de nouveaux
algorithmes ont émergé, comme [implicit language
Q-learning](https://arxiv.org/abs/2206.11871) (ILQL, un
[talk](https://youtu.be/fGq4np3brbs) sur ILQL par CarperAI), qui combine
aprticulièrement bien avec ce genre d'optimisation. D'autres trade-offs
dans le procédé RL, comme l'équilibre entre exploitation-exploration,
n'ont pas non plus été documentées. Explorer ces directions permettrait
au moins de développer une compréhensions substantielle du
fonctionnement des RLHF ou au moins améliorer leurs performances.

**Citation:**
If you found this useful for your academic work, please consider citing our work, in text:
```
Lambert, et al., "Illustrating Reinforcement Learning from Human Feedback (RLHF)", Hugging Face Blog, 2022.
```

BibTeX citation:
```
@article{lambert2022illustrating,
  author = {Lambert, Nathan and Castricato, Louis and von Werra, Leandro and Havrilla, Alex},
  title = {Illustrating Reinforcement Learning from Human Feedback (RLHF)},
  journal = {Hugging Face Blog},
  year = {2022},
  note = {https://huggingface.co/blog/rlhf},
}
```