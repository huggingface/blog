---
title: "Banque des Territoires (CDC Group) x Polyconseil x Hugging Face: Enhancing a Major French Environmental Program with a Sovereign Data Solution" 
thumbnail: /blog/assets/78_ml_director_insights/cdc_poly_hf.png
authors:
- user: AnthonyTruchet-Polyconseil
  guest: true
- user: jcailton
  guest: true
- user: StacyRamaherison
  guest: true
- user: florentgbelidji
- user: Violette
---

# Banque des Territoires (CDC Group) x Polyconseil x Hugging Face: Enhancing a Major French Environmental Program with a Sovereign Data Solution


## Table of contents

- Case Study in English - Banque des Territoires (CDC Group) x Polyconseil x Hugging Face: Enhancing a Major French Environmental Program with a Sovereign Data Solution
    - [Executive summary](#Executive-summary)   
    - [The power of RAG to meet environmental objectives](#The-power-of-RAG-to-meet-environmental-objectives)
    - [Industrializing while ensuring performance and sovereignty](#Industrializing-while-ensuring-performance-and-sovereignty)
    - [A modular solution to respond to a dynamic sector](#A-modular-solution-to-respond-to-a-dynamic-sector)
    - [Key Success Factors Success Factors](#Key-Success-Factors)
- Case Study in French - Banque des Territoires (Groupe CDC) x Polyconseil x Hugging Face : améliorer un programme environnemental français majeur grâce à une solution data souveraine
    - [Résumé](#Résumé)
    - [La puissance du RAG au service d'objectifs environnementaux](#La-puissance-du-RAG-au-service-d-'-objectifs-environnementaux)
    - [Industrialiser en garantissant performance et souveraineté](#Industrialiser-en-garantissant-performance-et-souveraineté)
    - [Une solution modulaire pour répondre au dynamisme du secteur](#Une-solution-modulaire-pour-répondre-au-dynamisme-du-secteur)
    - [Facteurs clés de succès](#Facteurs-clés-de-succès)

## Executive summary

The collaboration initiated last January between Banque des Territoires (part of the Caisse des Dépôts et Consignations group), Polyconseil, and Hugging Face illustrates the possibility of merging the potential of generative AI with the pressing demands of data sovereignty.

As the project's first phase has just finished, the tool developed is ultimately intended to support the national strategy for schools' environmental renovation. Specifically, the solution aims to optimize the support framework of Banque des Territoires’ EduRénov program, which is dedicated to the ecological renovation of 10,000 public school facilities (nurseries, grade/middle/high schools, and universities).

This article shares some key insights from a successful co-development between:
- A data science team from Banque des Territoires’ Loan Department, along with EduRénov’ Director ;
- A multidisciplinary team from Polyconseil, including developers, DevOps, and Product Managers ;
- A Hugging Face expert in Machine Learning and AI solutions deployment.

## The power of RAG to meet environmental objectives

Launched by Banque des Territoires (BdT), EduRénov is a flagship program within France's ecological and energy transformation strategy. It aims to simplify, support, and finance the energetic renovation of public school buildings. Its ambition is reflected in challenging objectives: assisting 10,000 renovation projects, from nurseries to universities - representing 20% of the national pool of infrastructures - to achieve 40% energy savings within 5 years. Banque des Territoires mobilizes unprecedented means to meet this goal: 2 billion euros in loans to finance the work and 50 million euros dedicated to preparatory engineering. After just one year of operation, the program signed nearly 2,000 projects but aims to expand further. As program director Nicolas Turcat emphasizes: 

> _'EduRénov has found its projects and cruising speed; now we will enhance the relationship quality with local authorities while seeking many new projects. We share a common conviction with Polyconseil and Hugging Face: the challenge of ecological transition will be won by scaling up our actions.'_

The success of the EduRénov program involves numerous exchanges - notably emails - between experts from Banque des Territoires, Caisse des Dépôts Group (CDC) leading the program, and the communities owning the involved buildings. These interactions are crucial but particularly time-consuming and repetitive. However, responses to these emails rely on a large documentation shared between all BdT experts. Therefore, a Retrieval Augmented Generation (RAG) solution to facilitate these exchanges is particularly appropriate.

Since the launch of ChatGPT and the growing craze around generative AI, many companies have been interested in RAG systems that leverage their data using LLMs via commercial APIs. Public actors have shown more measured enthusiasm due to data sensitivity and strategic sovereignty issues.

In this context, LLMs and open-source technological ecosystems present significant advantages, especially as their generalist performances catch up with proprietary solutions currently leading the field. Thus, the CDC launched a pilot data transformation project around the EduRénov program, chosen for its operational criticality and potential impact, with an unyielding condition:  to guarantee the sovereignty of compute services and models used.

## Industrializing while ensuring performance and sovereignty

Before starting the project, CDC teams experimented with different models and frameworks, notably using open-source solutions proposed by Hugging Face (Text Generation Inference, Transformers, Sentence Transformers, Tokenizers, etc.). These tests validated the potential of a RAG approach. The CDC, therefore, wished to develop a secure application to improve the responsiveness of BdT's support to communities.

Given Caisse des Dépôts (CDC) status in the French public ecosystem and the need to ensure the solution’s sovereignty and security for manipulated data, the CDC chose a French consortium formed by Polyconseil and Hugging Face. Beyond their respective technical expertise, the complementarity of this collaboration was deemed particularly suited to the project's challenges.

- Polyconseil is a technology firm that provides digital innovation expertise through an Agile approach at every stage of technically-intensive projects. From large corporations to startups, Polyconseil partners with clients across all sectors, including ArianeGroup, Canal+, France Ministry of Culture, SNCF, and FDJ. Certified Service France Garanti, Polyconseil has demonstrated expertise in on-premise and cloud deployment ([AWS Advanced Tier Services partner and labeled Amazon EKS Delivery](https://www.linkedin.com/feed/update/urn:li:activity:7201588363357827072/), GCP Cloud Architect, Kubernetes CKA certified consultants, etc.). The firm thus possesses all the necessary resources to deploy large-scale digital projects, with teams composed of Data Scientists, Data Engineers, full-stack/DevOps developers, UI/UX Designers, Product Managers, etc. Its generative AI and LLM expertise is based on a dedicated practice: Alivia, through the [Alivia App](https://www.alivia.app/), plus custom support and implementation offers.

- Founded in 2016, Hugging Face has become, over the years, the most widely used platform for AI collaboration on a global scale. Initially specializing in Transformers and publisher of the famous open-source library of the same name, Hugging Face is now globally recognized for its platform, the 'Hub',  which brings together the machine learning community. Hugging Face offers widely adopted libraries, more than 750,000 models, and over 175,000 datasets ready to use. Hugging Face has become, in a few years, an essential global player in artificial intelligence. With the mission to democratize machine learning, Hugging Face now counts more than 200,000 daily active users and 15,000 companies that build, train, and deploy models and datasets.

## A modular solution to respond to a dynamic sector

The imagined solution consists of an application made available to BdT employees, allowing them to submit an email sent by a prospect and automatically generate a suitable and sourced project response based on EduRénov documentation. The agent can then edit the response before sending it to their interlocutor. This final step enables alignment with the agents' expectations using a method such as Reinforcement Learning from Human Feedback (RLHF).

The following diagram illustrates this:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llama2-non-engineers/diagram_en.png" alt="RLHF" width=90%>
</p>

### Diagram explanation

1. A client sends a request by email through existing channels.
2. This request is transferred to the new user interface.
3. Call to the Orchestrator, which builds a query based on an email for the Retriever.
4. The Retriever module finds the relevant contextual elements indexed by their embeddings from the vector database.
5. The Orchestrator constructs a prompt incorporating the retrieved context and calls the Reader module by carefully tracing the documentary sources.
6. The Reader module uses an LLM to generate a response suggestion, which is returned to the agent via the user interface.
7. The agent evaluates the quality of the response in the interface, then corrects and validates it. This step allows for the collection of human intelligence feedback.
8. The response is transferred to the messaging system for sending.
9. The response is delivered to the client, mentioning references to certain sources.
10. The client can refer to the public repository of used documentary resources.

To implement this overall process, four main subsystems are distinguished:
- In green: the user interface for ingesting the documentary base and constituting qualitative datasets for fine-tuning and RLHF.
- In black: the messaging system and its interfacing.
- In purple: the Retrieval Augmented Generation system itself.
- In red: the entire pipeline and the fine-tuning and RLHF database.

## Key Success Factors Success Factors

The state-of-the-art in the GenAI field evolves at a tremendous pace; making it critical to modify models during a project without significantly affecting the developed solution. Polyconseil designed a modular architecture where simple configuration changes can adjust the LLM, embedding model, and retrieval method. This lets data scientists easily test different configurations to optimize the solution's performance. Finally,  this means that the optimal open and sovereign LLM solution to date can be available in production relatively simply.

We opted for a [modular monolith](https://www.milanjovanovic.tech/blog/what-is-a-modular-monolith) in [hexagonal architecture](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/) to optimize the design workload. However, as the efficient evaluation of an LLM requires execution on a GPU, we outsourced LLM calls outside the monolith. We used Hugging Face's [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index), which offers a highly performant and configurable dockerized service to host any LLM available on the Hub.

To ensure data independence and sovereignty, the solution primarily relies on open-source models deployed on a French cloud provider: [NumSpot](https://numspot.com/). This actor was chosen for its SecNumCloud qualification, backed by Outscale's IaaS, founded by Dassault Systèmes to meet its own security challenges.

Regarding open-source solutions, many French tools stand out. In particular, the unicorn [Mistral AI](https://mistral.ai/fr/) is one of them, whose [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model is currently used within the system’s Reader. However, other more discreet yet specific projects present strong potential to meet our challenges, such as [CroissantLLM](https://huggingface.co/blog/manu/croissant-llm-blog), which we are evaluating. This model results from a collaboration between the [MICS laboratory](https://www.mics.centralesupelec.fr/) of CentraleSupélec and [Illuin Technology](https://www.illuin.tech/). They aim to provide an ethical, responsible, and performant model tailored to French data.

Organizationally, we formed a single Agile team operating according to a flexible ScrumBan methodology, complemented by a weekly ritual of monitoring and training on AI breakthroughs. The latter is led by the Hugging Face expert from its [Expert Support program](https://huggingface.co/support). This structure facilitates a smooth transfer of skills and responsibilities to the BdT Data teams while ensuring regular and resilient deliveries amidst project context changes. Thus, we delivered an early naive MVP of the solution and both qualitative and quantitative evaluation notebooks. To this end, we utilize open-source libraries specializing in the evaluation of generative AI systems, such as RAGAS. This serves as the foundation upon which we iterate new features and performance improvements to the system.

Final Words from Hakim Lahlou, OLS Groups Innovation and Strategy Director at Banque des Territoires loan department:
> _'We are delighted to work at Banque des Territoires alongside these experts, renowned both in France and internationally, on a cutting edge fully sovereign data solution. Based on this pilot program, this approach opens a new pathway: this is likely how public policies will be deployed in the territories in the future, along with the necessary financing for the country's ecological and energy transformation. Currently, this approach is the only one that enables massive, efficient, and precise deployment.'_

_Are you involved in a project that has sovereignty challenges? Do you want to develop a solution that leverages the capabilities of LLMs? Or do you simply have questions about our services or the project? Reach out to us directly at alivia@polyconseil.fr._

_If you are interested in the Hugging Face Expert Support program for your company, please contact us [here](https://huggingface.co/contact/sales?from=support) - our sales team will get in touch to discuss your needs!_

---

# Banque des Territoires (Groupe CDC) x Polyconseil x Hugging Face : améliorer un programme environnemental français majeur grâce à une solution data souveraine

## Résumé

La collaboration lancée en janvier dernier entre la Banque des Territoires de la Caisse des Dépôts et Consignations (CDC), Polyconseil et Hugging Face démontre qu’il est possible d’allier le potentiel de l’IA générative avec les enjeux de souveraineté.

Alors que la première phase du projet vient d’aboutir, l’outil développé doit, à terme, soutenir la stratégie nationale de rénovation environnementale des établissements scolaires. Plus précisément, la solution vise à optimiser le parcours d'accompagnement du Programme EduRénov de la Banque des Territoires (BdT), dédié à la rénovation écologique de 10 000 écoles, collèges et lycées.

Cet article partage quelques enseignements clés d'un co-développement fructueux entre :
- une équipe data science de la Direction des Prêts de la Banque des Territoires ainsi que le Directeur du Programme EduRénov ;
- une équipe pluridisciplinaire de Polyconseil comprenant développeurs, DevOps et Product Manager ;
- un expert Hugging Face en déploiement de solutions de Machine Learning et d’IA.

## La puissance du RAG au service d'objectifs environnementaux

Mis en place par la Banque des Territoires, EduRénov est un programme phare de la stratégie de transformation écologique et énergétique française. Il vise à simplifier, accompagner et financer les démarches de rénovation énergétique des bâtiments scolaires publics. L’ambition se traduit par des objectifs exigeants : 10 000 projets de rénovation d’écoles, collèges, lycées, crèches ou universités - soit 20% du parc national - accompagnés afin qu’ils puissent réaliser 40% d’économie d’énergie en 5 ans. Pour y répondre, la Banque des Territoires mobilise des moyens d’action inédits : une enveloppe de 2 milliards d’euros de prêts pour financer les travaux et 50 millions d’euros dédiés à l’ingénierie préparatoire. Après seulement un an d’existence, le programme compte déjà presque 2 000 projets mais conforte les moyens de ses ambitions ; comme le souligne le directeur du programme Nicolas Turcat : 

> _'EduRénov a trouvé ses projets et son rythme de croisière, désormais nous allons intensifier la qualité de la relation avec les collectivités tout en allant chercher (beaucoup) de nouveaux projets. Nous portons une conviction commune avec Polyconseil et Hugging Face : le défi de la transition écologique se gagnera par la massification des moyens d’action.'_

Le succès du programme EduRénov passe par de nombreux échanges - notamment de courriels - entre les experts de la Banque des Territoires, le Groupe Caisse des Dépôts qui conduit le programme, et les collectivités qui détiennent ce patrimoine à rénover. Ces interactions sont cruciales, mais particulièrement chronophages et répétitives. Néanmoins, les réponses à ces courriels reposent sur une base documentaire large et commune à tous les experts de la BdT. Une solution à base de Retrieval Augmented Generation (RAG) pour faciliter ces échanges est donc particulièrement adaptée.

Depuis le lancement de ChatGPT et le début de l’engouement autour de l’IA générative, de nombreuses entreprises se sont intéressées aux systèmes RAG pour valoriser leurs bases documentaires en utilisant simplement des LLMs via leurs APIs commerciales. Compte tenu de la sensibilité de leurs données et d'enjeux stratégiques de souveraineté, l’enthousiasme est resté plus mesuré du côté des acteurs publics.

Dans ce contexte, les LLMs et les écosystèmes technologiques open source présentent des avantages significatifs, et ce d'autant plus que leurs performances généralistes rattrapent celles des solutions propriétaires, leaders du domaine. C'est ainsi que la CDC a décidé de lancer un projet de transformation data pilote autour du programme EduRénov, choisi pour sa criticité opérationnelle et son impact potentiel, en imposant une condition essentielle : garantir le caractère souverain du cloud et des modèles utilisés dans ce cadre.

## Industrialiser en garantissant performance et souveraineté

À la genèse du projet, les équipes de la CDC ont expérimenté avec différents modèles et frameworks, notamment à l’aide des solutions open source proposées par Hugging Face (Text Generation Inference, Transformers, Sentence Transformers, Tokenizers, etc.). Ces tests ont validé le potentiel de l’approche RAG envisagée. La CDC a donc souhaité développer une application sécurisée permettant d’améliorer la réactivité d’accompagnement des collectivités par la Banque des Territoires.

Compte tenu du statut de la Caisse des Dépôts dans l’écosystème public français, et afin de garantir la souveraineté de la solution et la sécurité des données travaillées, elle a choisi de s’orienter vers le groupement français constitué par Polyconseil et Hugging Face. Au-delà des expertises techniques respectives, la complémentarité de cette collaboration a été jugée particulièrement adaptée aux enjeux du projet.

- Polyconseil est un cabinet d’experts en innovation numérique qui agit de manière Agile sur chaque étape de projets à forte composante technique. Du grand compte à la startup, Polyconseil intervient pour des clients de tous secteurs d’activité, tels que ArianeGroup, Canal+, le Ministère de la Culture, la SNCF, la FDJ, etc. Certifié Service France Garanti, Polyconseil dispose d’une expertise éprouvée sur le déploiement on-premise et sur clouds ([AWS Advanced Tier Services partner et labellisé Amazon EKS Delivery](https://www.linkedin.com/feed/update/urn:li:activity:7201588363357827072/), consultants certifiés GCP Cloud Architect, Kubernetes CKA, etc.). Le cabinet possède ainsi l’ensemble des ressources nécessaires au déploiement de projets numériques d’envergure, avec des équipes de Data Scientists, Data Engineers, développeurs full stack /DevOps, UI/UX Designers, Product Managers, etc. L’expertise en matière d’IA générative et de LLM repose sur une practice dédiée : Alivia, au travers de la solution [Alivia App](https://www.alivia.app/) et d’offres d’accompagnement et de mise en œuvre sur-mesure.

- Fondée en 2016, Hugging Face est devenue au fil des années la plateforme la plus utilisée pour la collaboration sur l’Intelligence Artificielle à l’échelle mondiale. Hugging Face, d’abord spécialiste des Transformers et éditeur de la célèbre librairie Open-Source éponyme, est maintenant reconnue mondialement pour sa plateforme, le « Hub », qui rassemble la communauté du machine learning. Proposant à la fois des bibliothèques très largement adoptées, plus de 750 000 modèles, et plus de 175 000 jeux de données (datasets) prêts à l'emploi, Hugging Face est devenue en quelques années un acteur mondial incontournable en intelligence artificielle. Avec pour mission de démocratiser le machine learning, Hugging Face compte aujourd'hui plus de 200 000 utilisateurs actifs quotidiens et 15 000 entreprises qui construisent, entraînent et déploient des modèles et des ensembles de données. 

## Une solution modulaire pour répondre au dynamisme du secteur

La solution imaginée consiste en une application mise à disposition des collaborateurs de la Banque des Territoires, qui leur permet de soumettre un courriel envoyé par un prospect et de générer automatiquement un projet de réponse adapté et sourcé, basé sur la documentation métier. L’agent peut ensuite éditer la réponse avant de l’envoyer à son interlocuteur. Cette dernière étape permet d’envisager une phase d’alignement aux attentes des agents du système à l’aide grâce à différentes techniques comme “Reinforcement Learning from Human Feedback” (RLHF). 

Elle est illustrée par le schéma suivant :

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llama2-non-engineers/diagram_fr.png" alt="RLHF" width=90%>
</p>

### Explication du schéma

1. Un client envoie une demande par courriel selon les canaux existants.
2. Cette demande est transférée dans la nouvelle interface utilisateur 
3. Le module Retriever retrouve les éléments de contexte pertinents, indexés par leur embedding, depuis la base de données vectorielle.
4. The Retriever module finds the relevant contextual elements indexed by their embeddings from the vector database.
5. L'Orchestrateur construit un prompt incorporant le contexte récupéré et appelle le module Reader en retraçant soigneusement les sources documentaires.
6. Le module Reader mobilise un LLM pour générer une suggestion de réponse, qui est renvoyée à l'agent via l'interface utilisateur.
7. L'agent évalue dans l'interface la qualité de la réponse puis la corrige et la valide. Cette étape permet la collecte de feedback de l'intelligence humaine.
8. Transfert au système de messagerie pour envoi.
9. La réponse est acheminée au client et mentionne les références à certaines sources.
10. Le client peut se référer au référentiel public des ressources documentaires utilisées. 

Pour implémenter ce processus d'ensemble on distingue 4 grands sous-systèmes :
- en vert : l'interface d'utilisation, d'ingestion de la base documentaire et de constitution des jeux de données qualitatifs pour le fine-tuning et le RLHF.
- en noir : le système de messagerie et son interfaçage.
- en violet : le système Retrieval Augmented Generation proprement dit. 
- en rouge : l'ensemble du pipeline et de la base de données de fine-tuning et RLHF.


## Facteurs clés de succès

L'état de l'art du domaine évolue à très grande vitesse ; il est donc critique de pouvoir changer de modèles en cours de projet sans remettre en cause significativement la solution développée. Polyconseil a donc conçu une architecture modulaire, dans laquelle le LLM, le modèle d'embedding et la méthode de retrieval peuvent être modifiés par une simple configuration. Ceci permet en outre aux data scientists d'itérer facilement sur différentes configurations pour optimiser la performance de la solution. Cela permet enfin plus globalement de disposer en production et assez simplement de la meilleure solution de LLM à date, ouverte et assurant le caractère souverain.

Dans une optique d’optimisation de la charge de conception, nous avons opté pour un [monolithe modulaire](https://www.milanjovanovic.tech/blog/what-is-a-modular-monolith) en [architecture hexagonale](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/). Mais comme l'évaluation efficace d'un LLM demande une exécution sur un GPU nous avons déporté à l'extérieur du monolithe l'appel au LLM. Pour ce faire, nous avons utilisé [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) d’Hugging Face, qui offre un service dockerisé performant et configurable pour héberger n'importe quel LLM disponible sur le Hub.

Afin de garantir l’indépendance et la souveraineté des données, la solution s'appuie essentiellement sur des modèles open source, déployés sur un fournisseur de Cloud français : [NumSpot](https://numspot.com/). Cet acteur a été choisi pour sa qualification SecNumCloud, adossé à l'IaaS Outscale, fondée par Dassault Systèmes pour répondre à ses propres enjeux de sécurité.

Concernant les solutions open source, de nombreux outils français se démarquent. La licorne [Mistral AI](https://mistral.ai/fr/), dont nous utilisons actuellement le modèle [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), sort notamment du lot. Mais d’autres projets plus discrets mais plus spécifiques présentent un fort potentiel pour répondre à nos enjeux, tels que [CroissantLLM](https://huggingface.co/blog/manu/croissant-llm-blog) que nous évaluons. Ce modèle est issu d’une collaboration entre le [laboratoire MICS](https://www.mics.centralesupelec.fr/) de CentraleSupélec et [Illuin Technology](https://www.illuin.tech/). Il vise à offrir un modèle spécialisé sur des données en français, qui soit éthique, responsable et performant.

Sur le plan organisationnel, nous avons constitué une seule équipe Agile opérant selon une méthodologie de type ScrumBan souple, complétée par un rituel hebdomadaire de veille et de formation sur les avancées de l'IA. Ce dernier est conduit par l’expert Hugging Face du programme [Expert Support](https://huggingface.co/support). Cette organisation facilite un transfert fluide des compétences et des responsabilités vers les équipes Data de la BdT, tout en assurant des livraisons régulières et résilientes aux changements de contexte du projet. Ainsi nous avons livré tôt un MVP naïf de la solution et des notebooks d'évaluations qualitatives et quantitatives. Pour cela, nous utilisons des bibliothèques open source spécialisées dans l’évaluation des systèmes d’IA générative, telles que RAGAS. Ce travail constitue désormais le socle sur lequel nous itérons de nouvelles fonctionnalités et des améliorations de la performance du système.

Le mot de la fin, par Hakim Lahlou, Directeur Innovation et Stratégie Groupes OLS à la Direction des prêts de la Banque des Territoires :
>  _'Nous sommes heureux de travailler à la Banque des Territoires aux côtés de ces experts reconnus en France comme à l’international autour d’une solution data très innovante et pleinement souveraine. Sur la base de ce Programme pilote, cette approche ouvre une nouvelle voie : c’est probablement ainsi à l’avenir que se déploieront les politiques publiques dans les territoires ainsi que les financements nécessaires à la Transformation écologique et énergétique du pays. Cette approche est aujourd’hui la seule à permettre des déploiements massifs, efficaces et précis.'_

_Vous êtes concernés par un projet avec des enjeux de souveraineté ? Vous souhaitez mettre au point une solution qui tire profit des capacités des LLMs ? Ou vous avez tout simplement des questions sur nos services ou sur le projet ? Contactez-nous directement à alivia@polyconseil.fr_

_Si vous êtes intéressé par le programme Hugging Face Expert Support pour votre entreprise, veuillez nous contacter [ici](https://huggingface.co/contact/sales?from=support) - notre équipe commerciale vous contactera pour discuter de vos besoins !_
