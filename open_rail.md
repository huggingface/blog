---
title: "OpenRAIL: Towards open and responsible AI licensing frameworks"
thumbnail: /blog/assets/100_open_rail/100_open-rail.png
authors:
- user: CarlosMF
---


<h1>OpenRAIL: Towards open and responsible AI licensing frameworks</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->
  

Open & Responsible AI licenses ("OpenRAIL") are AI-specific licenses enabling open access, use and distribution of AI artifacts while requiring a responsible use of the latter. OpenRAIL licenses could be for open and responsible ML what current open software licenses are to code and Creative Commons to general content: **a widespread community licensing tool.**

Advances in machine learning and other AI-related areas have flourished these past years partly thanks to the ubiquity of the open source culture in the Information and Communication Technologies (ICT) sector, which has permeated into ML research and development dynamics. Notwithstanding the benefits of openness as a core value for innovation in the field, (not so already) recent events related to the ethical and socio-economic concerns of development and use of machine learning models have spread a clear message: Openness is not enough. Closed systems are not the answer though, as the problem persists under the opacity of firms' private AI development processes.

## **Open source licenses do not fit all**

Access, development and use of ML models is highly influenced by open source licensing schemes. For instance, ML developers might colloquially refer to "open sourcing a model" when they make its weights available by attaching an official open source license, or any other open software or content license such as Creative Commons. This begs the question: why do they do it? Are ML artifacts and source code really that similar? Do they share enough from a technical perspective that private governance mechanisms (e.g. open source licenses) designed for source code should also govern the development and use of ML models?

Most current model developers seem to think so, as the majority of openly released models have an open source license (e.g., Apache 2.0). See for instance the Hugging Face [Model Hub](https://huggingface.co/models?license=license:apache-2.0&sort=downloads) and [Muñoz Ferrandis & Duque Lizarralde (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4018413).

However, empirical evidence is also telling us that a rigid approach to open sourcing [and/or](https://www.gnu.org/philosophy/open-source-misses-the-point.en.html) Free Software dynamics and an axiomatic belief in Freedom 0 for the release of ML artifacts is creating socio-ethical distortions in the use of ML models (see [Widder et al. (2022)](https://davidwidder.me/files/widder-ossdeepfakes-facct22.pdf)). In simpler terms, open source licenses do not take the technical nature and capabilities of the model as a different artifact to software/source code into account, and are therefore ill-adapted to enabling a more responsible use of ML models (e.g. criteria 6 of the [Open Source Definition](https://opensource.org/osd)), see also [Widder et al. (2022)](https://davidwidder.me/files/widder-ossdeepfakes-facct22.pdf); [Moran (2021)](https://www.google.com/url?q=https://thegradient.pub/machine-learning-ethics-and-open-source-licensing-2/&sa=D&source=docs&ust=1655402923069398&usg=AOvVaw3yTXEfpRQOJ99w04v5GAEd); [Contractor et al. (2020)](https://facctconference.org/static/pdfs_2022/facct22-63.pdf).

If specific ad hoc practices devoted to documentation, transparency and ethical usage of ML models are already present and improving each day (e.g., model cards, evaluation benchmarks), why shouldn't open licensing practices also be adapted to the specific capabilities and challenges stemming from ML models?

Same concerns are rising in commercial and government ML licensing practices. In the words of [Bowe & Martin (2022)](https://www.gmu.edu/news/2022-04/no-10-implementing-responsible-ai-proposed-framework-data-licensing): "_Babak Siavoshy, general counsel at Anduril Industries, asked what type of license terms should apply to an AI algorithm privately developed for computer-vision object detection and adapt it for military targeting or threat-evaluation? Neither commercial software licenses nor standard DFARS data rights clauses adequately answer this question as neither appropriately protects the developer's interest or enable the government to gain the insight into the system to deploy it responsibly_".

If indeed ML models and software/source code are different artifacts, why is the former released under open source licenses? The answer is easy, open source licenses have become the de facto standard in software-related markets for the open sharing of code among software communities. This "open source" approach to collaborative software development has permeated and influenced AI development and licensing practices and has brought huge benefits. Both open source and Open & Responsible AI licenses ("OpenRAIL") might well be complementary initiatives.

**Why don't we design a set of licensing mechanisms inspired by movements such as open source and led by an evidence-based approach from the ML field?** In fact, there is a new set of licensing frameworks which are going to be the vehicle towards open and responsible ML development, use and access: Open & Responsible AI Licenses ([OpenRAIL](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses)).

## **A change of licensing paradigm: OpenRAIL**

The OpenRAIL [approach](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses) taken by the [RAIL Initiative](https://www.licenses.ai/) and supported by Hugging Face is informed and inspired by initiatives such as BigScience, Open Source, and Creative Commons. The 2 main features of an OpenRAIL license are:

- **Open:** these licenses allow royalty free access and flexible downstream use and re-distribution of the licensed material, and distribution of any derivatives of it.

- **Responsible:** OpenRAIL licenses embed a specific set of restrictions for the use of the licensed AI artifact in identified critical scenarios. Use-based restrictions are informed by an evidence-based approach to ML development and use limitations which forces to draw a line between promoting wide access and use of ML against potential social costs stemming from harmful uses of the openly licensed AI artifact. Therefore, while benefiting from an open access to the ML model, the user will not be able to use the model for the specified restricted scenarios.

The integration of use-based restrictions clauses into open AI licenses brings up the ability to better control the use of AI artifacts and the capacity of enforcement to the licensor of the ML model, standing up for a responsible use of the released AI artifact, in case a misuse of the model is identified. If behavioral-use restrictions were not present in open AI licenses, how would licensors even begin to think about responsible use-related legal tools when openly releasing their AI artifacts? OpenRAILs and RAILs are the first step towards enabling ethics-informed behavioral restrictions.

And even before thinking about enforcement, use-based restriction clauses might act as a deterrent for potential users to misuse the model (i.e., dissuasive effect). However, the mere presence of use-based restrictions might not be enough to ensure that potential misuses of the released AI artifact won't happen. This is why OpenRAILs require downstream adoption of the use-based restrictions by subsequent re-distribution and derivatives of the AI artifact, as a means to dissuade users of derivatives of the AI artifact from misusing the latter. 

The effect of copyleft-style behavioral-use clauses spreads the requirement from the original licensor on his/her wish and trust on the responsible use of the licensed artifact. Moreover, widespread adoption of behavioral-use clauses gives subsequent distributors of derivative versions of the licensed artifact the ability for a better control of the use of it. From a social perspective, OpenRAILs are a vehicle towards the consolidation of an informed and respectful culture of sharing AI artifacts acknowledging their limitations and the values held by the licensors of the model.

## **OpenRAIL could be for good machine learning what open software licensing is to code**

Three examples of OpenRAIL licenses are the recently released [BigScience OpenRAIL-M](https://www.licenses.ai/blog/2022/8/26/bigscience-open-rail-m-license), StableDiffusion's [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license), and the genesis of the former two: [BigSicence BLOOM RAIL v1.0](https://huggingface.co/spaces/bigscience/license) (see post and FAQ [here](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)). The latter was specifically designed to promote open and responsible access and use of BigScience's 176B parameter model named BLOOM (and related checkpoints). The license plays at the intersection between openness and responsible AI by proposing a permissive set of licensing terms coped with a use-based restrictions clause wherein a limited number of restricted uses is set based on the evidence on the potential that Large Language Models (LLMs) have, as well as their inherent risks and scrutinized limitations. The OpenRAIL approach taken by the RAIL Initiative is a consequence of the BigScience BLOOM RAIL v1.0 being the first of its kind in parallel with the release of other more restricted models with behavioral-use clauses, such as [OPT-175](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md) or [SEER](https://github.com/facebookresearch/vissl/blob/main/projects/SEER/MODEL_LICENSE.md), being also made available.

The licenses are BigScience's reaction to 2 partially addressed challenges in the licensing space: (i) the "Model" being a different thing to "code"; (ii) the responsible use of the Model. BigScience made that extra step by really focusing the license on the specific case scenario and BigScience's community goals. In fact, the solution proposed is kind of a new one in the AI space: BigScience designed the license in a way that makes the responsible use of the Model widespread (i.e. promotion of responsible use), because any re-distribution or derivatives of the Model will have to comply with the specific use-based restrictions while being able to propose other licensing terms when it comes to the rest of the license.

OpenRAIL also aligns with the ongoing regulatory trend proposing sectoral specific regulations for the deployment, use and commercialization of AI systems. With the advent of AI regulations (e.g., [EU AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206); Canada's [proposal](https://iapp.org/news/a/canada-introduces-new-federal-privacy-and-ai-legislation/) of an AI & Data Act), new open licensing paradigms informed by AI regulatory trends and ethical concerns have the potential of being massively adopted in the coming years. Open sourcing a model without taking due account of its impact, use, and documentation could be a source of concern in light of new AI regulatory trends. Henceforth, OpenRAILs should be conceived as instruments articulating with ongoing AI regulatory trends and part of a broader system of AI governance tools, and not as the only solution enabling open and responsible use of AI.

Open licensing is one of the cornerstones of AI innovation. Licenses as social and legal institutions should be well taken care of. They should not be conceived as burdensome legal technical mechanisms, but rather as a communication instrument among AI communities bringing stakeholders together by sharing common messages on how the licensed artifact can be used.

Let's invest in a healthy open and responsible AI licensing culture, the future of AI innovation and impact depends on it, on all of us, on you.

Author: Carlos Muñoz Ferrandis

Blog acknowledgments: Yacine Jernite, Giada Pistilli, Irene Solaiman, Clementine Fourrier, Clément Délange
