---
title: "Putting ethical principles at the core of the research lifecycle"
thumbnail: /blog/assets/71_ethical-charter/thumbnail.jpg
authors:
- user: SaulLu
- user: skaramcheti
- user: HugoLaurencon
- user: Leyo
- user: TimeRobber
- user: VictorSanh
- user: aps
- user: giadilli
- user: sasha
- user: yjernite
- user: meg
- user: douwekiela
---

<h1>Putting ethical principles at the core of the research lifecycle</h1>
<h2>Ethical charter - Multimodal project</h2>

{blog_metadata}
{authors}

## Purpose of the ethical charter

It has been well documented that machine learning research and applications can potentially lead to "data privacy issues, algorithmic biases, automation risks and malicious uses" (NeurIPS 2021 [ethics guidelines](https://nips.cc/public/EthicsGuidelines)). The purpose of this short document is to formalize the ethical principles that we (the multimodal learning group at Hugging Face) adopt for the project we are pursuing. By defining these ethical principles at the beginning of the project, we make them core to our machine learning lifecycle.

By being transparent about the decisions we're making in the project, who is working on which aspects of the system, and how the team can be contacted, we hope to receive feedback early enough in the process to make meaningful changes, and ground discussions about choices in an awareness of the goals we aim to achieve and the values we hope to incorporate.

This document is the result of discussions led by the multimodal learning group at Hugging Face (composed of machine learning researchers and engineers), with the contributions of multiple experts in ethics operationalization, data governance, and personal privacy.

## Limitations of this ethical charter

This document is a work in progress and reflects a state of reflection as of May 2022. There is no consensus nor official definition of "ethical AI" and our considerations are very likely to change over time. In case of updates, we will reflect changes directly in this document while providing the rationale for changes and tracking the history of updates [through GitHub](https://github.com/huggingface/blog/commits/main/ethical-charter-multimodal.md). This document is not intended to be a source of truth about best practices for ethical AI. We believe that even though it is imperfect, thinking about the impact of our research, the potential harms we foresee, and strategies we can take to mitigate these harms is going in the right direction for the machine learning community. Throughout the project, we will document how we operationalize the values described in this document, along with the advantages and limitations we observe in the context of the project.

## Content policy

Studying the current state-of-the-art multimodal systems, we foresee several misuses of the technologies we aim at as part of this project. We provide guidelines on some of the use cases we ultimately want to prevent:

- Promotion of content and activities which are detrimental in nature, such as violence, harassment, bullying, harm, hate, and all forms of discrimination. Prejudice targeted at specific identity subpopulations based on gender, race, age, ability status, LGBTQA+ orientation, religion, education, socioeconomic status, and other sensitive categories (such as sexism/misogyny, casteism, racism, ableism, transphobia, homophobia).
- Violation of regulations, privacy, copyrights, human rights, cultural rights, fundamental rights, laws, and any other form of binding documents.
- Generating personally identifiable information.
- Generating false information without any accountability and/or with the purpose of harming and triggering others.
- Incautious usage of the model in high-risk domains - such as medical, legal, finance, and immigration - that can fundamentally damage people’s lives.

## Values for the project

- **Be transparent:** We are transparent and open about the intent, sources of data, tools, and decisions. By being transparent, we expose the weak points of our work to the community and thus are responsible and can be held accountable.
- **Share open and reproducible work:** Openness touches on two aspects: the processes and the results. We believe it is good research practice to share precise descriptions of the data, tools, and experimental conditions. Research artifacts, including tools and model checkpoints, must be accessible - for use within the intended scope - to all without discrimination (e.g., religion, ethnicity, sexual orientation, gender, political orientation, age, ability). We define accessibility as ensuring that our research can be easily explained to an audience beyond the machine learning research community.
- **Be fair:** We define fairness as the equal treatment of all human beings. Being fair implies monitoring and mitigating unwanted biases that are based on characteristics such as race, gender, disabilities, and sexual orientation. To limit as much as possible negative outcomes, especially outcomes that impact marginalized and vulnerable groups, reviews of unfair biases - such as racism for predictive policing algorithms - should be conducted on both the data and the model outputs.
- **Be self-critical:** We are aware of our imperfections and we should constantly lookout for ways to better operationalize ethical values and other responsible AI decisions. For instance, this includes better strategies for curating and filtering training data. We should not overclaim or entertain spurious discourses and hype.
- **Give credit:** We should respect and acknowledge people's work through proper licensing and credit attribution.

We note that some of these values can sometimes be in conflict (for instance being fair and sharing open and reproducible work, or respecting individuals’ privacy and sharing datasets), and emphasize the need to consider risks and benefits of our decisions on a case by case basis.
