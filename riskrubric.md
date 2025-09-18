---
title: "Democratizing AI Safety with [RiskRubric.ai](https://RiskRubric.ai)"  
thumbnail: /blog/assets/riskrubric/thumbnail.png
authors:
- user: galmo-noma
  guest: true
---

# **Democratizing AI Safety with RiskRubric.ai**

*Building trust in the open model ecosystem through standardized risk assessment*

More than 500,000 models can be found on the Hugging Face hub, but it’s not always clear to users how to choose the best model for them, notably on the security aspects. Developers might find a model that perfectly fits their use case, but have no systematic way to evaluate its security posture, privacy implications, or potential failure modes. 

As models become more powerful and adoption accelerates, we need equally rapid progress in AI safety and security reporting. We're therefore excited to announce [RiskRubric.ai](https://riskrubric.ai/), a novel initiative led by Cloud Security Alliance and [Noma Security](https://noma.security), with contributions by Haize Labs and Harmonic Security, for standardized and transparent risk assessment in the AI model ecosystem.

## **Risk Rubric, a new Standardized Assessment of Risk for models**

RiskRubric.ai provides **consistent, comparable risk scores across the entire model landscape**, by evaluating models across six pillars: transparency, reliability, security, privacy, safety, and reputation. 

The platform's approach aligns perfectly with open-source values: rigorous, transparent, and reproducible. Using Noma Security capabilities to automate the effort, each model undergoes:

* **1,000+ reliability tests** checking consistency and edge case handling  
* **200+ adversarial security probes** for jailbreaks and prompt injections  
* **Automated code scanning** of model components  
* **Comprehensive documentation review** of training data and methods  
* **Privacy assessment** including data retention and leakage testing  
* **Safety evaluation** through structured harmful content tests

These assessments produce 0-100 scores for each risk pillar, rolling up to clear A-F letter grades. Each evaluation also includes specific vulnerabilities found, recommended mitigations, and suggestions for improvements.

RiskRubric also comes with filters to help developers and organizations make deployment decisions based on what’s important for them. Need a model with strong privacy guarantees for healthcare applications? Filter by privacy scores. Building a customer-facing application requiring consistent outputs? Prioritize reliability ratings. 

## **What we found (as of September 2025\)**

Evaluating both open and closed models with the exact same standards highlighted some interesting results: many open models actually outperform their closed counterparts in specific risk dimensions (particularly transparency, where open development practices shine).

Let’s look at general trends: 

**Risk distribution is polarized – most models are strong, but mid-tier scores show elevated exposure**

![total_score](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/riskrubric/RiskRubric.png)

The total risk scores range from 47 to 94, with a median of 81 (on a 100 points). Most models cluster in the “safer” range (54% are A or B level), but a long tail of underperformers drags the average down. That split shows a polarization: models tend to be either well-protected or in the middle-score range, with fewer in between.

The models concentrated in the 50–67 band (C/D range) are not outright broken, but they do provide only medium to low overall protection. This band represents the most practical area of concern, where security gaps are material enough to warrant prioritization.

**What this means:** Don’t assume the “average” model is safe. The tail of weak performers is real – and that’s where attackers will focus. Teams can use composite scores to set a **minimum threshold (e.g. 75\)** for procurement or deployment, ensuring outliers don’t slip into production.

**Safety risk is the “swing factor” – but it tracks closely with security posture**

![safety_histogram](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/riskrubric/Safety.png)

The *Safety & Societal* pillar (e.g. harmful output prevention) shows the widest variation across models. Importantly, models that invest in **security hardening** (prompt injection defenses, policy enforcement) almost always score better on safety as well.

**What this means**: Strengthening core security controls goes beyond preventing jailbreaks, but also directly reduces downstream harms\! Safety seems like it is a byproduct of robust security posture.

**Guardrails can erode transparency – unless you design for it**

Stricter protections often make models *less transparent* to end users (e.g. refusals without explanations, hidden boundaries). This can create a trust gap: users may perceive the system as “opaque” even while it’s secure.

**What this means**: Security shouldn’t come at the cost of trust. To balance both, pair strong safeguards with **explanatory refusals, provenance signals, and auditability**. This preserves transparency without loosening defenses.

An updating results sheet can be accessed [here](https://docs.google.com/spreadsheets/d/15adko_TbbR9lVK6OweBTSi1OZg9QHhEvZGMI-CbXCaM/edit?usp=sharing)

## **Conclusion**

When risk assessments are public and standardized, the entire community can work together to improve model safety. Developers can see exactly where their models need strengthening, and the community can contribute fixes, patches, and safer fine-tuned variants. This creates a virtuous cycle of transparent improvement that's impossible with closed systems. It also helps the community at large understand what works and does not, safety wise, by studying best models. 

If you want to take part in this initiative, you can submit your model for evaluation (or suggest existing models\!) to understand their risk profile\! 

We also welcome all feedback on the assessment methodology and scoring framework
