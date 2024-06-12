---
title: Space secrets security update
thumbnail: /blog/assets/space-secrets-security-update/space-secrets-security-update.png
authors:
- user: huggingface
---

# Space secrets leak disclosure

Earlier this week our team detected unauthorized access to our Spaces platform, specifically related to Spaces secrets. As a consequence, we have suspicions that a subset of Spaces’ secrets could have been accessed without authorization.

As a first step of remediation, we have revoked a number of HF tokens present in those secrets. Users whose tokens have been revoked already received an email notice. **We recommend you refresh any key or token and consider switching your HF tokens to fine-grained access tokens which are the new default.**

We are working with outside cyber security forensic specialists, to investigate the issue as well as review our security policies and procedures.

Over the past few days, we have made other significant improvements to the security of the Spaces infrastructure, including completely removing org tokens (resulting in increased traceability and audit capabilities), implementing key management service (KMS) for Spaces secrets, robustifying and expanding our system’s ability to identify leaked tokens and proactively invalidate them, and more generally improving our security across the board. We also plan on completely deprecating “classic” read and write tokens in the near future, as soon as fine-grained access tokens reach feature parity. We will continue to investigate any possible related incident.

Finally, we have also reported this incident to law enforcement agencies and Data protection authorities.

We deeply regret the disruption this incident may have caused and understand the inconvenience it may have posed to you. We pledge to use this as an opportunity to strengthen the security of our entire infrastructure. For any question, please contact us at security@huggingface.co.

