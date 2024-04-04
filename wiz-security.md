---
title: "Hugging Face partners with Wiz Research to Improve AI Security"
authors:
- user: JJoe206
- user: GuillaumeSalouHF
- user: michellehbn
- user: XciD
- user: mcpotato
- user: Narsil
- user: julien-c
---

# Hugging Face partners with Wiz Research to Improve AI Security

We are pleased to announce that we are partnering with Wiz with the goal of improving security across our platform and the AI/ML ecosystem at large.

Wiz researchers [collaborated with Hugging Face on the security of our platform and shared their findings](https://www.wiz.io/blog/wiz-and-hugging-face-address-risks-to-ai-infrastructure). Wiz is a cloud security company that helps their customers build and maintain software in a secure manner. Along with the publication of this research, we are taking the opportunity to highlight some related Hugging Face security improvements.

Hugging Face has recently integrated Wiz for Vulnerability Management, a continuous and proactive process to keep our platform free of security vulnerabilities. In addition, we are using Wiz for Cloud Security Posture Management (CSPM), which allows us to configure our cloud environment securely, and monitor to ensure it remains secure.  

One of our favorite Wiz features is a holistic view of Vulnerabilities, from storage to compute to network.  We run multiple Kubernetes (k8s) clusters and have resources across multiple regions and cloud providers, so it is extremely helpful to have a central report in a single location with the full context graph for each vulnerability. We’ve also built on top of their tooling, to automatically remediate detected issues in our products, most notably in Spaces.

As part of the joint work, Wiz’s security research team identified shortcomings of our sandboxed compute environments by running arbitrary code within the system thanks to pickle.  As you read this blog and the Wiz security research paper, it is important to remember that we have resolved all issues related to the exploit and continue to remain diligent in our Threat Detection and Incident Response process.  

## Hugging Face Security

At Hugging Face we take security seriously, as AI rapidly evolves, new threat vectors seemingly pop up every day. Even as Hugging Face announces multiple partnerships and business relationships with the largest names in tech, we remain committed to allow our users and the AI community to responsibly experiment with and operationalize AI/ML systems and technologies.  We are dedicated to securing our platform as well as democratizing AI/ML, such that the community can contribute to and be a part of this paradigm shifting event that will impact us all.  We are writing this blog to reaffirm our commitment to protecting our users and customers from security threats.  Below we will also discuss Hugging Face’s philosophy regarding our support of the controversial pickle files as well as discuss the shared responsibility of moving away from the pickle format. 

There are many other exciting security improvements and announcements coming in the near future.  The publications will not only discuss the security risks to the Hugging Face platform community, but also cover systemic security risks of AI as well as best practices for mitigation.  We remain committed to making our products, our infrastructure, and the AI community secure, stay tuned for followup security blog posts and whitepapers.

## Open Source Security Collaboration and Tools for the Community

We highly value transparency and collaboration with the community and this includes participation in the identification and disclosure of vulnerabilities, collaborating on resolving security issues, and security tooling. Below are examples of our security wins born from collaboration, which help the entire AI community lower their security risk:
- Picklescan was built in partnership with Microsoft; Matthieu Maitre started the project and given we had our own internal version of the same tool, we joined forces and contributed to picklescan. Refer to the following documentation page if you are curious to know more on how it works:
https://huggingface.co/docs/hub/en/security-pickle
- Safetensors, which was developed by Nicolas Patry, is a secure alternative to pickle files. Safetensors has been audited by Trail of Bits on a collaborative initiative with EuletherAI & Stability AI.
https://huggingface.co/docs/safetensors/en/index
- We have a robust bug bounty program, with many amazing researchers from all around the world. Researchers who have identified a security vuln may inquire about joining our program through security@huggingface.co
- Malware Scanning: https://huggingface.co/docs/hub/en/security-malware
- Secrets Scanning:  https://huggingface.co/docs/hub/security-secrets
- As previously mentioned, we’re also collaborating with Wiz to lower Platform security risks 
- We are starting a series of security publications which address security issues facing the AI/ML community.

## Security Best Practices for Open Source AI/ML users

AI/ML has introduced new vectors of attack, but for many of these attacks mitigants are long standing and well known. Security professionals should ensure that they apply relevant security controls to AI resources and models. In addition, below are some resources and best practices when working with open source software and models:

- Know the contributor:  Only use models from trusted sources and pay attention to commit signing.  https://huggingface.co/docs/hub/en/security-gpg
- Don’t use pickle files in production environments
- Use Safetensors: https://huggingface.co/docs/safetensors/en/index 
- Review the OWASP top 10:  https://owasp.org/www-project-top-ten/
- Enable MFA on your Hugging Face accounts
- Establish a Secure Development Lifecycle, which includes code review by a security professional or engineer with appropriate security training
Test models in non-production and virtualized test/dev environments

## Pickle Files - The Insecure Elephant in the Room

Pickle files have been at the core of most of the research done by Wiz and other recent publications by security researchers about Hugging Face. Pickle files have long been considered to have security risks associated with them, see our doc files for more information: https://huggingface.co/docs/hub/en/security-pickle

Despite these known security flaws, the AI/ML community still frequently uses pickles (or similarly trivially exploitable formats). Many of these use cases are low risk or for test purposes making the familiarity and ease of use of pickle files more attractive than the secure alternative.  
As the open source AI platform, we are left with the following options:
- Ban pickle files entirely
- Do nothing about pickle files
- Finding a middle ground that both allows for pickle use as well as reasonably and practicably mitigating the risks associated with pickle files

We have chosen option 3, the middle ground for now. This option is a burden on our engineering and security teams and we have put in significant effort to mitigate the risks while allowing the AI community to use tools they choose. Some of the key mitigants we have implemented to the risks related to pickle include: 

- Creating clear documentation outlining the risks
- Developing automated scanning tools
- Using scanning tools and labeling models with security vulnerabilities with clear warnings
- We have even provided a secure solution to use in lieu of pickle (Safetensors)
- We have also made Safetensors a first class citizen on our platform to protect the community members who may not understand the risks
- In addition to the above, we have also had to significantly segment and enhance security of the areas in which models are used to account for potential vulnerabilities within them

We intend to continue to be the leader in protecting and securing the AI Community. Part of this will be monitoring and addressing risks related to pickle files. Sunsetting support of pickle is also not out of the question either, however, we do our best to balance the impact on the community as part of a decision like this. 

An important note that the upstream open source communities as well as large tech and security firms, have been largely silent on contributing to solutions here and left Hugging Face to both define philosophy and invest heavily in developing and implementing mitigating controls to ensure the solution is both acceptable and practicable. 

## Closing remarks

I spoke extensively to Nicolas Patry, the creator of Safetensors in writing this blog post and he requested that I add a call to action to the AI open source community and AI enthusiasts:

- Pro-actively start replacing your pickle files with Safetensors. As mentioned earlier, pickle contains inherent security flaws and may be unsupported in the near future.
- Keep opening issues/PRs upstream about security to your favorite libraries to push secure defaults as much as possible upstream. 

The AI industry is rapidly changing and new attack vectors / exploits are being identified all the time. Huggingface has a one of a kind community and we partner heavily with you to help us maintain a secure platform.  

Please remember to responsibly disclose security vulns/bugs through the appropriate channels to avoid potential legal liability and violation of laws.

Want to join the discussion?  Reach out to us as security@huggingface.co or follow us on Linkedin/Twitter.
