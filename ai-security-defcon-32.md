---
title: AI + Security Highlights from DEF CON 32
thumbnail: /blog/assets/ai-security-defcon-32/thumbnail.png
authors:
- user: jack-kumar
---

# AI + Security Highlights from DEF CON 32
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai-security-defcon-32/dc32.png)

The world's top hackers and security researchers gathered in Las Vegas early this month for DEF CON 32. This year's conference saw a notable increase in attention towards AI Security, reflecting the field's growing importance in the wake of rapid advancements in AI technology. Building on the momentum from DEF CON 31, which marked the first major security conference in the post-ChatGPT era, this year's event showcased a more mature understanding of the AI Security landscape. Eighteen months after ChatGPT's mainstream debut, the industry has had time to assess the challenges and opportunities at the intersection of AI and Security. In this blog post, we'll dive into some talks and events from DEF CON 32, highlighting key insights and resources from the forefront of AI Security research.


## AIxCC

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai-security-defcon-32/aixcc-1.png)

The headliner of this year was the AI Cyber Challenge. The event follows on the heels of the [2016 Cyber Grand Challenge](https://www.darpa.mil/program/cyber-grand-challenge) in which autonomous systems running on supercomputers competed to create defensive systems in real time. This year saw the return of the DARPA challenge in a new format. The AIxCC is a two-year competition in which entrants design novel AI systems using contemporary techniques to automatically detect issues and secure critical code, competing for a prize pool of $29.5 million. Of the 42 semifinalists, 7 finalists emerged and will move on the the final round at DEF CON 33 in 2025. The competition is fully open source - to claim their prizes and move on to the finals, each team must agree to open source their work with an OSI approved license under the stewardship of Linux Foundation's Open Source Security Foundation (OpenSSF). We're keeping an eye on this space as there's a lot more to come. 

Learn More:
- https://aicyberchallenge.com/
- [DARPA: AI Cyber Challenge Proves Promise of AI-Driven Cybersecurity](https://www.darpa.mil/news-events/2024-08-11)
- [Finalist Blog: Team Atlanta](https://team-atlanta.github.io/blog/post-atl/)
- [Finalist Blog: Trail of Bits](https://blog.trailofbits.com/2024/08/12/trail-of-bits-advances-to-aixcc-finals/)


## GRT2

![](https://placehold.co/1280x720.png)

Following up on last year's White House sponsored Generative AI Red Team Challenge, the AI Village returned with GRT 2. Building on their prior experience, this year's challenge featured real model evaluations in a bug bash format. Participants were asked to prepare reports about flaws in LLMs using the Inspect AI framework and submit them to the platform. The reports were reviewed by independent experts and were awarded cash bounties. All data generated will be made public by the AI Village in the near future.

Learn More:
- [AI Village: Announcing Generative Red Team 2](https://aivillage.org/generative%20red%20team/generative-red-team-2/)
- [GitHub: Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
- [Dreadnode: Crucible](https://crucible.dreadnode.io)


## AI Village Evolves

![](https://placehold.co/1280x720.png)

Founded in 2018, the volunteer led AI Village has been leading the AI + Security conversation at DEF CON for the past 7 years. Founder Sven announced the formation of ____, which will become the parent oragnization of AI Village. This new structure will allow the AI Village to move from volunteers to paid employees who will handle operations for the AI Village. This change is expected to lead to bigger, better and even more exciting talks, demos and events in the coming years.


## Talks

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ai-security-defcon-32/talks.png)

DEF CON 32 featured several AI + Security themed talks, many of which were unfortunately not recorded. Below we'll cover some talks that had slides, videos or papers for the community to explore. **Where Are The Talk Recordings?** DEF CON typically uploads recordings of talks to [YouTube](https://www.youtube.com/@DEFCONConference) and the [DEF CON Media Server](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/) 6-8 months after the event. However, you can purchase early access to the talks at [defcononline.com](http://defcononline.com).


### Taming the Beast: Inside the Llama 3 Red Teaming Process
The Llama 3 Red Team took the stage to share their comprehensive approach to red teaming Llama 3. Their presentation covered the automation of large-scale testing, with a focus on simulating multi-turn conversations between the model and "attacker LLMs" posing as malicious users. The team explored adversarial prompting attacks, discussing both established and novel techniques. The team also discussed their efforts to assess the cybersecurity risks and capabilities of large language models, which culminated in the publication of their recent paper. This research examined various cybersecurity scenarios, including multiturn phishing, automated cyber attacks, and the use of LLMs as assistants for technical humans with limited cybersecurity expertise. The findings were surprising: LLMs fell short in almost every category.

Learn More:
- [Meta: CyberSecEval 3: Advancing the Evaluations of Cybersecurity Risks and Capabilites](https://ai.meta.com/research/publications/cyberseceval-3-advancing-the-evaluation-of-cybersecurity-risks-and-capabilities-in-large-language-models/)


### Leveraging AI for Smarter Bug Bounties
Security Researchers Joel and Deigo from XBOW presented their research and development into leveraging AI in bug bounty hunting. The team demonstrated their tool, which achieved success rates of 75% in PortSwigger Labs, 72% in PortSwigger Exercises, and 85% in their custom-built benchmarks - all without human intervention. In multiple demoes, they showed an AI agent autonomously exploiting several scenarios, where it established goals, performed reconnaissance, handled error cases, searched the internet for proof-of-concepts, and ultimately fully exploited the service. Their research also compared the performance of human pentesters to their AI-powered platform. In a controlled experiment, they hired five pentesters from different companies with varying skill levels. Their platform outperformed all human pentesters except those at the Principal level on the most challenging task category, completing all tasks in just 30 minutes compared to 40 hours for humans. 

Learn More:
- [DEF CON Media Server: Presentation Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20villages/DEF%20CON%2032%20-%20Bug%20Bounty%20Village%20-%20Diego%20Jurado%20-%20Leveraging%20AI%20for%20Smarter%20Bug%20Bounties.pdf)
- [XBOW: XBOW vs Humans](https://xbow.com/blog/xbow-vs-humans/)
- [XBOW: Traces](https://xbow.com/#traces)
- [PortSwigger: Web Security Labs](https://portswigger.net/web-security/all-labs)


### Bolabuster: Harnessing LLMs for Automating BOLA Dectection
Ravid and Jay from Palo Alto Networks spoke about BOLABuster, their technique using LLMs to find BOLAs. Broken Object Level Authorization (BOLA) are classified as the highest risk on OWASP Top 10 API Security Risks. Broken object level authorization is a security vulnerability where an API fails to verify if the user is authorized to access the assets they are attempting to access. The team highlighted how they use LLMs for detecting vulnerable endpoints, building endpoint dependency graphs and generating test scripts. They compare their tool to RESTler, showing their method detected all BOLAs across three environments where as RESTler detected zero, doing so with less than 1% of the API calls.

Learn More:
- [Palo Alto Networks: Harnessing LLMs for Automating BOLA Detection](https://unit42.paloaltonetworks.com/automated-bola-detection-and-ai/)
- [OWASP Top 10 API Security Risks: Broken Object Level Authorization](https://owasp.org/Top10/A01_2021-Broken_Access_Control/)
- [GitHub: RESTler](https://github.com/microsoft/restler-fuzzer)


### On Your Ocean's 11 Team, I'm the AI Guy (technically Girl)
AI Hacker Harriet Farlow demonstrated a futuristic casino heist in which the "AI Guy" builds a facial recognition bypass. Harriet demonstrates a model attack called Distributed Adversarial Regions, where an attacker creates adversarial noise near the object being classified rather than modifying the object itself. Harriet demontrates that the attack is effective against a wide range of open source image classifiers, many of which are used commercially, reducing confidence level of classifiers by an average of 40%. Harriet demonstrates this in a heist scenario where she creates earrings with an image of the adversarial regions and infiltrates Canberra Casino (with their permission). 

Learn More:
- [DEF CON Media Server: Presentations Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/DEF%20CON%2032%20-%20Harriet%20Farlow%20-%20On%20Your%20Oceans%2011%20Team%20Im%20the%20AI%20Guy%20%28technically%20Girl%29.pdf)
- [Arxiv: The race to robustness: exploiting fragile models for urban camouflage](https://arxiv.org/abs/2306.14609)
- [YouTube: HarrietHacks](https://www.youtube.com/@HarrietHacks/videos)


### Attacks on GenAI Data and Using Vector Encryption to Stop Them
Bob and Patrick from IronCore Labs presented a talk on the vulnerabilities of GenAI data and the potential of vector encryption to mitigate these threats. They began by explaining how data is typically fed into models using Retrieval-Augmented Generation (RAG) and vector search, and provided a primer on vector searches, vector databases, and their inner workings. Next, the presenters highlighted the risk of "inversion models" that can reverse-engineer embeddings back to their original meaning, compromising sensitive information. To address this concern, they highlight two primary encryption methods: Fully Homomorphic Encryption (FHE) and Partially Homomorphic Encryption (PHE). Focusing on the latter, they spoke about vector encryption using Distance Comparison Preserving Encryption (DCPE), which encrypts data using a scale and perturb technique preserving its vector comparability.

Learn More:
- [DEF CON Media Server: Presentation Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20villages/DEF%20CON%2032%20-%20Crypto%20%26%20Privacy%20Village%20-%20Patrick%20Walsh%20%26%20Bob%20Wall%20-%20Attacks%20on%20GenAI%20data%20and%20using%20vector%20encryption%20to%20stop%20them.pdf)
- [YouTube: RMISC 2024 - Exploitable Weaknesses in Gen AI Workflows: From RAG to Riches](https://www.youtube.com/watch?v=Mrx-i5M-RfU)
- [YouTube: Encrypting Vector Databases: How and Why Embeddings Need to be Protected](https://www.youtube.com/watch?v=e7fZvXkUfjU)
- [IronCore Labs: Securing AI: The Stack You Need to Protect Your GenAI Systems](https://ironcorelabs.com/blog/2024/the-ai-security-stack/)
- [IronCore Labs: Securing AI: The Hidden Dangers of Face Embeddings: Unmasking the Privacy Risks](https://ironcorelabs.com/blog/2024/face-embedding-privacy-risks)
- [IronCore Labs: There and Back Again: An Embedding Attack Journey](https://ironcorelabs.com/blog/2024/text-embedding-privacy-risks/) 
- [Arxiv: Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)
- [ICAR ePrint: Approximate Distance-Comparison-Preserving Symmetric Encryption](https://eprint.iacr.org/2021/1666.pdf)
- [GitHub: IronCore Alloy](https://github.com/IronCoreLabs/ironcore-alloy)


### Your AI Assistant has a Big Mouth: A New Side-Channel Attack
Researchers from Ben-Gurion University showcased a novel attack which gives an attacker the ability snoop on private conversations with chatbots that stream responses back one token at a time. Although the stream is HTTPS encrypted, a man-in-the-middle can capture and observe the size of each packet, which represents the length of the token. Next, the researchers trained a model to guess sentences from a list of token lengths. Their techniques was 55% effective in correctly guessing the first sentence of the respones and 38% effective in guessing the whole text against OpenAI GPT-4.

Learn More:
- [YouTube: Token Length Side Channel Video Demo](https://youtu.be/UfenH7xKO1s)
- [GitHub: GPT_Keylogger](https://github.com/royweiss1/GPT_Keylogger)
- [Arxiv: What Was Your Prompt? A Remote Keylogging Attack on AI Assistants](https://arxiv.org/abs/2403.09751)
- [Ars Technica: Hackers can read private AI-assistant chats](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/2/)
- [DEF CON Media Server: Presentation Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/DEF%20CON%2032%20-%20Harriet%20Farlow%20-%20On%20Your%20Oceans%2011%20Team%20Im%20the%20AI%20Guy%20%28technically%20Girl%29.pdf)


### Incubated Machine Learning Exploits: Backdooring ML Pipelines Using Input Handling Bugs
Suha from Trail of Bits covered how ML backdoor attacks are established, giving a malicious actor the ablity to force a model to produce specific outputs given the input of an attacker-chosen trigger. Suha highlighted how system security issues can be combined with model vulnerabilities to establish backdoors in models using input handling bugs. 

Learn More:
- [DEF CON Media Server: Presentation Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/DEF%20CON%2032%20-%20Suha%20Sabi%20Hussain%20-%20Incubated%20Machine%20Learning%20Exploits%20Backdooring%20ML%20Pipelines%20Using%20Input-Handling%20Bugs.pdf)
- [GitHub: Trail of Bits Publications](https://github.com/trailofbits/publications)
- [Vimeo Livestream: Talk Recording from HOPE XV](https://livestream.com/accounts/9198012/events/11160133/videos/248463591)


### Hacker vs AI: Prespectives From an Ex Spy
In this Policy talk, former data scientist and technical director at the Australian Signals Directorate Harriet Fowler shared her expertise on the intersection of AI and Cybersecurity. Fowler discussed how AI is being leveraged for both cyber defense and offense, highlighting the security implications of these developments. She also explored how Advanced Persistent Threat (APT) groups are utilizing AI to enhance their operations. Notably, Fowler argued that AI is not replacing traditional hacking methods, but rather augmenting the existing kill chain. She also called out important distinctions between AI for security and security of AI, providing a nuanced understanding of the relationships between the two concepts. 

- [DEF CON Media Server: Presentation Slides](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20villages/DEF%20CON%2032%20-%20Policy%20Village%20-%20Harriet%20Farlow%20-%20Hacker%20vs%20AI%20-%20perspectives%20from%20an%20ex-spy.pdf)
- [YouTube: HarrietHacks](https://www.youtube.com/@HarrietHacks/videos)
- [OCED.AI: Live Data](https://oecd.ai/en/data)
- [OCED.AI: Incidents](https://oecd.ai/en/incidents)


### I've Got 99 Problems but a Prompt Injection ain't Pineapple/Watermelon
Threat Intelligence and Security Researchers Chloe and Kasimir from HiddenLayer covered a variety of model attacks, including Poisoning, Evasion, and Theft. The team covered the extent of AI vulnerabilities, breaking them down into three categories: Model and Dataset, AI Infrastructure, and AI Development Frameworks. HiddenLayer makes a wealth of research available on their website.

Learn More:
- [HiddenLayer: Research](https://hiddenlayer.com/research/)
- [YouTube: HiddenLayer Webinar: A Guide to AI Red Teaming](https://www.youtube.com/watch?v=7JUoiNXO_Ok)


# Summary
As we reflect on DEF CON this year, it's clear that the AI + Security community is growing and evolving at a rapid pace. At Hugging Face, we're committed to staying at the forefront of security research and implementing the latest lessons learned into our own products and practices. We're already looking forward to next year's conference and the opportunity to continue learning from and contributing to the security community. If you have thoughts, questions or comments, please reach out at security@huggingface.co!

