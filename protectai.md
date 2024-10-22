---
title: "Hugging Face Teams Up with Protect AI: Enhancing Model Security for the Community"
thumbnail: /blog/assets/protectai-partnership/thumbnail.png
authors:
- user: mcpotato
---


We are very pleased to announce our partnership with Protect AI, as part of our [long-standing commitment](https://huggingface.co/blog/2024-security-features) to provide a safe and reliable platform for the community.

[Protect AI](https://protectai.com/) is a company founded with a mission to create a safer AI powered world. They are developing powerful tools, namely [Guardian](https://protectai.com/guardian), to ensure that the rapid pace of AI innovation can continue without compromising on security.

Our decision to partner with Protect AI stems from their [community driven](https://huntr.com/) approach to security, active support of [open source](https://github.com/protectai) and expertise in all things security x AI.


## Model security refresher

To share models, we serialize the data structures we use to interact with the models, in order to facilitate storage and transport. Some serialization formats are vulnerable to nasty exploits, such as arbitrary code execution (looking at you pickle), making sharing models potentially dangerous.

As Hugging Face has become the de facto platform for model sharing, we’d like to protect the community from this, hence why we have developed tools like [picklescan](https://github.com/mmaitre314/picklescan) and why we are integrating Guardian in our scanner suite.

Pickle is not the only exploitable format out there, [see for reference](https://github.com/Azure/counterfit/wiki/Abusing-ML-model-file-formats-to-create-malware-on-AI-systems:-A-proof-of-concept) how one can exploit Keras Lambda layers to achieve arbitrary code execution. The good news is that Guardian catches both of these exploits and more in additional file formats! See their [Knowledge Base](https://protectai.com/insights/knowledge-base/) for up to date scanner information.


## Integration

In the effort of integrating Guardian, we have capitalized on the opportunity to revamp our frontend to display scan results. Here is what it now looks like:

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/third-party-scans-list.png"/>

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/security-scanner-status-banner.png"/>

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/security-scanner-pickle-import-list.png"/>
<em>As you can see here, the old pickle button remains the same and present when a pickle import scan occurred</em>

As you can see from the pictures, you have nothing to do to benefit from this! All public model repositories will be scanned by Guardian automatically as soon as you push your files to the Hub. Here is an example repository you can check out to see the feature in action: [mcpotato/42-eicar-street](https://huggingface.co/mcpotato/42-eicar-street).

Note that you might not see a scan for your model as of today, as we have over 1 million, it may take some time to catch up.

In total, we have already scanned hundreds of millions of files, because we believe that empowering the community to share models in a safe and frictionless manner will lead to growth for everyone, including us.

*Interested in joining our security partnership / providing scanning information on the Hub? Please get in touch with us over at security@huggingface.co.*
