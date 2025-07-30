---
title: "Introducing Training Cluster as a Service - a new collaboration with NVIDIA" 
thumbnail: /blog/assets/nvidia-training-cluster/nvidia-training-cluster-thumbnail-compressed.png
authors:
- user: jeffboudier
- user: ark393
- user: pagezyhf
---

# Introducing Training Cluster as a Service - a new collaboration with NVIDIA

Today at GTC Paris, we are excited to announce [Training Cluster as a Service](https://huggingface.co/training-cluster) in collaboration with NVIDIA, to make large GPU clusters more easily accessible for research organizations all over the world, so they can train the foundational models of tomorrow in every domain.

## Making GPU Clusters Accessible

Many Gigawatt-size GPU supercluster projects are being built to train the next gen of AI models. This can make it seem that the compute gap between the “GPU poor” and the “GPU rich” is quickly widening. But the GPUs are out there, as hyperscalers, regional and AI-native cloud providers all quickly expand their capacity. 

How do we then connect AI compute capacity with the researchers who need it? How do we enable universities, national research labs and companies all over the world to build their own models?

This is what Hugging Face and NVIDIA are tackling with Training Cluster as a Service - providing GPU cluster accessibility, with the flexibility to only pay for the duration of training runs.

To get started, any of the 250,000 organizations on Hugging Face can request the GPU cluster size they need, when they need it.

## How it works

To get started, you can request a GPU cluster on behalf of your organization at [hf.co/training-cluster](https://huggingface.co/training-cluster)
 
Training Cluster as a Service integrates key components from NVIDIA and Hugging Face into a complete solution:
- NVIDIA Cloud Partners provide capacity for the latest NVIDIA accelerated computing like NVIDIA Hopper and [NVIDIA GB200](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) in regional datacenters, all centralized within [NVIDIA DGX Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/?ncid=pa-srch-goog-128355-DGX-Brand-prsp&_bt=749738455198&_bk=nvidia%20dgx%20cloud&_bm=b&_bn=g&_bg=180515995564&gad_source=1&gad_campaignid=22505579974&gbraid=0AAAAAD4XAoGVodXdazBIYN4fH53MAZVLQ&gclid=EAIaIQobChMI98m9zOjdjQMVLBHUAR2l4hg5EAAYASAAEgKMZvD_BwE)
- NVIDIA DGX Cloud Lepton - announced today at GTC Paris - provides easy access to the infrastructure provisioned for researchers, and enables training run scheduling and monitoring
- Hugging Face developer resources and open source libraries make it easy to get training runs started.

Once your GPU cluster request is accepted, Hugging Face and NVIDIA will collaborate to source, price, provision and set up your GPU cluster per your size, region and duration requirements.

## Clusters at Work

### Advancing Rare Genetic Disease Research with TIGEM

The [Telethon Institute of Genomics and Medicine](https://huggingface.co/TigemAI) - TIGEM for short - is a research center dedicated to understanding the molecular mechanisms behind rare genetic diseases and developing novel treatments. Training new AI models is a new path to predict the effect of pathogenic variants and for drug repositioning.

> AI offers new ways to research the causes of rare genetic diseases and to develop treatments, but our domain requires training new models. Training Cluster as a Service made it easy to procure the GPU capacity we needed, at the right time

-- _Diego di Bernardo, Coordinator of the Genomic Medicine program at TIGEM_

### Advancing AI for Mathematics with Numina

[Numina](https://huggingface.co/AI-MO) is a non-profit organization building open-source, open-dataset AI for reasoning in math - and won the 2024 [AIMO progress prize](https://aimoprize.com/). 

> We are tracking well on our objective to build open alternatives to the best closed-source models, such as Deepmind's AlphaProof. Computing resources is our bottleneck today - with Training Cluster as a Service we will be able to reach our goal!

-- _Yann Fleureau, cofounder of Project Numina_

### Advancing Material Science with Mirror Physics

Mirror Physics is a startup creating frontier AI systems for chemistry and materials science.

> Together with the MACE team, we're working to push the limits of AI for chemistry. With Training Cluster as a Service, we're producing high-fidelity chemical models at unprecedented scale. This is going to be a significant step forward for the field.

-- _Sam Walton Norwood, CEO and founder at Mirror_

## Powering the Diversity of AI Research

Training Cluster as a Service is a new collaboration between Hugging Face and NVIDIA to make AI compute more readily available to the global community of AI researchers.

> Access to large-scale, high-performance compute is essential for building the next generation of AI models across every domain and language. Training Cluster as a Service will remove barriers for researchers and companies, unlocking the ability to train the most advanced models and push the boundaries of what’s possible in AI.

-- _Clément Delangue, cofounder and CEO of Hugging Face_

> Integrating DGX Cloud Lepton with Hugging Face’s Training Cluster as a Service gives developers and researchers a seamless way to access high-performance NVIDIA GPUs across a broad network of cloud providers. This collaboration makes it easier for AI researchers and organizations to scale their AI training workloads while using familiar tools on Hugging Face.

-- _Alexis Bjorlin, vice president of DGX Cloud at NVIDIA_

## Enabling AI Builders with NVIDIA
We are excited to collaborate with NVIDIA to offer Training Cluster as a Service to Hugging Face organizations - you can get started today at hf.co/training-cluster

Today at GTC Paris, NVIDIA announced many new contributions for Hugging Face users, from agents to robots!
- [NVIDIA DGX Cloud Lepton connects Europe’s developers to global NVIDIA compute ecosystem](https://nvidianews.nvidia.com/news/nvidia-dgx-cloud-lepton-connects-europes-developers-to-global-nvidia-compute-ecosystem)
- [NVIDIA AI customers can now deploy over 100k Hugging Face models with NIM](https://developer.nvidia.com/blog/simplify-llm-deployment-and-ai-inference-with-unified-nvidia-nim-workflow/)
- Hugging Face users can build custom Physical AI models with NVIDIA Cosmos Predict-2
- NVIDIA Isaac GR00T N1.5 lands on Hugging Face to power humanoid robots
