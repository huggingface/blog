---
title: "Introducing The World's Largest Open Multilingual Language Model: BLOOM"
thumbnail: /blog/assets/86_bloom/thumbnail.png
authors:
- user: bigscience
---

<html>
<head>
<link rel=“canonical” href=“http://bigscience.huggingface.co/blog/bloom” />
<style>
.grandmahugs {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;
}
</style>
<h1>🌸 Introducing The World's Largest Open Multilingual Language Model: BLOOM 🌸</h1>

{blog_metadata}

{authors}
</head>
<body>
<a href="https://huggingface.co/bigscience/bloom"><img style="middle" width="950" src="/blog/assets/86_bloom/thumbnail-2.png"></a>  

Large language models (LLMs) have made a significant impact on AI research. These powerful, general models can take on a wide variety of new language tasks from a user’s instructions. However, academia, nonprofits and smaller companies' research labs find it difficult to create, study, or even use LLMs as only a few industrial labs with the necessary resources and exclusive rights can fully access them. Today, we release [BLOOM](https://huggingface.co/bigscience/bloom), the first multilingual LLM trained in complete transparency, to change this status quo — the result of the largest collaboration of AI researchers ever involved in a single research project.

With its 176 billion parameters, BLOOM is able to generate text in 46 natural languages and 13 programming languages. For almost all of them, such as Spanish, French and Arabic, BLOOM will be the first language model with over 100B parameters ever created. This is the culmination of a year of work involving over 1000 researchers from 70+ countries and 250+ institutions, leading to a final run of 117 days (March 11 - July 6) training the BLOOM model on the [Jean Zay supercomputer](http://www.idris.fr/eng/info/missions-eng.html) in the south of Paris, France thanks to a compute grant worth an estimated €3M from French research agencies CNRS and GENCI.

Researchers can [now download, run and study BLOOM](https://huggingface.co/bigscience/bloom) to investigate the performance and behavior of recently developed large language models down to their deepest internal operations. More generally, any individual or institution who agrees to the terms of the model’s [Responsible AI License](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) (developed during the BigScience project itself) can use and build upon the model on a local machine or on a cloud provider. In this spirit of collaboration and continuous improvement, we’re also releasing, for the first time, the intermediary checkpoints and optimizer states of the training. Don’t have 8 A100s to play with? An inference API, currently backed by Google’s TPU cloud and a FLAX version of the model, also allows quick tests, prototyping, and lower-scale use. You can already play with it on the Hugging Face Hub.

<img class="grandmahugs" style="center" width="950" src="/blog/assets/86_bloom/bloom-examples.jpg"></a>  

This is only the beginning. BLOOM’s capabilities will continue to improve as the workshop continues to experiment and tinker with the model. We’ve started work to make it instructable as our earlier effort T0++ was and are slated to add more languages, compress the model into a more usable version with the same level of performance, and use it as a starting point for more complex architectures… All of the experiments researchers and practitioners have always wanted to run, starting with the power of a 100+ billion parameter model, are now possible. BLOOM is the seed of a living family of models that we intend to grow, not just a one-and-done model, and we’re ready to support community efforts to expand it.

</body>
</html>
