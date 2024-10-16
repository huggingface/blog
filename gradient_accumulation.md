
---
title: "Fixing Gradient Accumulation" 
thumbnail: /blog/assets/159_autogptq_transformers/thumbnail.jpg
authors:
- user: lysandre
- user: ArthurZ
- user: muellerzr
---

# Fixing Gradient Accumulation

Our friends at Unsloth shared an issue regarding gradient accumulation yesterday that is affecting the transformers Trainer.

Gradient accumulation is supposed to be mathematically equivalent to full batch training; however, losses did not match between training runs where the setting was toggled on and off.

### Where does it stem from?

First of all, inside the modeling code of each model, transformers offers a "default" loss function that's the most typically used loss function by the model architecture. It depends on what the class itself should be used for: question answering, token classification, causal LM, masked LM.

This is the default method which is not meant to be customizable: it is only computed when labels, as well as input_ids, are passed sa inputs to the model. The default loss is useful but is limited by design: for anything different being done, then we expect the labels to not be passed directly, and for users to get the logits back from the model and use them to compute the loss outside of the model.

However, the transformers Trainer, as well as many others, is heavily leveraging these methods: by the simplicity it offers, it is a double-edged sword. Providing a simple API that becomes different as the use-case differs is not a well-thought API, and we've been caught by surprise ourselves.

### How we're fixing it

To address this issue, we’re changing the way our models and training works in two ways:
If users are using the “default” loss functions by each model architecture, we will automatically take into account the needed changes when using gradient accumulation on them to make sure the proper loss is reported and utilized, fixing the core issue at hand. 
To ensure that any future issues with calculating losses might come up in the future won’t block users, we’ll be exposing an API to let users pass in their own loss functions to the `Trainer` directly so they can use their own fix in easily until we have fixed any issues internally and pushed onto a new release. 

### When will it land

We are working to ship the first change for the most popular models in this PR: https://github.com/huggingface/transformers/pull/34191#pullrequestreview-2372725010
Following this, a call for contributions to help propagate this to the rest of the models will be done so that the majority of models is supported by next release.

We are also actively working to ship the second change in this PR: https://github.com/huggingface/transformers/pull/34198, which will allow users to use their own loss function and make use of the number of samples seen per-batch to help with calculating their loss (and will perform the correct loss calculation during gradient accumulation as more models are supported from the prior ship)

—

By tomorrow, you should expect the Trainer to behave correctly with gradient accumulation. In general, we are very responsive to bug reports submitted to our issue tracker: https://github.com/huggingface/transformers/issues

This particular bug was fixed within 24 hours of disclosure; and that’s what we aim for bugs like this one in transformers. Please, come and submit your issues if you have some; this is the only way we can get transformers to improve and fit well within your different use-cases.

The Transformers team 🤗
