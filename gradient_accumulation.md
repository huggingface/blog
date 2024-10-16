
---
title: "Fixing Gradient Accumulation" 
thumbnail: /blog/assets/gradient_accumulation/gradient_accumulation.png
authors:
- user: lysandre
- user: ArthurZ
- user: muellerzr
- user: ydshieh
- user: BenjaminB
---

# Fixing Gradient Accumulation

Our friends at Unsloth [shared an issue](https://unsloth.ai/blog/gradient) regarding gradient accumulation yesterday that is affecting the transformers Trainer. The initial report comes from @bnjmn_marie (kudos to him!).

Gradient accumulation is *supposed* to be mathematically equivalent to full batch training; however, losses did not match between training runs where the setting was toggled on and off.

## Where does it stem from?

Inside the modeling code of each model, `transformers` offers a "default" loss function that's the most typically used one for the model's task. It is determined by what the modeling class should be used for: question answering, token classification, causal LM, masked LM.

This is the default loss function and it was not meant to be customizable: it is only computed when `labels` and `input_ids` are passed as inputs to the model, so the user doesn't have to compute the loss. The default loss is useful but is limited **by design**: for anything different being done, we expect the labels to **not be passed directly, and for users to get the logits back from the model and use them to compute the loss outside of the model.**

However, the transformers Trainer, as well as many Trainers, heavily leverage these methods because of the simplicity it offers: it is a double-edged sword. Providing a simple API that becomes different as the use-case differs is not a well-thought out API, and we've been caught by surprise ourselves.

To be precise, for gradient accumulation across token-level tasks like causal LM training, the correct loss should be computed by the total loss across all batches in a gradient accumulation step divided by the total number of all non padding tokens in those batches. This is not the same as the average of the per-batch loss values.
The fix is quite simple, see the following:
```diff
def ForCausalLMLoss(logits, labels, vocab_size, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    num_items = kwargs.pop("num_items", None)
+        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
+        loss = loss / num_items
-        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss
## How we're fixing it

To address this issue, we’re changing the way our models and training work in two ways:

* If users are using the “default” loss functions, we will automatically take into account the needed changes when using gradient accumulation, to make sure the proper loss is reported and utilized, fixing the core issue at hand. 
* To ensure that any future issues with calculating losses won’t block users, we’ll be exposing an API to let users pass in their own loss functions to the `Trainer` directly so they can use their own fix easily until we have fixed any issues internally and made a new transformers release. 

All model that inherit from `PreTrainedModel` now have a `loss_function` property, which is determined by either: 
- the `config.loss_type`: this is to make sure anyone can use his custom loss. You can do this by modifying the `LOSS_MAPPING`:
```python
def my_super_loss(logits, labels):
    return loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)


LOSS_MAPPING["my_loss_type"] = my_super_loss

We are working to ship the first change for the most popular models in this PR: https://github.com/huggingface/transformers/pull/34191#pullrequestreview-2372725010. Following this, a call for contributions to help propagate this to the rest of the models will be done so that the majority of models is supported by next release.

We are also actively working to ship the second change in this PR: https://github.com/huggingface/transformers/pull/34198, which will allow users to use their own loss function and make use of the number of samples seen per-batch to help with calculating their loss (and will perform the correct loss calculation during gradient accumulation as more models are supported from the prior change)

—

By tomorrow, you should expect the Trainer to behave correctly with gradient accumulation. Please install from `main` in order to benefit from the fix then:

```
pip install git+https://github.com/huggingface/transformers
```

In general, we are very responsive to bug reports submitted to our issue tracker: https://github.com/huggingface/transformers/issues

This issue has been in Transformers for some time as it's mostly a default that should be updated by the end-user; however, when defaults become non-intuitive, they are bound to be changed. In this instance, we've updated the code and shipped a fix in less than 24 hours, which is what we aim for issues like this one in transformers. Please, come and submit your issues if you have some; this is the only way we can get transformers to improve and fit well within your different use-cases.

The Transformers team 🤗
