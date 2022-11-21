---
title: "XXX"
thumbnail: /blog/assets/XXX/XXX

---

<h1>XXX</h1>

<div class="blog-metadata">
    <small>Published November XXX, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/openvino.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
        <a href="https://twitter.com/julsimon">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1633343465505-noauth.jpeg?w=128&h=128&f=face" title="Julien Simon">
        <div class="bfc">
            <code>juliensimon</code>
            <span class=fullname">Julien Simon</span>
        </div>
    </a>
</div>


Transformer models are quickly becoming the de facto architecture for a wide range of machine learning (ML) applications, including natural language processing, computer vision, speech, and more. Every day, developers and organizations are adopting these models to turn ideas into proof-of-concept demos, and demos into production-grade applications. 

At Hugging Face, we are obsessed with simplifying ML development and operations, without compromising on state-of-the-art quality. In this respect, the ability to test and deploy the latest models with minimal friction is critical all along the lifecycle of an ML project. Optimizing the cost-performance ratio is equally important, and we'd like to thank our friends at [Intel](https://huggingface.co/intel) for sponsoring our free CPU-based inference solutions. This is another major step in our [partnership](https://huggingface.co/blog/intel). It's also great news for our user community, who can now enjoy the speedup delivered by the [Intel Xeon Ice Lake](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/3rd-gen-xeon-scalable-processors-brief.html) architecture at zero cost.

Now, let's review your inference options with Hugging Face.

## Free Inference Widget

One of my favorite features on the Hugging Face hub is the Inference [Widget](https://huggingface.co/docs/hub/models-widgets). Located on the model page, the Inference Widget lets you upload sample data and predict it in a single click. 

XXX Screenshot

It's the best way to quickly get a sense of what a model does, what its output looks like, and how it performs on a few samples from your dataset. The model is loaded on-demand on our servers and unloaded when it's not needed anymore. You don't have to write any code and the feature is free. What's not to love?
 
## Free Inference API

The [Inference API](https://huggingface.co/docs/api-inference/) is equally developer-friendly. With a simple HTTP request, you can load any hub model and predict your data with it in seconds. The model URL and a valid hub token are all you need, for example: 

```
curl https://api-inference.huggingface.co/models/xlm-roberta-base \
	-X POST \
	-d '{"inputs": "The answer to the universe is <mask>."}' \
	-H "Authorization: Bearer HF_TOKEN"
```

The Inference API is the simplest way to build a prediction service that you can immediately call from your application during development and tests. No need for a bespoke API, or a model server. In addition, you can instantly switch from one model to the next and compare their performance in your application. And guess what? The Inference API is free to use. 

As rate limiting is enforced, we don't recommend using the Inference API for production. Instead, you should consider Inference Endpoints.

## Production with Inference Endpoints

Once you're happy with the performance of your ML model, it's time to deploy it for production. Unfortunately, when you leave the sandbox, everything becomes a concern: security, scaling, monitoring, etc. This is where a lot of ML stumble and sometimes fall.
We built [Inference Endpoints](https://huggingface.co/inference-endpoints) to solve this problem.

In just a few clicks, you deploy any hub model on secure and scalable infrastructure hosted in your AWS or Azure region of choice. Additional settings include CPU and GPU hosting, built-in auto-scaling, and more. This lets you find the appropriate cost/performance ratio, with [pricing](https://huggingface.co/pricing#endpoints) starting as low as $0.06 per hour.

XXX screenshot

Inference Endpoints support three security levels:

* Public: the endpoint runs in a public Hugging Face subnet, and anyone on the Internet can access it without any authentication.

* Protected: the endpoint runs in a public Hugging Face subnet, and anyone on the Internet with the appropriate organization token can access it.

* Private: the endpoint runs in a private Hugging Face subnet. It's not accessible on the Internet. It's only available through a private connection in your AWS or Azure account. This will satisfy the strictest compliance requirements.

To learn more, please read this [tutorial](https://huggingface.co/blog/inference-endpoints) and the [documentation](https://huggingface.co/docs/inference-endpoints/).

## Getting started

It couldn't be simpler. Just log in to the Hugging Face [hub](https://huggingface.co/) and browse our [models](https://huggingface.co/models). Once you've found one that you like, you can try the Inference Widget directly on the page. Clicking on the "Deploy" button, you'll get auto-generated code to deploy the model on the free Inference API for evaluation, and a direct link to deploy it to production with Inference Endpoints.

Please give it a try and let us know what you think. We'd love to read your feedback on the Hugging Face [forum](https://discuss.huggingface.co/).

Thank you for reading!




