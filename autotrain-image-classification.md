---
title: Image Classification with AutoTrain 
thumbnail: /blog/assets/103_autotrain-image-classification/thumbnail.png
---

<h1>
Image Classification with AutoTrain
</h1>

<div class="blog-metadata">
    <small>Published Sep 28th, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/autotrain-image-classification.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/nimaboscarino">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1647889744246-61e6a54836fa261c76dc3760.jpeg?w=200&h=200&f=face" width="100" title="Gravatar">
        <div class="bfc">
            <code>NimaBoscarino</code>
            <span class="fullname">Nima Boscarino</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

So you’ve heard all about the cool things that are happening in the machine learning world, and you want to join in. There’s just one problem – you don’t know how to code! 😱 Or maybe you’re a seasoned software engineer who wants to add some ML to your side-project, but you don’t have the time to pick up a whole new tech stack! For many people, the technical barriers to picking up machine learning feel insurmountable. That’s why Hugging Face created [AutoTrain](https://huggingface.co/autotrain), and with the latest feature we’ve just added, we’re making “no-code” machine learning better than ever. Best of all, you can create your first project for ✨ free! ✨

[Hugging Face AutoTrain](https://huggingface.co/autotrain) lets you train models with **zero** configuration needed. Just choose your task (translation? how about question answering?), upload your data, and let Hugging Face do the rest of the work! We’ve been expanding the number of tasks that we support, and we’re proud to announce that **you can now use AutoTrain for Computer Vision**! Image Classification is the latest task we’ve added, with more on the way. But what does this mean for you?

[Image Classification](https://huggingface.co/tasks/image-classification) models learn to *categorize* images, meaning that you can train one of these models to label any image. Do you want a model that can recognize signatures? Distinguish bird species? Identify plant diseases? As long as you can find an appropriate dataset, an image classification model has you covered.

## How can you train your own image classifier?

If you haven’t [created a Hugging Face account](https://huggingface.co/join) yet, now’s the time! Following that, make your way over to the [AutoTrain homepage](https://huggingface.co/autotrain) and click on “Create new project” to get started. You’ll be asked to fill in some basic info about your project. In the screenshot below you’ll see that I created a project named `butterflies-classification`, and I chose the “Image Classification” task. I’ve also chosen the “Automatic” model option, since I want to let AutoTrain do the work of finding the best model architectures for my project.

<div class="flex justify-center">
  <figure class="image table text-center m-0 w-1/2">
    <medium-zoom background="rgba(0,0,0,.7)" alt="The 'New Project' form for AutoTrain, filled out for a new Image Classification project named 'butterflies-classification'." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/new-project.png"></medium-zoom>
  </figure>
</div>

Once AutoTrain creates your project, you just need to connect your data. If you have the data locally, you can drag and drop the folder into the window. Since we can also use [any of the image classification datasets on the Hugging Face Hub](https://huggingface.co/datasets?task_categories=task_categories:image-classification), in this example I’ve decided to use the [NimaBoscarino/butterflies](https://huggingface.co/datasets/NimaBoscarino/butterflies) dataset. You can select separate training and validation datasets if available, or you can ask AutoTrain to split the data for you.

<div class="grid grid-cols-2 gap-4">
  <figure class="image table text-center m-0 w-full">
    <medium-zoom background="rgba(0,0,0,.7)" alt="A modal for importing a dataset to the AutoTrain project. A dataset named 'NimaBoscarino/butterflies' has been selected." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/import-dataset.png"></medium-zoom>
  </figure>

  <figure class="image table text-center m-0 w-full">
    <medium-zoom background="rgba(0,0,0,.7)" alt="A form showing configurations to select for the imported dataset, including split types and data columns." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/add-dataset.png"></medium-zoom>
  </figure>
</div>

Once the data has been added, simply choose the number of model candidates that you’d like AutoModel to try out, review the expected training cost (training with 5 candidate models and less than 500 images is free 🤩), and start training!

<div class="grid grid-cols-2 gap-4">
  <figure class="image table text-center m-0 w-full">
    <medium-zoom background="rgba(0,0,0,.7)" alt="Screenshot showing the model-selection options. Users can choose various numbers of candidate models, and the final training budget is displayed." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/select-models.png"></medium-zoom>
  </figure>
  <div>
    <figure class="image table text-center m-0 w-full">
      <medium-zoom background="rgba(0,0,0,.7)" alt="Five candidate models are being trained, one of which has already completed training." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/training-in-progress.png"></medium-zoom>
    </figure>
    <figure class="image table text-center m-0 w-full">
      <medium-zoom background="rgba(0,0,0,.7)" alt="All the candidate models have finished training, with one in the 'stopped' state." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/training-complete.png"></medium-zoom>
    </figure>
  </div>
</div>

In the screenshots above you can see that my project started 5 different models, which each reached different accuracy scores. One of them wasn’t performing very well at all, so AutoTrain went ahead and stopped it so that it wouldn’t waste resources. The very best model hit 84% accuracy, with effectively zero effort on my end 😍  To wrap it all up, you can visit your freshly trained models on the Hub and play around with them through the integrated [inference widget](https://huggingface.co/docs/hub/models-widgets). For example, check out my butterfly classifier model over at [NimaBoscarino/butterflies](https://huggingface.co/NimaBoscarino/butterflies) 🦋

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="An automatically generated model card for the butterflies-classification model, showing validation metrics and an embedded inference widget for image classification. The widget is displaying a picture of a butterfly, which has been identified as a Malachite butterfly." src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/autotrain-image-classification/model-card.png"></medium-zoom>
</figure>

We’re so excited to see what you build with AutoTrain! Don’t forget to join the community over at [hf.co/join/discord](https://huggingface.co/join/discord), and reach out to us if you need any help 🤗
