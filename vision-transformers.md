---
title: "Deep Dive: Vision Transformers On Hugging Face Optimum Graphcore"
thumbnail: /blog/assets/97_vision_transformers/thumbnail.png
---

<h1>Deep Dive: Vision Transformers On Hugging Face Optimum Graphcore</h1>

<div class="blog-metadata">
    <small>Published August, 18 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/vision-transformers.md">
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
<div class="author-card">
    <a href="https://huggingface.co/internetoftim">
        <img class="avatar avatar-user" src="https://huggingface.co/avatars/94a72281274ff9f1259384af15d73861.svg" title="Tim Santos">
        <div class="bfc">
            <code>Tim Santos (Graphcore)</code>
            <span class=fullname">Tim Santos</span>
        </div>
    </a>
</div>
<link rel="canonical" href="https://www.graphcore.ai/posts/deep-dive-vision-transformers-on-hugging-face-optimum-graphcore" />

This blog post will show how easy it is to fine-tune pre-trained Transformer models for your dataset using the Hugging Face Optimum library on Graphcore Intelligence Processing Units (IPUs). As an example, we will show a step-by-step guide and provide a notebook that takes a large, widely-used chest X-ray dataset and trains a vision transformer (ViT) model.

<h2>Introducing vision transformer (ViT) models</h2>

<p>In 2017 a group of Google AI researchers published a paper introducing the transformer model architecture. Characterised by a novel self-attention mechanism, transformers were proposed as a new and efficient group of models for language applications. Indeed, in the last five years, transformers have seen explosive popularity and are now accepted as the de facto standard for natural language processing (NLP).</p>
<p>Transformers for language are perhaps most notably represented by the rapidly evolving GPT and BERT model families. Both can run easily and efficiently on Graphcore IPUs as part of the growing <a href="/posts/getting-started-with-hugging-face-transformers-for-ipus-with-optimum" rel="noopener" target="_blank">Hugging Face Optimum Graphcore library</a>).</p>
<p><img src="https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=1024&amp;name=transformers_chrono.png" alt="transformers_chrono" loading="lazy" style="width: 1024px; margin-left: auto; margin-right: auto; display: block;" width="1024" srcset="https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=512&amp;name=transformers_chrono.png 512w, https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=1024&amp;name=transformers_chrono.png 1024w, https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=1536&amp;name=transformers_chrono.png 1536w, https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=2048&amp;name=transformers_chrono.png 2048w, https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=2560&amp;name=transformers_chrono.png 2560w, https://www.graphcore.ai/hs-fs/hubfs/transformers_chrono.png?width=3072&amp;name=transformers_chrono.png 3072w" sizes="(max-width: 1024px) 100vw, 1024px"></p>
<div class="blog-caption" style="max-height: 100%; max-width: 90%; margin-left: auto; margin-right: auto; line-height: 1.4;">
<p>A timeline showing releases of prominent transformer language models (credit: Hugging Face)</p>
</div>
<p>An in-depth explainer about the transformer model architecture (with a focus on NLP) can be found <a href="https://huggingface.co/course/chapter1/4?fw=pt" rel="noopener" target="_blank">on the Hugging Face website</a>.</p>
<p>While transformers have seen initial success in language, they are extremely versatile and can be used for a range of other purposes including computer vision (CV), as we will cover in this blog post.</p>
<p>CV is an area where convolutional neural networks (CNNs) are without doubt the most popular architecture. However, the vision transformer (ViT) architecture, first introduced in a <a href="https://arxiv.org/abs/2010.11929" rel="noopener" target="_blank">2021 paper</a> from Google Research, represents a breakthrough in image recognition and uses the same self-attention mechanism as BERT and GPT as its main component.</p>
<p>Whereas BERT and other transformer-based language processing models take a sentence (i.e., a list of words) as input, ViT models divide an input image into several small patches, equivalent to individual words in language processing. Each patch is linearly encoded by the transformer model into a vector representation that can be processed individually. This approach of splitting images into patches, or visual tokens, stands in contrast to the pixel arrays used by CNNs.</p>
<p>Thanks to pre-training, the ViT model learns an inner representation of images that can then be used to extract visual features useful for downstream tasks. For instance, you can train a classifier on a new dataset of labelled images by placing a linear layer on top of the pre-trained visual encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.</p>
<p><img src="https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=1024&amp;name=vit%20diag.png" alt="vit diag" loading="lazy" style="width: 1024px; margin-left: auto; margin-right: auto; display: block;" width="1024" srcset="https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=512&amp;name=vit%20diag.png 512w, https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=1024&amp;name=vit%20diag.png 1024w, https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=1536&amp;name=vit%20diag.png 1536w, https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=2048&amp;name=vit%20diag.png 2048w, https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=2560&amp;name=vit%20diag.png 2560w, https://www.graphcore.ai/hs-fs/hubfs/vit%20diag.png?width=3072&amp;name=vit%20diag.png 3072w" sizes="(max-width: 1024px) 100vw, 1024px"></p>
<div class="blog-caption" style="max-height: 100%; max-width: 90%; margin-left: auto; margin-right: auto; line-height: 1.4;">
<p>An overview of the ViT model structure as introduced in <a href="https://arxiv.org/abs/2010.11929" rel="noopener" target="_blank">Google Research’s original 2021 paper</a></p>
</div>
<p>Compared to CNNs, ViT models have displayed higher recognition accuracy with lower computational cost, and are applied to a range of applications including image classification, object detection, and segmentation. Use cases in the healthcare domain alone include detection and classification for <a href="https://www.mdpi.com/1660-4601/18/21/11086/pdf" rel="noopener" target="_blank">COVID-19</a>, <a href="https://towardsdatascience.com/vision-transformers-for-femur-fracture-classification-480d62f87252" rel="noopener" target="_blank">femur fractures</a>, <a href="https://iopscience.iop.org/article/10.1088/1361-6560/ac3dc8/meta" rel="noopener" target="_blank">emphysema</a>, <a href="https://arxiv.org/abs/2110.14731" rel="noopener" target="_blank">breast cancer</a>, and <a href="https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full" rel="noopener" target="_blank">Alzheimer’s disease</a>—among many others.</p>
<h2>ViT models – a perfect fit for IPU</h2>
<p>Graphcore IPUs are particularly well-suited to ViT models due to their ability to parallelise training using a combination of data pipelining and model parallelism. Accelerating this massively parallel process is made possible through IPU’s MIMD architecture and its scale-out solution centred on the IPU-Fabric.</p>
<p>By introducing pipeline parallelism, the batch size that can be processed per instance of data parallelism is increased, the access efficiency of the memory area handled by one IPU is improved, and the communication time of parameter aggregation for data parallel learning is reduced.</p>
<p>Thanks to the addition of a range of pre-optimized transformer models to the open-source Hugging Face Optimum Graphcore library, it’s incredibly easy to achieve a high degree of performance and efficiency when running and fine-tuning models such as ViT on IPUs.</p>
<p>Through Hugging Face Optimum, Graphcore has released ready-to-use IPU-trained model checkpoints and configuration files to make it easy to train models with maximum efficiency. This is particularly helpful since ViT models generally require pre-training on a large amount of data. This integration lets you use the checkpoints released by the original authors themselves within the Hugging Face model hub, so you won’t have to train them yourself. By letting users plug and play any public dataset, Optimum shortens the overall development lifecycle of AI models and allows seamless integration to Graphcore’s state-of-the-art hardware, giving a quicker time-to-value.</p>
<p>For this blog post, we will use a ViT model pre-trained on ImageNet-21k, based on the paper <a href="https://arxiv.org/abs/2010.11929" rel="noopener" target="_blank">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> by Dosovitskiy et al. As an example, we will show you the process of using Optimum to fine-tune ViT on the <a href="https://paperswithcode.com/dataset/chestx-ray14" rel="noopener" target="_blank">ChestX-ray14 Dataset</a>.</p>
<h2>The value of ViT models for X-ray classification</h2>
<p>As with all medical imaging tasks, radiologists spend many years learning reliably and efficiently detect problems and make tentative diagnoses on the basis of X-ray images. To a large degree, this difficulty arises from the very minute differences and spatial limitations of the images, which is why computer aided detection and diagnosis (CAD) techniques have shown such great potential for impact in improving clinician workflows and patient outcomes.</p>
<p>At the same time, developing any model for X-ray classification (ViT or otherwise) will entail its fair share of challenges:</p>
<ul>
<li>Training a model from scratch takes an enormous amount of labeled data;</li>
<li>The high resolution and volume requirements mean powerful compute is necessary to train such models; and</li>
<li>The complexity of multi-class and multi-label problems such as pulmonary diagnosis is exponentially compounded due to the number of disease categories.</li>
</ul>
<p>As mentioned above, for the purpose of our demonstration using Hugging Face Optimum, we don’t need to train ViT from scratch. Instead, we will use model weights hosted in the <a href="https://huggingface.co/google/vit-base-patch16-224-in21k" rel="noopener" target="_blank">Hugging Face model hub</a>.</p>
<p>As an X-ray image can have multiple diseases, we will work with a multi-label classification model. The model in question uses <a href="https://huggingface.co/google/vit-base-patch16-224-in21k" rel="noopener" target="_blank">google/vit-base-patch16-224-in21k</a> checkpoints. It has been converted from the <a href="https://github.com/rwightman/pytorch-image-models" rel="noopener" target="_blank">TIMM repository</a> and pre-trained on 14 million images from ImageNet-21k. In order to parallelise and optimise the job for IPU, the configuration has been made available through the <a href="https://huggingface.co/Graphcore/vit-base-ipu" rel="noopener" target="_blank">Graphcore-ViT model card</a>.</p>
<p>If this is your first time using IPUs, read the <a href="https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/" rel="noopener" target="_blank">IPU Programmer's Guide</a> to learn the basic concepts. To run your own PyTorch model on the IPU see the <a href="https://github.com/graphcore/tutorials/blob/master/tutorials/pytorch/basics" rel="noopener" target="_blank">Pytorch basics tutorial</a>, and learn how to use Optimum through our <a href="https://github.com/huggingface/optimum-graphcore/tree/main/notebooks" rel="noopener" target="_blank">Hugging Face Optimum Notebooks</a>.</p>
<h2>Training ViT on the ChestXRay-14 dataset</h2>
<p>First, we need to download the National Institutes of Health (NIH) Clinical Center’s <a href="http://nihcc.app.box.com/v/ChestXray-NIHCC" rel="noopener" target="_blank">Chest X-ray dataset</a>. This dataset contains 112,120 deidentified frontal view X-rays from 30,805 patients over a period from 1992 to 2015. The dataset covers a range of 14 common diseases based on labels mined from the text of radiology reports using NLP techniques.</p>
<p><img src="https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=700&amp;name=chest%20x-ray%20examples.png" alt="chest x-ray examples" loading="lazy" style="width: 700px; margin-left: auto; margin-right: auto; display: block;" width="700" srcset="https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=350&amp;name=chest%20x-ray%20examples.png 350w, https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=700&amp;name=chest%20x-ray%20examples.png 700w, https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=1050&amp;name=chest%20x-ray%20examples.png 1050w, https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=1400&amp;name=chest%20x-ray%20examples.png 1400w, https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=1750&amp;name=chest%20x-ray%20examples.png 1750w, https://www.graphcore.ai/hs-fs/hubfs/chest%20x-ray%20examples.png?width=2100&amp;name=chest%20x-ray%20examples.png 2100w" sizes="(max-width: 700px) 100vw, 700px"></p>
<div class="blog-caption" style="max-height: 100%; max-width: 90%; margin-left: auto; margin-right: auto; line-height: 1.4;">
<p>Eight visual examples of common thorax diseases (Credit: NIC)</p>
</div>
<h2>Setting up the environment</h2>
<p>Here are the requirements to run this walkthrough:</p>
<ul>
<li>A Jupyter Notebook server with the latest Poplar SDK and PopTorch environment enabled (see our <a href="https://github.com/graphcore/tutorials/blob/master/tutorials/standard_tools/using_jupyter/README.md" rel="noopener" target="_blank">guide on using IPUs from Jupyter notebooks</a>)</li>
<li>The ViT Training Notebook from the <a href="https://github.com/graphcore/tutorials" rel="noopener" target="_blank">Graphcore Tutorials repo</a></li>
</ul>
<p>The Graphcore Tutorials repository contains the step-by-step tutorial notebook and Python script discussed in this guide. Clone the repository and launch the walkthrough.ipynb notebook found in&nbsp; <code><a href="https://github.com/graphcore/tutorials" rel="noopener" target="_blank">tutorials</a>/<a href="https://github.com/graphcore/tutorials/tree/master/tutorials" rel="noopener" target="_blank">tutorials</a>/<a href="https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch" rel="noopener" target="_blank">pytorch</a>/vit_model_training/</code>.</p>
<p style="font-weight: bold;">We’ve even made it easier and created the HF Optimum Gradient so you can launch the getting started tutorial in Free IPUs. <a href="http://paperspace.com/graphcore" rel="noopener" target="_blank">Sign up</a> and launch the runtime:<br><a href="https://console.paperspace.com/github/gradient-ai/Graphcore-HuggingFace?machine=Free-IPU-POD16&amp;container=graphcore%2Fpytorch-jupyter%3A2.6.0-ubuntu-20.04-20220804&amp;file=%2Fget-started%2Fwalkthrough.ipynb" rel="noopener" target="_blank"><img src="https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=200&amp;name=gradient-badge-gradient-05-d-05.png" alt="run on Gradient" loading="lazy" style="width: 200px; float: left;" width="200" srcset="https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=100&amp;name=gradient-badge-gradient-05-d-05.png 100w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=200&amp;name=gradient-badge-gradient-05-d-05.png 200w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=300&amp;name=gradient-badge-gradient-05-d-05.png 300w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=400&amp;name=gradient-badge-gradient-05-d-05.png 400w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=500&amp;name=gradient-badge-gradient-05-d-05.png 500w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=600&amp;name=gradient-badge-gradient-05-d-05.png 600w" sizes="(max-width: 200px) 100vw, 200px"></a></p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<h2>Getting the dataset</h2>
<a id="getting-the-dataset" data-hs-anchor="true"></a>
<p>Download the <a href="http://nihcc.app.box.com/v/ChestXray-NIHCC" rel="noopener" target="_blank">dataset's</a> <code>/images</code> directory. You can use <code>bash</code> to extract the files: <code>for f in images*.tar.gz; do tar xfz "$f"; done</code>.</p>
<p>Next, download the <code>Data_Entry_2017_v2020.csv</code> file, which contains the labels. By default, the tutorial expects the <code>/images</code> folder and .csv file to be in the same folder as the script being run.</p>
<p>Once your Jupyter environment has the datasets, you need to install and import the latest Hugging Face Optimum Graphcore package and other dependencies in <code><a href="https://github.com/graphcore/tutorials/blob/master/tutorials/pytorch/vit_model_training/requirements.txt" rel="noopener" target="_blank">requirements.txt</a></code>:</p>
<p><span style="color: #6b7a8c;"><code>%pip install -r requirements.txt </code></span></p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/24206176ff0ae6c1780dc47893997b80.js"></script>
</div>
<p><span style="color: #6b7a8c;"><code></code></span><code><span style="color: #6b7a8c;"></span></code></p>
<p>The examinations contained in the Chest X-ray dataset consist of X-ray images (greyscale, 224x224 pixels) with corresponding metadata: <code>Finding Labels, Follow-up #,Patient ID, Patient Age, Patient Gender, View Position, OriginalImage[Width Height] and OriginalImagePixelSpacing[x y]</code>.</p>
<p>Next, we define the locations of the downloaded images and the file with the labels to be downloaded in <a href="#getting-the-dataset" rel="noopener">Getting the dataset</a>:</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/cbcf9b59e7d3dfb02221dfafba8d8e10.js"></script>
</div>
<p>We are going to train the Graphcore Optimum ViT model to predict diseases (defined by "Finding Label") from the images. "Finding Label" can be any number of 14 diseases or a "No Finding" label, which indicates that no disease was detected. To be compatible with the Hugging Face library, the text labels need to be transformed to N-hot encoded arrays representing the multiple labels which are needed to classify each image. An N-hot encoded array represents the labels as a list of booleans, true if the label corresponds to the image and false if not.</p>
<p>First we identify the unique labels in the dataset.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/832eea2e60f94fb5ac6bb14f112a10ad.js"></script>
</div>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/7783093c436e570d0f7b1ed619771ae6.js"></script>
</div>
<p>Now we transform the labels into N-hot encoded arrays:</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/cf9fc70bee43b51ffd38c2046ee4380e.js"></script>
</div>
<p>When loading data using the <code>datasets.load_dataset</code> function, labels can be provided either by having folders for each of the labels (see "<a href="https://huggingface.co/docs/datasets/v2.3.2/en/image_process%22%20/l%20%22imagefolder" rel="noopener" target="_blank">ImageFolder</a>" documentation) or by having a <code>metadata.jsonl</code> file (see "<a href="https://huggingface.co/docs/datasets/v2.3.2/en/image_process%22%20/l%20%22imagefolder-with-metadata" rel="noopener" target="_blank">ImageFolder with metadata</a>" documentation). As the images in this dataset can have multiple labels, we have chosen to use a <code>metadata.jsonl file</code>. We write the image file names and their associated labels to the <code>metadata.jsonl</code> file.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/b59866219a4ec051da2e31fca6eb7e4d.js"></script>
</div>
<h2>Creating the dataset</h2>
<p>We are now ready to create the PyTorch dataset and split it into training and validation sets. This step converts the dataset to the <a href="https://arrow.apache.org/" rel="noopener" target="_blank">Arrow file format</a> which allows data to be loaded quickly during training and validation (<a href="https://huggingface.co/docs/datasets/v2.3.2/en/about_arrow" rel="noopener" target="_blank">about Arrow and Hugging Face</a>). Because the entire dataset is being loaded and pre-processed it can take a few minutes.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/6d2e26d5c1ad3df6ba966567086f8413.js"></script>
</div>
<p>We are going to import the ViT model from the checkpoint <code>google/vit-base-patch16-224-in21k</code>. The checkpoint is a standard model hosted by Hugging Face and is not managed by Graphcore.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/1df44cf80f72e1132441e539e3c3df84.js"></script>
</div>
<p>To fine-tune a pre-trained model, the new dataset must have the same properties as the original dataset used for pre-training. In Hugging Face, the original dataset information is provided in a config file loaded using the <code>AutoFeatureExtractor</code>. For this model, the X-ray images are resized to the correct resolution (224x224), converted from grayscale to RGB, and normalized across the RGB channels with a mean (0.5, 0.5, 0.5) and a standard deviation (0.5, 0.5, 0.5).</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/15c3fa337c2fd7e0b3cad23c421c3d28.js"></script>
</div>
<p>For the model to run efficiently, images need to be batched. To do this, we define the <code>vit_data_collator</code> function that returns batches of images and labels in a dictionary, following the <code>default_data_collator</code> pattern in <a href="https://huggingface.co/docs/transformers/main_classes/data_collator" rel="noopener" target="_blank">Transformers Data Collator</a>.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/a8af618ee4032b5984917ac8fe129cf5.js"></script>
</div>
<h2>Visualising the dataset</h2>
<p>To examine the dataset, we display the first 10 rows of metadata.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/f00def295657886e166e93394077d6cd.js"></script>
</div>
<p>Let's also plot some images from the validation set with their associated labels.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/20752216ae9ab314563d87cb3d6aeb94.js"></script>
</div>
<p><img src="https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=1024&amp;name=x-ray%20images%20transformed.jpg" alt="x-ray images transformed" loading="lazy" style="width: 1024px; margin-left: auto; margin-right: auto; display: block;" width="1024" srcset="https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=512&amp;name=x-ray%20images%20transformed.jpg 512w, https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=1024&amp;name=x-ray%20images%20transformed.jpg 1024w, https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=1536&amp;name=x-ray%20images%20transformed.jpg 1536w, https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=2048&amp;name=x-ray%20images%20transformed.jpg 2048w, https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=2560&amp;name=x-ray%20images%20transformed.jpg 2560w, https://www.graphcore.ai/hs-fs/hubfs/x-ray%20images%20transformed.jpg?width=3072&amp;name=x-ray%20images%20transformed.jpg 3072w" sizes="(max-width: 1024px) 100vw, 1024px"></p>
<div class="blog-caption" style="max-height: 100%; max-width: 90%; margin-left: auto; margin-right: auto; line-height: 1.4;">
<p>The images are chest X-rays with labels of lung diseases the patient was diagnosed with. Here, we show the transformed images.</p>
</div>
<p>Our dataset is now ready to be used.</p>
<h2>Preparing the model</h2>
<p>To train a model on the IPU we need to import it from Hugging Face Hub and define a trainer using the IPUTrainer class. The IPUTrainer class takes the same arguments as the original <a href="https://huggingface.co/docs/transformers/main_classes/trainer" rel="noopener" target="_blank">Transformer Trainer</a> and works in tandem with the IPUConfig object which specifies the behaviour for compilation and execution on the IPU.</p>
<p>Now we import the ViT model from Hugging Face.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/dd026fd7056bbe918f7086f42c4e58e3.js"></script>
</div>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/68664b599cfe39b633a8853364b81008.js"></script>
</div>
<p>To use this model on the IPU we need to load the IPU configuration, <code>IPUConfig</code>, which gives control to all the parameters specific to Graphcore IPUs (existing IPU configs <a href="https://huggingface.co/Graphcore" rel="noopener" target="_blank">can be found here</a>). We are going to use <code>Graphcore/vit-base-ipu</code>.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/3759d2f899ff75e61383b2cc54593179.js"></script>
</div>
<p>Let's set our training hyperparameters using <code>IPUTrainingArguments</code>. This subclasses the Hugging Face <code>TrainingArguments</code> class, adding parameters specific to the IPU and its execution characteristics.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/aaad87d4b2560cc288913b9ec85ed312.js"></script>
</div>
<h2>Implementing a custom performance metric for evaluation</h2>
<p>The performance of multi-label classification models can be assessed using the area under the ROC (receiver operating characteristic) curve (AUC_ROC). The AUC_ROC is a plot of the true positive rate (TPR) against the false positive rate (FPR) of different classes and at different threshold values. This is a commonly used performance metric for multi-label classification tasks because it is insensitive to class imbalance and easy to interpret.</p>
<p>For this dataset, the AUC_ROC represents the ability of the model to separate the different diseases. A score of 0.5 means that it is 50% likely to get the correct disease and a score of 1 means that it can perfectly separate the diseases. This metric is not available in Datasets, hence we need to implement it ourselves. HuggingFace Datasets package allows custom metric calculation through the <code>load_metric()</code> function. We define a <code>compute_metrics</code> function and expose it to Transformer’s evaluation function just like the other supported metrics through the datasets package. The <code>compute_metrics</code> function takes the labels predicted by the ViT model and computes the area under the ROC curve. The <code>compute_metrics</code> function takes an <code>EvalPrediction</code> object (a named tuple with a <code>predictions</code> and <code>label_ids</code> field), and has to return a dictionary string to float.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/1924be9dc0aeb17e301936c5566b4de2.js"></script>
</div>
<p>To train the model, we define a trainer using the <code>IPUTrainer</code> class which takes care of compiling the model to run on IPUs, and of performing training and evaluation. The <code>IPUTrainer</code> class works just like the Hugging Face Trainer class, but takes the additional <code>ipu_config</code> argument.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/0b273df36666ceb85763e3210c39d5f6.js"></script>
</div>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/c94c59a6aed6165b0519af24e168139b.js"></script>
</div>
<h2>Running the training</h2>
<p>To accelerate training we will load the last checkpoint if it exists.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/6033ce6f471af9f2136cf45002db97ab.js"></script>
</div>
<p>Now we are ready to train.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/e203649cd06809ecf52821efbbdac7f6.js"></script>
</div>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/cc5e9367cfd1f8c295d016c35b552620.js"></script>
</div>
<h2>Plotting convergence</h2>
<p>Now that we have completed the training, we can format and plot the trainer output to evaluate the training behaviour.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/05fbef22532f22c64572e9a62d9f219b.js"></script>
</div>
<p>We plot the training loss and the learning rate.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/3f124ca1d9362c51c6ebd7573019133d.js"></script>
</div>
<p><img src="https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=1024&amp;name=vit%20output.png" alt="vit output" loading="lazy" style="width: 1024px; margin-left: auto; margin-right: auto; display: block;" width="1024" srcset="https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=512&amp;name=vit%20output.png 512w, https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=1024&amp;name=vit%20output.png 1024w, https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=1536&amp;name=vit%20output.png 1536w, https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=2048&amp;name=vit%20output.png 2048w, https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=2560&amp;name=vit%20output.png 2560w, https://www.graphcore.ai/hs-fs/hubfs/vit%20output.png?width=3072&amp;name=vit%20output.png 3072w" sizes="(max-width: 1024px) 100vw, 1024px">The loss curve shows a rapid reduction in the loss at the start of training before stabilising around 0.1, showing that the model is learning. The learning rate increases through the warm-up of 25% of the training period, before following a cosine decay.</p>
<h2>Running the evaluation</h2>
<p>Now that we have trained the model, we can evaluate its ability to predict the labels of unseen data using the validation dataset.</p>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/bd946bc17558c3045662262da31890b3.js"></script>
</div>
<div style="font-size: 14px; line-height: 1.3;">
<script src="https://gist.github.com/nickmaxfield/562ceec321a9f4ac16483c11cb3694c2.js"></script>
</div>
<p>The metrics show the validation AUC_ROC score the tutorial achieves after 3 epochs.</p>
<p>There are several directions to explore to improve the accuracy of the model including longer training. The validation performance might also be improved through changing optimisers, learning rate, learning rate schedule, loss scaling, or using auto-loss scaling.</p>
<h2>Try Hugging Face Optimum on IPUs for free</h2>
<p>In this post, we have introduced ViT models and have provided a tutorial for training a Hugging Face Optimum model on the IPU using a local dataset.</p>
<p>The entire process outlined above can now be run end-to-end within minutes for free, thanks to Graphcore’s <a href="/posts/paperspace-graphcore-partner-free-ipus-developers" rel="noopener" target="_blank" style="font-weight: bold;">new partnership with Paperspace</a>. Launching today, the service will provide access to a selection of Hugging Face Optimum models powered by Graphcore IPUs within Gradient—Paperspace’s web-based Jupyter notebooks.</p>
<p><a href="https://console.paperspace.com/github/gradient-ai/Graphcore-HuggingFace?machine=Free-IPU-POD16&amp;container=graphcore%2Fpytorch-jupyter%3A2.6.0-ubuntu-20.04-20220804&amp;file=%2Fget-started%2Fwalkthrough.ipynb" rel="noopener" target="_blank"><img src="https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=200&amp;name=gradient-badge-gradient-05-d-05.png" alt="run on Gradient" loading="lazy" style="width: 200px; float: left;" width="200" srcset="https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=100&amp;name=gradient-badge-gradient-05-d-05.png 100w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=200&amp;name=gradient-badge-gradient-05-d-05.png 200w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=300&amp;name=gradient-badge-gradient-05-d-05.png 300w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=400&amp;name=gradient-badge-gradient-05-d-05.png 400w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=500&amp;name=gradient-badge-gradient-05-d-05.png 500w, https://www.graphcore.ai/hs-fs/hubfs/gradient-badge-gradient-05-d-05.png?width=600&amp;name=gradient-badge-gradient-05-d-05.png 600w" sizes="(max-width: 200px) 100vw, 200px"></a></p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>If you’re interested in trying Hugging Face Optimum with IPUs on Paperspace Gradient including ViT, BERT, RoBERTa and more, you can <a href="https://www.paperspace.com/graphcore" rel="noopener" target="_blank" style="font-weight: bold;">sign up here</a> and find a getting started guide <a href="/posts/getting-started-with-ipus-on-paperspace" rel="noopener" target="_blank" style="font-weight: bold;">here</a>.</p>
<h2>More Resources for Hugging Face Optimum on IPUs</h2>
<ul>
<li><a href="https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/vit_model_training" rel="noopener" target="_blank">ViT Optimum tutorial code on Graphcore GitHub</a></li>
<li><a href="https://huggingface.co/Graphcore" rel="noopener" target="_blank">Graphcore Hugging Face Models &amp; Datasets</a></li>
<li><a href="https://github.com/huggingface/optimum-graphcore" rel="noopener" target="_blank">Optimum Graphcore on GitHub</a></li>
</ul>
<p>This deep dive would not have been possible without extensive support, guidance, and insights from Eva Woodbridge, James Briggs, Jinchen Ge, Alexandre Payot, Thorin Farnsworth, and all others contributing from Graphcore, as well as Jeff Boudier, Julien Simon, and Michael Benayoun from Hugging Face.</p></span>
</div>
   </article>
