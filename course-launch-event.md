---
title: "Course Launch Community Event"
thumbnail: /blog/assets/34_course_launch/speakers_day1_thumb.png
---

<h1>
    Course Launch Community Event
</h1>

<div class="author-card">
    <a href="/sgugger">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1593126474392-5ef50182b71947201082a4e5.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>sgugger</code>
            <span class="fullname">Sylvain Gugger</span>
        </div>
    </a>
</div>

We are excited to share that after a lot of work from the Hugging Face team, part 2 of the [Hugging Face Course](https://hf.co/course) will be released on November 15th! Part 1 focused on teaching you how to use a pretrained model, fine-tune it on a text classification task then upload the result to the [Model Hub](https://hf.co/models). Part 2 will focus on all the other common NLP tasks: token classification, language modeling (causal and masked), translation, summarization and question answering. It will also take a deeper dive in the whole Hugging Face ecosystem, in particular [🤗 Datasets](https://github.com/huggingface/datasets) and [🤗 Tokenizers](https://github.com/huggingface/tokenizers).

To go with this release, we are organizing a large community event to which you are invited! The program includes two days of talks, then team projects focused on fine-tuning a model on any NLP task ending with live demos like [this one](https://huggingface.co/spaces/flax-community/chef-transformer). Those demos will go nicely in your portfolio if you are looking for a new job in Machine Learning. We will also deliver a certificate of completion to all the participants that achieve building one of them.

AWS is sponsoring this event by offering free compute to participants via [Amazon SageMaker](https://aws.amazon.com/sagemaker/). 

<div class="flex justify-center">
<img src="/blog/assets/34_course_launch/amazon_logo_dark.png" width=30% class="hidden dark:block"> 
<img src="/blog/assets/34_course_launch/amazon_logo_white.png" width=30% class="dark:hidden">
</div>

To register, please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSd17_u-wMCdO4fcOPOSMLKcJhuIcevJaOT8Y83Gs-H6KFF5ew/viewform). You will find below more details on the two days of talks.

## Day 1 (November 15th): A high-level view of Transformers and how to train them

The first day of talks will focus on a high-level presentation of Transformers models and the tools we can use to train or fine-tune them.

<div
    class="container md:grid md:grid-cols-2 gap-2 max-w-7xl"
>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/thom_wolf.png" width=50% style="border-radius: 50%;">
        <p><strong>Thomas Wolf: <em>Transfer Learning and the birth of the Transformers library</em></strong></p>
        <p>Thomas Wolf is co-founder and Chief Science Officer of HuggingFace. The tools created by Thomas Wolf and the Hugging Face team are used across more than 5,000 research organisations including Facebook Artificial Intelligence Research, Google Research, DeepMind, Amazon Research, Apple, the Allen Institute for Artificial Intelligence as well as most university departments. Thomas Wolf is the initiator and senior chair of the largest research collaboration that has ever existed in Artificial Intelligence: <a href="https://bigscience.huggingface.co">“BigScience”</a>, as well as a set of widely used <a href="https://github.com/huggingface/">libraries and tools</a>. Thomas Wolf is also a prolific educator and a thought leader in the field of Artificial Intelligence and Natural Language Processing, a regular invited speaker to conferences all around the world (<a href="https://thomwolf.io">https://thomwolf.io</a>).</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/meg_mitchell.png" width=50% style="border-radius: 50%;">
        <p><strong>Margaret Mitchell: <em>On Values in ML Development</em></strong></p>
        <p>Margaret Mitchell is a researcher working on Ethical AI, currently focused on the ins and outs of ethics-informed AI development in tech. She has published over 50 papers on natural language generation, assistive technology, computer vision, and AI ethics, and holds multiple patents in the areas of conversation generation and sentiment classification. She previously worked at Google AI as a Staff Research Scientist, where she founded and co-led Google&#39;s Ethical AI group, focused on foundational AI ethics research and operationalizing AI ethics Google-internally. Before joining Google, she was a researcher at Microsoft Research, focused on computer vision-to-language generation; and was a postdoc at Johns Hopkins, focused on Bayesian modeling and information extraction. She holds a PhD in Computer Science from the University of Aberdeen and a Master&#39;s in computational linguistics from the University of Washington. While earning her degrees, she also worked from 2005-2012 on machine learning, neurological disorders, and assistive technology at Oregon Health and Science University. She has spearheaded a number of workshops and initiatives at the intersections of diversity, inclusion, computer science, and ethics. Her work has received awards from Secretary of Defense Ash Carter and the American Foundation for the Blind, and has been implemented by multiple technology companies. She likes gardening, dogs, and cats.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/jakob_uszkoreit.png" width=50% style="border-radius: 50%;">
        <p><strong>Jakob Uszkoreit: <em>It Ain&#39;t Broke So <del>Don&#39;t Fix</del> Let&#39;s Break It</em></strong></p>
        <p>
            <!-- Feel free to add more description -->
        </p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/jay_alammar.png" width=50% style="border-radius: 50%;">
        <p><strong>Jay Alammar: <em>A gentle visual intro to Transformers models</em></strong></p>
        <p>Jay Alammar, Cohere. Through his popular ML blog, Jay has helped millions of researchers and engineers visually understand machine learning tools and concepts from the basic (ending up in numPy, pandas docs) to the cutting-edge (Transformers, BERT, GPT-3).</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/matthew_watson.png" width=50% style="border-radius: 50%;">
        <p><strong>Matthew Watson: <em>NLP workflows with Keras</em></strong></p>
        <p>Matthew Watson is a machine learning engineer on the Keras team, with a focus on high-level modeling APIs. He studied Computer Graphics during undergrad and a Masters at Stanford University. An almost English major who turned towards computer science, he is passionate about working across disciplines and making NLP accessible to a wider audience.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/chen_qian.png" width=50% style="border-radius: 50%;">
        <p><strong>Chen Qian: <em>NLP workflows with Keras</em></strong></p>
        <p>Chen Qian is a software engineer from Keras team, with a focus on high-level modeling APIs. Chen got a Master degree of Electrical Engineering from Stanford University, and he is especially interested in simplifying code implementations of ML tasks and large-scale ML.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/mark_saroufim.png" width=50% style="border-radius: 50%;">
        <p><strong>Mark Saroufim: <em>How to Train a Model with Pytorch</em></strong></p>
        <p>Mark Saroufim is a Partner Engineer at Pytorch working on OSS production tools including TorchServe and Pytorch Enterprise. In his past lives, Mark was an Applied Scientist and Product Manager at Graphcore, <a href="http://yuri.ai/">yuri.ai</a>, Microsoft and NASA&#39;s JPL. His primary passion is to make programming more fun.</p>
    </div>
</div>

## Day 2 (November 16th): The tools you will use

Day 2 will be focused on talks by the Hugging Face, [Gradio](https://www.gradio.app/), and [AWS](https://aws.amazon.com/) teams, showing you the tools you will use.

<div
    class="container md:grid md:grid-cols-2 gap-2 max-w-7xl"
>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/lewis_tunstall.png" width=50% style="border-radius: 50%;">
        <p><strong>Lewis Tunstall: <em>Simple Training with the 🤗 Transformers Trainer</em></strong></p>
        <p>Lewis is a machine learning engineer at Hugging Face, focused on developing open-source tools and making them accessible to the wider community. He is also a co-author of an upcoming O’Reilly book on Transformers and you can follow him on Twitter (@_lewtun) for NLP tips and tricks!</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/matthew_carrigan.png" width=50% style="border-radius: 50%;">
        <p><strong>Matthew Carrigan: <em>New TensorFlow Features for 🤗 Transformers and 🤗 Datasets</em></strong></p>
        <p>Matt is responsible for TensorFlow maintenance at Transformers, and will eventually lead a coup against the incumbent PyTorch faction which will likely be co-ordinated via his Twitter account @carrigmat.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/lysandre_debut.png" width=50% style="border-radius: 50%;">
        <p><strong>Lysandre Debut: <em>The Hugging Face Hub as a means to collaborate on and share Machine Learning projects</em></strong></p>
        <p>Lysandre is a Machine Learning Engineer at Hugging Face where he is involved in many open source projects. His aim is to make Machine Learning accessible to everyone by developing powerful tools with a very simple API.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/sylvain_gugger.png" width=50% style="border-radius: 50%;">
        <p><strong>Sylvain Gugger: <em>Supercharge your PyTorch training loop with 🤗 Accelerate</em></strong></p>
        <p>Sylvain is a Research Engineer at Hugging Face and one of the core maintainers of 🤗 Transformers and the developer behind 🤗 Accelerate. He likes making model training more accessible.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/lucile_saulnier.png" width=50% style="border-radius: 50%;">
        <p><strong>Lucile Saulnier: <em>Get your own tokenizer with 🤗 Transformers & 🤗 Tokenizers</em></strong></p>
        <p>Lucile is a machine learning engineer at Hugging Face, developing and supporting the use of open source tools. She is also actively involved in many research projects in the field of Natural Language Processing such as collaborative training and BigScience.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/merve_noyan.png" width=50% style="border-radius: 50%;">
        <p><strong>Merve Noyan: <em>Showcase your model demos with 🤗 Spaces</em></strong></p>
        <p>Merve is a developer advocate at Hugging Face, working on developing tools and building content around them to democratize machine learning for everyone.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/abubakar_abid.png" width=50% style="border-radius: 50%;">
        <p><strong>Abubakar Abid: <em>Building Machine Learning Applications Fast</em></strong></p>
        <p>Abubakar Abid is the CEO of <a href="www.gradio.app">Gradio</a>. He received his Bachelor&#39;s of Science in Electrical Engineering and Computer Science from MIT in 2015, and his PhD in Applied Machine Learning from Stanford in 2021. In his role as the CEO of Gradio, Abubakar works on making machine learning models easier to demo, debug, and deploy.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/mathieu_desve.png" width=50% style="border-radius: 50%;">
        <p><strong>Mathieu Desvé: <em>AWS ML Vision: Making Machine Learning Accessible to all Customers</em></strong></p>
        <p>Technology enthusiast, maker on my free time. I like challenges and solving problem of clients and users, and work with talented people to learn every day. Since 2004, I work in multiple positions switching from frontend, backend, infrastructure, operations and managements. Try to solve commons technical and managerial issues in agile manner.</p>
    </div>
    <div class="text-center flex flex-col items-center">
        <img src="/blog/assets/34_course_launch/philipp_schmid.png" width=50% style="border-radius: 50%;">
        <p><strong>Philipp Schmid: <em>Managed Training with Amazon SageMaker and 🤗 Transformers</em></strong></p>
        <p>Philipp Schmid is a Machine Learning Engineer and Tech Lead at Hugging Face, where he leads the collaboration with the Amazon SageMaker team. He is passionate about democratizing and productionizing cutting-edge NLP models and improving the ease of use for Deep Learning.</p>
    </div>
</div>
