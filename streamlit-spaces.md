---
title: "Hosting your Models and Datasets on Hugging Face Spaces using Streamlit"
thumbnail: /blog/assets/29_streamlit-spaces/thumbnail.png

---

<h1>
    Hosting your Models and Datasets on Hugging Face Spaces using Streamlit
</h1>

<div class="blog-metadata">
    <small>Published September 28, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/streamlit-spaces.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/merve">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1631694399207-6141a88b3a0ec78603c9e784.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>merve</code>
            <span class="fullname">Merve Noyan</span>
        </div>
    </a>
</div>



## Showcase your Datasets and Models using Streamlit on Hugging Face Spaces

[Streamlit](https://streamlit.io/) allows you to visualize datasets and build demos of Machine Learning models in a neat way. In this blog post we will walk you through hosting models and datasets and serving your Streamlit applications in Hugging Face Spaces. 

## Building demos for your models

You can load any Hugging Face model and build cool UIs using Streamlit. In this particular example we will recreate ["Write with Transformer"](https://transformer.huggingface.co/doc/gpt2-large) together. It's an application that lets you write anything using transformers like GPT-2 and XLNet. 
 
![write-with-transformers](assets/29_streamlit-spaces/write-tr.png)

We will not dive deep into how the inference works. You only need to know that you need to specify some hyperparameter values for this particular application. Streamlit provides many [components](https://docs.streamlit.io/en/stable/api.html) for you to easily implement custom applications. We will use some of them to receive necessary hyperparameters inside the inference code.

- The ```.text_area``` component creates a nice area to input sentences to be completed.
- The streamlit ```.sidebar``` method enables you to take your variables in a sidebar. 
- The ```slider``` is used to take continuous values, don't forget to give a step, otherwise the ```slider``` will treat them as integers. 
- You can let the end-user input integer vaues with ```number_input``` .


```
import streamlit as st

# adding the text that will show in the text box as default
default_value = "See how a modern neural network auto-completes your text 🤗 This site, built by the Hugging Face team, lets you write a whole document directly from your browser, and you can trigger the Transformer anywhere using the Tab key. Its like having a smart machine that completes your thoughts 😀 Get started by typing a custom snippet, check out the repository, or try one of the examples. Have fun!"

sent = st.text_area("Text", default_value, height = 275)
max_length = st.sidebar.slider("Max Length", min_value = 10, max_value=30)
temperature = st.sidebar.slider("Temperature", value = 1.0, min_value = 0.0, max_value=1.0, step=0.05)
top_k = st.sidebar.slider("Top-k", min_value = 0, max_value=5, value = 0)
top_p = st.sidebar.slider("Top-p", min_value = 0.0, max_value=1.0, step = 0.05, value = 0.9)
num_return_sequences = st.sidebar.number_input('Number of Return Sequences', min_value=1, max_value=5, value=1, step=1)
```

The inference code returns the generated output, you can print the output using simple ```st.write```.
```st.write(generated_sequences[-1])```

Here's what our replicated version looks like.
![streamlit-rep](assets/29_streamlit-spaces/streamlit-rep.png)

You can checkout the full code [here](https://huggingface.co/spaces/merve/write-with-transformer).

## Showcase your Datasets and Data Visualizations

Streamlit provides many components to help you visualize datasets. It works seamlessly with 🤗 [datasets](https://huggingface.co/docs/datasets/), [pandas](https://pandas.pydata.org/docs/index.html), and visualization libraries such as [matplotlib](https://matplotlib.org/stable/index.html), [seaborn](https://seaborn.pydata.org/) and [bokeh](https://bokeh.org/). 

Let's start by loading a dataset. A new feature in `datasets`, called [streaming](https://huggingface.co/docs/datasets/dataset_streaming.html), helps you work with datasets without downloading all of them and loading into memory. This is extremely useful to work with very large datasets.

```
from datasets import load_dataset
import streamlit as st

dataset = load_dataset("merve/poetry", streaming=True)
df = pd.DataFrame.from_dict(dataset["train"])
```

 If you have a structured data like mine you can simply use  ```st.dataframe(df) ``` to show your dataset. There are many streamlit components to plot data interactively, one is ```st.barchart() ```, which I used to visualize the most used words in the poem contents. 

```
st.write("Most appearing words including stopwords")
st.bar_chart(words[0:50])
```

 If you'd like to use libraries like matplotlib, seaborn or bokeh, all you have to do is to put  ```st.pyplot() ``` at the end of your plotting script.
 
```
st.write("Number of poems for each author")
sns.catplot(x="author", data=df, kind="count", aspect = 4)
plt.xticks(rotation=90)
st.pyplot()
```

You can see the interactive bar chart, dataframe component and hosted matplotlib and seaborn visualizations below. You  can check out the code [here](https://huggingface.co/spaces/merve/streamlit-dataset-demo).

![spaces-streamlit-dataset-demo](assets/29_streamlit-spaces/streamlit-dataset-vid.gif)

## Hosting your Projects in Hugging Face Spaces

You can simply drag and drop your files as shown below. Note that you need to include your additional dependencies in the requirements.txt. Also note that the version of the streamlit you have on your local is the same, for seamless usage, refer to [Spaces API reference](https://huggingface.co/docs/hub/spaces#reference). 

![spaces-streamlit](assets/29_streamlit-spaces/streamlit.gif)

There are so many components and [packages](https://streamlit.io/components) you can use to demonstrate your models, datasets, and visualizations. You can get started [here](https://huggingface.co/spaces).