---
title: "Visualize proteins on huggingface spaces"
thumbnail: /blog/assets/94_visualize_proteins_using_3dmoljs/thumbnail.png
---

<h1>Visualize proteins on Huggingface spaces</h1>

<div class="blog-metadata">
    <small>Published 10.08, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/pytorch-fsdp.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/duerrsimon">
        <img class="avatar avatar-user" src="hhttps://avatars.githubusercontent.com/u/16979577?s=200&v=4" title="Gravatar">
        <div class="bfc">
            <code>duerrsimon</code>
            <span class="fullname">Simon Dürr</span>
        </div>
    </a>
</div>

In this post we will look at how we can visualize proteins on Huggingface spaces.

## Motivation 🤗

Proteins have a huge impact on our life - from medicines to washing powder. Machine learning on proteins is a rapidly growing field to help us design new and interesting proteins. Proteins are complex 3D objects generally composed of a series of building blocks called amino acids that are arranged in 3D space to give the protein its function. For machine learning purposes a protein can for example be represented as coordinates, as graph or as 1D sequence of letters for use in a protein language model.

A famous ML model for proteins is AlphaFold2 which predicts the structure of a protein sequence using a multiple sequence alignment of similar proteins and a structure module. 

Since AlphaFold2 made its debut many more such models have come out such as OmegaFold, OpenFold etc. (see this [list](https://github.com/yangkky/Machine-learning-for-proteins) or this [list](https://github.com/sacdallago/folding_tools) for more). 


## Seeing is believing

The structure of a protein is an integral part to our understanding of what a protein does. Nowadays, there are a few tools available to visualize proteins directly in the browser such as [mol*](molstar.org) or [3dmol.js](https://3dmol.csb.pitt.edu/). In this post, you will learn how to integrate structure visualization into your Huggingface space using 3Dmol.js and the HTML block. 

## Prerequisites

Make sure you have the `gradio` Python package already [installed](/getting_started) and basic knowledge of Javascript/JQuery.


## Taking a Look at the Code

Let's take a look at how to create the minimal working demo of our interface before we dive into how to setup 3Dmol.js. 

We will built a simple demo app that can accept either a 4-digit PDB code  or a PDB file. Our app will then retrieve the pdb file from the RCSB Protein Databank and display it or use the uploaded file for display.

<iframe src="https://hf.space/embed/simonduerr/3dmol.js/+
" frameBorder="0" width="1400" height="690" title="Gradio app" class="p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

```python
import gradio as gr

def update(inp, file):
    # in this simple example we just retrieve the pdb file using its identifier from the RCSB or display the uploaded  file
    pdb_path = get_pdb(inp, file)
    return molecule(pdb_path) # this returns an iframe with our viewer
    

demo = gr.Blocks()

with demo:
    gr.Markdown("# PDB viewer using 3Dmol.js")
    with gr.Row():
        with gr.Box():
            inp = gr.Textbox(
                placeholder="PDB Code or upload file below", label="Input structure"
            )
            file = gr.File(file_count="single")
            btn = gr.Button("View structure")
        mol = gr.HTML()
    btn.click(fn=update, inputs=[inp, file], outputs=mol)
demo.launch()
```

`update`: This is the function that does the processing of our proteins and returns an `iframe` with the viewer

Our `get_pdb` function is also simple: 

```python
import os
def get_pdb(pdb_code="", filepath=""):
    if pdb_code is None or len(pdb_code) != 4:
        try:
            return filepath.name
        except AttributeError as e:
            return None
    else:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
```

Now, how to visualize the protein since Gradio does not have 3Dmol directly available as a block?
We use an `iframe` for this. 

Our `molecule` function which returns the `iframe` conceptually looks like this: 

```python
def molecule(input_pdb):
    mol = read_mol(input_pdb)
    # setup HTML document
    x = ("""<!DOCTYPE html><html> [..] </html>""") # do not use ' in this input
    return f"""<iframe  [..] srcdoc='{x}'></iframe>
```
This is a bit clunky to setup but is necessary because of the security rules in modern browsers. 

3Dmol.js setup is pretty easy and the documentation provides a [few examples](https://3dmol.csb.pitt.edu/). 

The `head` of our returned document needs to load 3Dmol.js (which in turn also loads JQuery). 

```html
<head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <style>
    .mol-container {
    width: 100%;
    height: 700px;
    position: relative;
    }
    .mol-container select{
        background-image:None;
    }
    </style>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
</head>
```
The styles for `.mol-container` can be used to modify the size of the molecule viewer. 

The `body` looks as follows:

```html
<body>  
    <div id="container" class="mol-container"></div>
    <script>
        let pdb = `"""
                    + mol
                    + """`  
        $(document).ready(function () {
            let element = $("#container");
            let config = { backgroundColor: "white" };
            let viewer = $3Dmol.createViewer(element, config);
            viewer.addModel(pdb, "pdb");
            viewer.getModel(0).setStyle({}, { cartoon: { colorscheme:"whiteCarbon" } });
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(0.8, 2000);
            })
    </script>
</body>
```
We use a template literal (denoted by backticks) to store our pdb file in the html document directly and then output it using 3dmol.js.

And that's it, now you can couple your favorite protein ML model to a fun and easy to use gradio app and directly visualize predicted or redesigned structures. If you are predicting properities of a structure (e.g how likely each amino acid is to bind a ligand) 3Dmol.js also allows to use custom `colorfunc` based on a property of each atom. 

You can check the [source code](https://huggingface.co/spaces/simonduerr/3dmol.js/blob/main/app.py) of the 3Dmol.js space for the full code.

For a production example, you can check the [ProteinMPNN](https://hf.space/simonduerr/ProteinMPNN) space where a user can upload a backbone, the inverse folding model ProteinMPNN predicts new optimal sequences and then one can run AlphaFold2 on all predicted sequences to verify whether they adopt the initial input backbone. Successful redesigns that qualitiatively adopt the same structure as predicted by AlphaFold2 with high pLDDT score should be tested in the lab. 

<iframe src="https://hf.space/embed/simonduerr/ProteinMPNN/+
" frameBorder="0" width="1400" height="690" title="Gradio app" class="p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

# Issues

If you encounter any issues with the integration of 3Dmol.js in Gradio/HF spaces, please open a discussion in [hf.space/simonduerr/3dmol.js](https://hf.space/simonduerr/3dmol.js/discussions).

If you have problems with 3Dmol.js configuration - you need to ask the developers, please, open a [3Dmol.js Issue](https://github.com/3dmol/3Dmol.js/issues) instead and describe your problem.

