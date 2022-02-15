---
title: "Image search with 🤗 datasets"
thumbnail: 
---

🤗 [`datasets`](https://huggingface.co/docs/datasets/) is a library that makes it easy to access and share datasets. It also makes it easy to process data efficiently - including working with data which doesn't fit into memory. 

When `datasets` was first launched it was more usually associated with text data. However, recently, `datasets` has added increased support for images. In particular there is now a `datasets` [feature type for images](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image). A previous [blog post](https://huggingface.co/blog/fine-tune-vit) showed how `datasets` can be used with 🤗 `transformers` to train an image classification model. In this blog post we'll see how we can combine `datasets` and a few other libraries to create an image search application. 

To start lets take a look at the image feature. We can use the wonderful [rich](https://rich.readthedocs.io/) library to poke around python objects (functions, classes etc.)

```python
from rich import inspect
from datasets.features import features
```


```python
inspect(features.Image, help=True)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080">╭──────────────────── </span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'datasets.features.image.Image'</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">&gt;</span><span style="color: #000080; text-decoration-color: #000080"> ─────────────────────╮</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #00ffff; text-decoration-color: #00ffff; font-style: italic">def </span><span style="color: #800000; text-decoration-color: #800000; font-weight: bold">Image</span><span style="font-weight: bold">(</span>id: Union<span style="font-weight: bold">[</span>str, NoneType<span style="font-weight: bold">]</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">)</span> -&gt; <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>:                              <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">Image feature to read image data from an image file.</span>                             <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">Input: The Image feature accepts as input:</span>                                       <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">- A :obj:`str`: Absolute path to the image file </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080">i.e. random access is allowed</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">)</span><span style="color: #008080; text-decoration-color: #008080">.</span> <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">- A :obj:`dict` with the keys:</span>                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">    - path: String with relative path of the image file to the archive file.</span>     <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">    - bytes: Bytes of the image file.</span>                                            <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">  This is useful for archived files with sequential access.</span>                      <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">- An :obj:`np.ndarray`: NumPy array representing an image.</span>                       <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008080; text-decoration-color: #008080">- A :obj:`PIL.Image.Image`: PIL image object.</span>                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>   <span style="color: #808000; text-decoration-color: #808000; font-style: italic">dtype</span> = <span style="color: #008000; text-decoration-color: #008000">'dict'</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>      <span style="color: #808000; text-decoration-color: #808000; font-style: italic">id</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #808000; text-decoration-color: #808000; font-style: italic">pa_type</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">╰──────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



We can see there a few different ways in which we can pass in our images. We'll come back to this in a little while. 

A really nice feature of the `datasets` library (beyond the functionality for processing data, memory mapping etc.) is that you get some nice things 'for free'. One of these is the ability to add a [`faiss`](https://github.com/facebookresearch/faiss) index to a dataset. [ `faiss`](https://github.com/facebookresearch/faiss)  is a ["library for efficient similarity search and clustering of dense vectors"](https://github.com/facebookresearch/faiss). 

The `datasets` [docs](https://huggingface.co/docs/datasets) show an [example](https://huggingface.co/docs/datasets/faiss_es.html#id1) of using a `faiss` index for text retrieval. In this post we'll see if we can do the same for images. 

## The dataset: "Digitised Books - Images identified as Embellishments. c. 1510 - c. 1900. JPG"

This is a dataset of images which have been pulled from a collection of digitised books from the British Library. These images come from books across a wide time period and from a broad range of domains. The images were extracted using information contained in the OCR output for each book. As a result it's known which book the images came from but not 
necessarily anything else about that image i.e. what it is of. 

Some attempts to help overcome this have included uploading the images to [flickr](https://www.flickr.com/photos/britishlibrary/albums). This allows people to tag the images or put them into various different categories. 

There have also been projects to tag the dataset [using machine learning](https://blogs.bl.uk/digital-scholarship/2016/11/sherlocknet-update-millions-of-tags-and-thousands-of-captions-added-to-the-bl-flickr-images.html). This work already makes it possible to search by tags but we might want a 'richer' ability to search. For this particular experiment we'll work with a subset of the collections which contain "embellishments". This dataset is a bit smaller so will be better for experimenting with. We can get the data from the British Library's data repository: [https://doi.org/10.21250/db17](https://doi.org/10.21250/db17). 


```python
wget -O dig19cbooks-embellishments.zip "https://bl.iro.bl.uk/downloads/ba1d1d12-b1bd-4a43-9696-7b29b56cdd20?locale=en"
```

We can now unzip the directory. 

```python
unzip -q dig19cbooks-embellishments.zip
```

## Install required packages

There are a few packages we'll need for building an image search interface to this collection. To start with we'll need the `datasets` library. Since we're working with images we'll also need the `pillow` library. 


```python
pip install datasets pillow
```

Now we have the data downloaded we can move to loading it into the  `datasets` library. There are various ways of doing this. To start with we can grab all of the image files from our dataset. 


```python
from pathlib import Path
```


```python
files = list(Path('embellishments/').rglob("*.jpg"))
```

Since the file path includes the year of publication for the book the image came from let's create a function to grab that.


```python
def get_parts(f:Path):
    _,year,fname =  f.parts
    return year, fname
```


## 📸 Loading the images 

The images are fairly large, since this is an experiment we'll resize them a little using the `thumbnail` method from the `pillow` library (this makes sure we keep the same aspect ratio for our images).


```python
from PIL import Image
import io
```


```python
def load_image(path):
    with open(path, 'rb') as f:
        im = Image.open(io.BytesIO(f.read()))
        im.thumbnail((224,224))
    return im 
``` 


```python
im = load_image(files[0])
im
```




    
![png](2022-01-13-image_search_files/2022-01-13-image_search_19_0.png)
    



### Where is the image 🤔
You may have noticed that the `load_image` function doesn't load the filepath into pillow directly. Often we would do `Image.open(filepath.jpg)`. This is done deliberately. If we load it this way when we inspect the resulting image you'll see that the filepath attribute is empty.


```python
#collapse_output
inspect(im)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080">╭─────────────────────── </span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'PIL.JpegImagePlugin.JpegImageFile'</span><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">&gt;</span><span style="color: #000080; text-decoration-color: #000080"> ───────────────────────╮</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008000; text-decoration-color: #008000">╭───────────────────────────────────────────────────────────────────────────────────────╮</span> <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008000; text-decoration-color: #008000">│</span> <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">PIL.JpegImagePlugin.JpegImageFile</span><span style="color: #000000; text-decoration-color: #000000"> image </span><span style="color: #808000; text-decoration-color: #808000">mode</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #800080; text-decoration-color: #800080">RGB</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">size</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #800080; text-decoration-color: #800080">20</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0x224</span><span style="color: #000000; text-decoration-color: #000000"> at </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0x7FBBB392D040</span><span style="font-weight: bold">&gt;</span>     <span style="color: #008000; text-decoration-color: #008000">│</span> <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #008000; text-decoration-color: #008000">╰───────────────────────────────────────────────────────────────────────────────────────╯</span> <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                                                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                <span style="color: #808000; text-decoration-color: #808000; font-style: italic">app</span> = <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'APP0'</span>: <span style="color: #008000; text-decoration-color: #008000">b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'</span><span style="font-weight: bold">}</span>            <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>            <span style="color: #808000; text-decoration-color: #808000; font-style: italic">applist</span> = <span style="font-weight: bold">[(</span><span style="color: #008000; text-decoration-color: #008000">'APP0'</span>, <span style="color: #008000; text-decoration-color: #008000">b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'</span><span style="font-weight: bold">)]</span>          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>               <span style="color: #808000; text-decoration-color: #808000; font-style: italic">bits</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>    <span style="color: #808000; text-decoration-color: #808000; font-style: italic">custom_mimetype</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>      <span style="color: #808000; text-decoration-color: #808000; font-style: italic">decoderconfig</span> = <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">)</span>                                                               <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>    <span style="color: #808000; text-decoration-color: #808000; font-style: italic">decodermaxblock</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65536</span>                                                                <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>      <span style="color: #808000; text-decoration-color: #808000; font-style: italic">encoderconfig</span> = <span style="font-weight: bold">(</span><span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>, <span style="color: #008000; text-decoration-color: #008000">b''</span><span style="font-weight: bold">)</span>                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>        <span style="color: #808000; text-decoration-color: #808000; font-style: italic">encoderinfo</span> = <span style="font-weight: bold">{}</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>           <span style="color: #808000; text-decoration-color: #808000; font-style: italic">filename</span> = <span style="color: #008000; text-decoration-color: #008000">''</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>             <span style="color: #808000; text-decoration-color: #808000; font-style: italic">format</span> = <span style="color: #008000; text-decoration-color: #008000">'JPEG'</span>                                                               <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span> <span style="color: #808000; text-decoration-color: #808000; font-style: italic">format_description</span> = <span style="color: #008000; text-decoration-color: #008000">'JPEG (ISO 10918)'</span>                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                 <span style="color: #808000; text-decoration-color: #808000; font-style: italic">fp</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>             <span style="color: #808000; text-decoration-color: #808000; font-style: italic">height</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">224</span>                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>         <span style="color: #808000; text-decoration-color: #808000; font-style: italic">huffman_ac</span> = <span style="font-weight: bold">{}</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>         <span style="color: #808000; text-decoration-color: #808000; font-style: italic">huffman_dc</span> = <span style="font-weight: bold">{}</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>            <span style="color: #808000; text-decoration-color: #808000; font-style: italic">icclist</span> = <span style="font-weight: bold">[]</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                 <span style="color: #808000; text-decoration-color: #808000; font-style: italic">im</span> = <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">ImagingCore</span><span style="color: #000000; text-decoration-color: #000000"> object at </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0x7fbba120dc10</span><span style="font-weight: bold">&gt;</span>                               <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>               <span style="color: #808000; text-decoration-color: #808000; font-style: italic">info</span> = <span style="font-weight: bold">{</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008000; text-decoration-color: #008000">'jfif'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">257</span>,                                                     <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008000; text-decoration-color: #008000">'jfif_version'</span>: <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">)</span>,                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008000; text-decoration-color: #008000">'jfif_unit'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008000; text-decoration-color: #008000">'jfif_density'</span>: <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">)</span>                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                      <span style="font-weight: bold">}</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>              <span style="color: #808000; text-decoration-color: #808000; font-style: italic">layer</span> = <span style="font-weight: bold">[(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">)</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">)</span>, <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">)]</span>                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>             <span style="color: #808000; text-decoration-color: #808000; font-style: italic">layers</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                <span style="color: #808000; text-decoration-color: #808000; font-style: italic">map</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>               <span style="color: #808000; text-decoration-color: #808000; font-style: italic">mode</span> = <span style="color: #008000; text-decoration-color: #008000">'RGB'</span>                                                                <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>            <span style="color: #808000; text-decoration-color: #808000; font-style: italic">palette</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>           <span style="color: #808000; text-decoration-color: #808000; font-style: italic">pyaccess</span> = <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>                                                                 <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>       <span style="color: #808000; text-decoration-color: #808000; font-style: italic">quantization</span> = <span style="font-weight: bold">{</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: <span style="font-weight: bold">[</span>                                                             <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">15</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">15</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">17</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">17</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="font-weight: bold">]</span>,                                                               <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: <span style="font-weight: bold">[</span>                                                             <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>,                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>,                                                          <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                              <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                          <span style="font-weight: bold">]</span>                                                                <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>                      <span style="font-weight: bold">}</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>           <span style="color: #808000; text-decoration-color: #808000; font-style: italic">readonly</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>                                                                    <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>               <span style="color: #808000; text-decoration-color: #808000; font-style: italic">size</span> = <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">224</span><span style="font-weight: bold">)</span>                                                           <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>               <span style="color: #808000; text-decoration-color: #808000; font-style: italic">tile</span> = <span style="font-weight: bold">[]</span>                                                                   <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">│</span>              <span style="color: #808000; text-decoration-color: #808000; font-style: italic">width</span> = <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                                                                  <span style="color: #000080; text-decoration-color: #000080">│</span>
<span style="color: #000080; text-decoration-color: #000080">╰───────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>


You can also directly see this


```python
im.filename
```


Pillow usually loads images in a lazy way i.e. it only opens them when they are needed. The filepath is used to access the image. We can see the filename attribute is present if we open it from the filepath


```python
im_file = Image.open(files[0])
im_file.filename
```




    '/Users/dvanstrien/Documents/daniel/blog/_notebooks/embellishments/1855/000811462_05_000205_1_The Pictorial History of England  being a history of the people  as well as a hi_1855.jpg'


The reason I don't want the filename attribute present here is because not only do I want to use datasets to process our images but also *store* the images. If we pass a Pillow object with the filename attribute datasets will also use this for loading the images. This is often what we'd want but we don't want this here for reasons we'll see shortly. 

### Preparing images for datasets

We can now load our images. What we'll do is is loop through all our images and then load the information for each image into a dictionary. Since we want our keys to represent the columns in our dataset, and the values of those keys to be lists of the relevant data we can use a [`defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) for this. 


```python
from collections import defaultdict

data = defaultdict(list)
```

We'll loop through our image files and append to our `defaultdict`

```python
for file in files:
    year, fname = get_parts(file)
    data['fname'].append(fname)
    data['year'].append(year)
    data['path'].append(str(file))
```

 
We can now use the load `from_dict` method to create a new dataset. 

```python
from datasets import Dataset

dataset = Dataset.from_dict(data)
```

We can look at one example to see what this looks like. 


```python
dataset[0]
```

```python
{'fname': '000811462_05_000205_1_The Pictorial History of England  being a history of the people  as well as a hi_1855.jpg',
'year': '1855',
'path': 'embellishments/1855/000811462_05_000205_1_The Pictorial History of England  being a history of the people  as well as a hi_1855.jpg'}
```


### Loading our images

At the moment our dataset has the filename and full path for each image. However, we want to have an actual image loaded into our dataset. We already have a `load_image` function. This gets us most of the way there but we might also want to add some ability to deal with image errors. The datasets library has gained increased support for handling `None` types. This includes support for `None` types for images see [pull request 3195](https://github.com/huggingface/datasets/pull/3195). 

We'll wrap our `load_image` function in a try block, catch a `Image.UnidentifiedImageError` error and return `None` if we can't load the image. 


```python
def try_load_image(filename):
    try:
        image = load_image(filename)
        if isinstance(image, Image.Image):
            return image
    except Image.UnidentifiedImageError:
        return None
```

We can now use the `map` method to load our images. We return a dictionary containing the key `img` with the value being either a `PIL.Image` or `None`. Note here we also specify `writer_batch_size=50`, this controls how often `datasets` should write to disk. Since our images are fairly large we don't want to hold too much data in memory while running `map`.

```python
%%time
dataset = dataset.map(lambda example: {"img": try_load_image(example['path'])},writer_batch_size=50)
```

Let's see what this looks like


```python
dataset
```

```python
Dataset({
    features: ['fname', 'year', 'path', 'img'],
    num_rows: 416944
    })
```


We have an image column but let's check the type of all our features:


```python
dataset.features
```

```python
{'fname': Value(dtype='string', id=None),
'year': Value(dtype='string', id=None),
'path': Value(dtype='string', id=None),
'img': Image(id=None)}
```


This is looking great already. Since we might have some `None` types for images let's get rid of these. We can use the `datasets` `filter` method to only keep examples where the image is not `None`. 


```python
dataset = dataset.filter(lambda example: example['img'] is not None)
```


Taking a look at our dataset again.

```python
dataset
```

```python
Dataset({
    features: ['fname', 'year', 'path', 'img'],
    num_rows: 416935
    })
```


You'll see we lost a few rows by doing this filtering. We should now just have images which are successfully loaded. 

If we access an example and index into the `img` column we'll see our image 😃


```python
dataset[10]['img']
```


## Push all the things to the hub! 

![Push all the things to the hub!](https://i.imgflip.com/613c0r.jpg)

One of the super awesome things about the huggingface ecosystem is the huggingface hub. We can use the hub to access models and datasets. Often this is used for sharing work with others but it can also be a useful tool for work in progress. The datasets library recently added a `push_to_hub` method that allows you to push a dataset to the hub with minimal fuss. This can be really helpful by allowing you to pass around a dataset with all the transformers etc. already done. 

When I started playing around with this feature I was also keen to see if it could be used as a way of 'bundling' everything together. This is where I noticed that if you push a dataset containing images which have been loaded in from filepaths by pillow the version on the hub won't have the images attached. If you always have the image files in the same place when you work with the dataset then this doesn't matter. If you want to have the images stored in the parquet file(s) associated with the dataset we need to load it without the filename attribute present (there might be another way of ensuring that datasets doesn't rely on the image file being on the file system -- if you of this I'd love to hear about it). 

Since we loaded our images this way when we download the dataset from the hub onto a different machine we have the images already there 🤗


For now we'll push the dataset to the hub and keep them private initially.


```python
dataset.push_to_hub('davanstrien/embellishments', private=True)
```

   

### Switching machines

At this point I've created a dataset and moved it to the huggingface hub. This means it is possible to pickup the work/dataset elsewhere. 
For the next stage of the process having access to a GPU is important. The next parts of this notebook are run on [Colab](colab.research.google.com/) instead of locally on my laptop. 

We'll need to login since the dataset is currently private. We can do this using `huggingface-cli login` or alternatively we can use `notebook_login`

```python
from huggingface_hub import notebook_login

notebook_login()
```


Once we've done this we can load our dataset


```python
from datasets import load_dataset

dataset = load_dataset("davanstrien/embellishments", use_auth_token=True)
```

## Creating embeddings 🕸

We now have a dataset with a bunch of images in it. To begin creating our image search app we need to create some embeddings for these images. There are various ways in which we can try and do this but one possible way is to use the clip models via the `sentence_transformers` library. The [clip model](https://openai.com/blog/clip/) from [OpenAI](https://openai.com/) learns a joint representation for both images and text which is very useful for what we want to do since we want to be able to input text and get back an image. We can download the model using the `SentenceTransformer` class. 


```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')
```

This model will take as input either an image or some text and return an embedding. We can use the `datasets` `map` method to encode all our images using this model. When we call map we return a dictionary with the key `embeddings` containing the embeddings returned by the embedding model. We also pass `device='cuda'` when we call model, this ensures that we're doing the encoding on the GPU.


```python
ds_with_embeddings = dataset.map(
    lambda example: {'embeddings':model.encode(example['img'],device='cuda')},
                                 batch_size=32)
```


We can "save" our work by pushing back to the 🤗 hub using `push_to_hub`.


```python
ds_with_embeddings.push_to_hub('davanstrien/embellishments', private=True)
```

If we were to move to a different machine we could grab our work again by loading it from the hub 😃. For example if we moved to a different machine (or a new colab session) we could load our dataset with the current changes using:


```python
from datasets import load_dataset

ds_with_embeddings = load_dataset("davanstrien/embellishments", use_auth_token=True)
```



We now have a new column which contains the embeddings for our images. We could manually loop through these and compare it to some input embedding, however, `datasets` has an `add_faiss_index` method which we can use instead. This uses the [faiss](https://github.com/facebookresearch/faiss) library to create an efficient index for searching embeddings. For more background on this library you can watch this YouTube video

<iframe width="560" height="315" src="https://www.youtube.com/embed/sKyvsdEv6rk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

To use the `add_faise_index` method we need to pass in the column which contain our embeddings. 

```python
ds_with_embeddings['train'].add_faiss_index(column='embeddings')
```


## Image search

We now have everything we need to create a simple image search. We can use the same model we used to encode our images to encode some input text. This will act as the prompt we try and find close examples for. Let's start with "a steam engine" 


```python
prompt = model.encode("A steam engine")
```

Now we have a prompt we can use another method from the `datasets` library `get_nearest_examples` to get images which have an embedding close to our input prompt embedding. We can pass in a number of results we want to get back. 


```python
scores, retrieved_examples = ds_with_embeddings['train'].get_nearest_examples('embeddings', prompt,k=9)
```

We can index into the first example this retrieves:


```python
retrieved_examples['img'][0]
```

![png](asssests/assests/52)

    
![png](2022-01-13-image_search_files/2022-01-13-image_search_78_0.png)
    



This isn't quite a steam engine but it's also not a completely weird result. We can plot the other results to see what was returned 


```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(20, 20))
columns = 3
for i in range(9):
    image = retrieved_examples['img'][i]
    plt.subplot(9 / columns + 1, columns, i + 1)
    plt.imshow(image)
```


    
![png](2022-01-13-image_search_files/2022-01-13-image_search_81_0.png)
    


Some of these results look pretty reasonable in relation to our input prompt. We can wrap this in a function so can more easily play around with different prompts


```python
def get_image_from_text(text_prompt, number_to_retrieve=9):
    prompt = model.encode(text_prompt)
    scores, retrieved_examples = ds_with_embeddings['train'].get_nearest_examples('embeddings', prompt,k=number_to_retrieve)
    plt.figure(figsize=(20, 20))
    columns = 3
    for i in range(9):
        image = retrieved_examples['img'][i]
        plt.title(text_prompt)
        plt.subplot(9 / columns + 1, columns, i + 1)
        plt.imshow(image)
```


```python
get_image_from_text("An illustration of the sun behind a mountain")
```


    
![png](2022-01-13-image_search_files/2022-01-13-image_search_84_0.png)
    


### Trying a bunch of prompts ✨

Now we have a function for getting a few results we can try a bunch of different prompts:

- For some of these I'll choose prompts which are a broad 'category' i.e. 'a musical instrument' or 'an animal', others are specific i.e. 'a guitar'. 

- Out of interest I also tried a boolean operator:  "An illustration of a cat or a dog". 

- Finally, something a little more abstract: "an empty abyss"


```python
prompts = ["A musical instrument", "A guitar", "An animal", "An illustration of a cat or a dog", "an empty abyss"]
```


```python
for prompt in prompts:
    get_image_from_text(prompt)
```


    
![png](2022-01-13-image_search_files/2022-01-13-image_search_87_0.png)
    



    
![png](2022-01-13-image_search_files/2022-01-13-image_search_87_1.png)
    



    
![png](2022-01-13-image_search_files/2022-01-13-image_search_87_2.png)
    



    
![png](2022-01-13-image_search_files/2022-01-13-image_search_87_3.png)
    



    
![png](2022-01-13-image_search_files/2022-01-13-image_search_87_4.png)
    


We can see these results aren't always right but they are usually some reasonable results in there. It already seems like this could be useful for searching for a the semantic content of an image in this dataset. However we might hold off on sharing this as is...

## Creating a 🤗 space? 🤷🏼

One obvious next step for this kind of project is to create a hugginface [spaces](https://huggingface.co/spaces) demo. This is what I've done for other [models](https://huggingface.co/spaces/BritishLibraryLabs/British-Library-books-genre-classifier-v2)

It was a fairly simple process to get a [Gradio app setup](https://gradio.app/) from the point we got to here. Here is a screenshot of this app.  

![](../images/spaces_image_search.png)


However, I'm a little bit vary about making this public straightaway. Looking at the model card for the CLIP model we can look at the primary intended uses:

> ### Primary intended uses 
We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of computer vision models. [source](https://huggingface.co/openai/clip-vit-base-patch32)

This is fairly close to what we are interested in here. Particularly we might be interested in how well the model deals with the kinds of images in our dataset (illustrations from mostly 19th century books). The images in our dataset are (probably) fairly different from the training data. The fact that some of the images also contain text might help CLIP since it displays some [OCR ability](https://openai.com/blog/clip/). 

However, looking at the out-of-scope use cases in the model card:

> ### Out-of-Scope Use Cases
Any deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. This is because our safety assessment demonstrated a high need for task specific testing especially given the variability of CLIP’s performance with different class taxonomies. This makes untested and unconstrained deployment of the model in any use case currently potentially harmful. [source](https://huggingface.co/openai/clip-vit-base-patch32)

suggests that 'deployment' is not a good idea. Whilst the results I got are interesting I haven't played around with the model enough yet (and haven't done anything more systematic to evaluate its performance and biases). Another additional consideration is the target dataset itself. The images are drawn from books covering a variety of subjects and time periods. There are plenty of books which represent colonial attitudes and as a result some of the images included may represent certain groups of people in a negative way. This could potentially be a bad combo with a tool which allows any arbitrary text input to be encoded as a prompt.

There may be ways around this issue but this will require a bit more thought. 

## Conclusion 

Although we don't have a nice demo to show for it, we've seen how we can use `datasets` to:

- load images into the new `Image` feature type
- 'save' our work using `push_to_hub` and use this to move data between machines/sessions
- create a `faiss` index for images that we can use to retrieve images from a text (or image) input


