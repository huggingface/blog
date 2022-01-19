# **Supercharged Searching on the 🤗 Hub**

The `huggingface_hub` library is a light-weight interface that provides a progamatic approach to exploring the hosting endpoints Hugging Face provides. Specifically: models, datasets, and spaces.

Up until now, searching on the Hub through this interface was tricky to pull of, and there were many aspects of it a user had to "just know" and get accustomed to it. 

In this article, we will be looking at a few exciting new features added to the `huggingface_hub` to help lower that bar and provide users with a friendly API to search for the models and datasets they want to use without leaving their Jupyter or Python interfaces.

> Before we begin, if you do not have the latest version of the `huggingface_hub` library on your system please run the following cell:


```
!pip install huggingface_hub -U
```

## The `AttributeDictionary`

A key foundation in how most of these new helpers work is understanding the new `AttributeDictionary` class that was introduced. It is heavily inspired and based on the [fastcore](https://fastcore.fast.ai/basics.html#AttrDict) `AttrDict` class, with some important distinctions we'll talk about later. 

The general idea of this class is we take a normal dictionary and supercharge it for *exploratory programming*, by providing tab-completion for every key in a dictionary. It also works with nested dictionaries as well!

> This class mimics how the `object` class in JavaScript works

Let's look at an example:


```
from huggingface_hub.utils.endpoint_helpers import AttributeDictionary
```


```
# Write a small dictionary
d = {"a":2, "b":"This is b", "3_a":"A number"}
# Convert it to an `AttributeDictionary`
ad = AttributeDictionary(d)
```

With this we can now call the `keys` from our dictionary as both properties *or* as a key-lookup (with tab-completion):


```
# As a normal dictionary
ad["a"]
```




    2




```
# As a property with tab-completion
ad.a
```




    2



This tab-completion aspect gets even stronger when we deal with nested `AttributeDictionary` objects:


```
d = {
    "a": 
     AttributeDictionary(
         {"first": 1, "second": 2}
         ), 
     "b": 
     AttributeDictionary(
         {"third": 3, "fourth": 4}
         )
     }
ad = AttributeDictionary(d)
```


```
# Go to `ad["a"]["first"]`
print(ad.a.first)
```

    1


As mentioned before, we expand on the ideas of `fastcore`'s `AttrDict` in a few ways:
- We can delete keys with either `del ad[key]` or `del ad.key`
- A cleaner `__repr__` is available, showing what keys support tab-completion

Let's look at that second point a little more.

In Python, properties cannot have any numbers or special characters. As a result, if an `AttributeDictionary`'s key has one, it will only be able to be indexed as a **dictionary** and *not* as an object, such as below:


```
d = {"a":2, "b":3, "3_c":4}
ad = AttributeDictionary(d)
```


```
# View the __repr__
ad
```




    Available Attributes or Keys:
     * 3_c (Key only)
     * a
     * b




We can see that `3_c` can only be accessed as a key, since it has a number in it (shown visually below):


```
# As an attribute (fails)
ad.3_c
```


      File "<ipython-input-25-b138ac5be6e2>", line 2
        ad.3_c
            ^
    SyntaxError: invalid token




```
# As a dictionary key
ad["3_c"]
```




    4



## Knowing what we can search for: `SearchArguments`

Now that we understand the `AttributeDictionary`, let's talk about one of the most important parts of this update: the `ModelSearchArguments` and `DatasetSearchArguments`!

By using the power of the `AttributeDictionary`, these two classes search through all public models hosted on the Hub, and populate a nested dictionary for us to explore. 

Each of these nested dictionaries follow the same guiding principal:

- Overall Dictionary
  - Parameter Category
    - Specific parameter item


Let's see an example:


```
from huggingface_hub import ModelSearchArguments, DatasetSearchArguments
```


```
model_args = ModelSearchArguments()
dataset_args = DatasetSearchArguments()
```

> Note: These may take a moment to run, as they have to search through all the models and datasets hosted

First we will explore the `ModelSearchArguments`:


```
model_args
```




    Available Attributes or Keys:
     * author
     * dataset
     * language
     * library
     * license
     * model_name
     * pipeline_tag




In it we find different **categories** for search parameters we may want. These correspond to how we will later pass them in for searching. 

Let's explore deeper in the `pipeline_tag`:


```
model_args.pipeline_tag
```




    Available Attributes or Keys:
     * AudioClassification
     * Audio_to_Audio
     * AutomaticSpeechRecognition
     * Conversational
     * FeatureExtraction
     * Fill_Mask
     * ImageClassification
     * ImageSegmentation
     * Image_to_Text
     * ObjectDetection
     * QuestionAnswering
     * SentenceSimilarity
     * StructuredDataClassification
     * Summarization
     * TableQuestionAnswering
     * Text2TextGeneration (Key only)
     * TextClassification
     * TextGeneration
     * Text_to_Image
     * Text_to_Speech
     * TokenClassification
     * Translation
     * VoiceActivityDetection
     * Zero_ShotClassification




Here we find every single `pipeline_tag` in existance that we can use. Finally, to see what the API would use as a query:


```
model_args.pipeline_tag.Text_to_Image
```




    'text-to-image'



With this exploratory fashion, you can now go and fine-tune what you would like to search for in an organized fashion for both Datasets and Models

Below is a quick example of doing the same with `DatasetSearchArguments`:


```
dataset_args
```




    Available Attributes or Keys:
     * author
     * benchmark
     * dataset_name
     * language_creators
     * languages
     * licenses
     * multilinguality
     * size_categories
     * task_categories
     * task_ids





```
# Searching available benchmarks
dataset_args.benchmark
```




    Available Attributes or Keys:
     * gem
     * raft
     * superb
     * test





```
# Grabbing the gem benchmark
dataset_args.benchmark.gem
```




    'benchmark:gem'



## Filters and Searching the Hub

Now that we understand the search parameters we can use, *how* do we use them? 

We've added two classes to help us with that: `ModelFilter` and `DatasetFilter`. These are two namespace classes that simply hold our arguments, but what makes them special is that the `list_models` and `list_datasets` functions (which we will see later) know how to unpack these and query the API for us while our code stays clean and readable!

Let's take a look.


```
from huggingface_hub import ModelFilter, DatasetFilter
```

For a clear understanding, we'll read its docstring below:


```
print(ModelFilter.__doc__)
```

    A class that converts human-readable model search parameters into ones compatible with
        the REST API. For all parameters capitalization does not matter.
    
        Args:
            author (:obj:`str`, `optional`):
                A string that can be used to identify models on the Hub
                by the original uploader (author or organization), such as `facebook` or `huggingface`
                Example usage:
    
                    >>> from huggingface_hub import Filter
                    >>> new_filter = ModelFilter(author_or_organization="facebook")
    
             library (:obj:`str` or :class:`List`, `optional`):
                A string or list of strings of foundational libraries models were originally trained from,
                such as pytorch, tensorflow, or allennlp
                Example usage:
    
                    >>> new_filter = ModelFilter(library="pytorch")
    
             language (:obj:`str` or :class:`List`, `optional`):
                A string or list of strings of languages, both by name
                and country code, such as "en" or "English"
                Example usage:
    
                    >>> new_filter = ModelFilter(language="french")
    
             model_name (:obj:`str`, `optional`):
                A string that contain complete or partial names for models on the Hub,
                such as "bert" or "bert-base-cased"
                Example usage:
    
                    >>> new_filter = ModelFilter(model_name="bert")
    
    
             task (:obj:`str` or :class:`List`, `optional`):
                A string or list of strings of tasks models were designed for,
                such as: "fill-mask" or "automatic-speech-recognition"
                Example usage:
    
                    >>> new_filter = ModelFilter(task="text-classification")
    
             tags (:obj:`str` or :class:`List`, `optional`):
                A string tag or a list of tags to filter models on the Hub by,
                such as `text-generation` or `spacy`. For a full list of tags do:
                    >>> from huggingface_hub import HfApi
                    >>> api = HfApi()
                    # To list model tags
                    >>> api.get_model_tags()
                    # To list dataset tags
                    >>> api.get_dataset_tags()
    
                Example usage:
                    >>> new_filter = ModelFilter(tags="benchmark:raft")
    
            trained_dataset (:obj:`str` or :class:`List`, `optional`):
                A string tag or a list of string tags of the trained dataset for a model on the Hub.
                Example usage:
                    >>> new_filter = ModelFilter(trained_dataset="common_voice")
    
        


As you can imagine, it is quite easy to take our `ModelSearchArguments` (or `DatasetSearchArguments`) and then utilize them inside of our `ModelFilter` (or `DatasetFilter`)!

> Remember: Since they are just strings, you can always just pass the string in directly if you know it!

Let's use the same example provided in the official [documentation](https://huggingface.co/docs/hub/searching-the-hub#searching-for-a-model) to search the Hub for a particular model.

We'll set our query as:
- I want all models for "Text Classification"
- They should be trained on the "GLUE" dataset
- They should be compatible with PyTorch

Let's format our `ModelFilter` accordingly:


```
filt = ModelFilter(
    task = model_args.pipeline_tag.TextClassification,
    trained_dataset = model_args.dataset.glue,
    library = model_args.library.PyTorch
)
```

Another way of writing this without the `model_args` would be like so:


```
filt = ModelFilter(
    task = "text-classification",
    trained_dataset = "glue", # or dataset:glue
    library = "pytorch"
)
```

Finally, let's build a `HfApi` and search the Hub!


```
from huggingface_hub import HfApi
```


```
api = HfApi()
```


```
api.list_models(filter=filt)[-1]
```




    ModelInfo: {
    	modelId: harithapliyal/distilbert-base-uncased-finetuned-cola
    	sha: 8d5a07a64338385fe0a732a62ec820495aa6b34e
    	lastModified: 2022-01-18T18:44:28.000Z
    	tags: ['pytorch', 'tensorboard', 'distilbert', 'text-classification', 'dataset:glue', 'transformers', 'license:apache-2.0', 'generated_from_trainer', 'model-index', 'infinity_compatible']
    	pipeline_tag: text-classification
    	siblings: [ModelFile(rfilename='.gitattributes'), ModelFile(rfilename='.gitignore'), ModelFile(rfilename='README.md'), ModelFile(rfilename='config.json'), ModelFile(rfilename='pytorch_model.bin'), ModelFile(rfilename='special_tokens_map.json'), ModelFile(rfilename='tokenizer.json'), ModelFile(rfilename='tokenizer_config.json'), ModelFile(rfilename='training_args.bin'), ModelFile(rfilename='vocab.txt'), ModelFile(rfilename='runs/Jan18_14-16-18_f5e821c8415e/events.out.tfevents.1642515510.f5e821c8415e.60.0'), ModelFile(rfilename='runs/Jan18_14-16-18_f5e821c8415e/events.out.tfevents.1642521209.f5e821c8415e.60.2'), ModelFile(rfilename='runs/Jan18_14-16-18_f5e821c8415e/1642515510.2724547/events.out.tfevents.1642515510.f5e821c8415e.60.1'), ModelFile(rfilename='runs/Jan18_17-09-35_add2990e9a92/events.out.tfevents.1642525942.add2990e9a92.61.0'), ModelFile(rfilename='runs/Jan18_17-09-35_add2990e9a92/events.out.tfevents.1642531259.add2990e9a92.61.2'), ModelFile(rfilename='runs/Jan18_17-09-35_add2990e9a92/1642525942.325853/events.out.tfevents.1642525942.add2990e9a92.61.1')]
    	config: None
    	private: False
    	downloads: 0
    	library_name: transformers
    	likes: 0
    }



If we look at `distilbert-base-uncased-finetuned-cola` as our example, it matches all of our queries for what we wanted!

Where this API really comes in handy is handling very complex queries, such as:
- All models for Text Classification
- That are both for PyTorch and TensorFlow
- Were trained on the "SST-2" dataset


```
filt = ModelFilter(
    task = model_args.pipeline_tag.TextClassification,
    library = [model_args.library.PyTorch, model_args.library.TensorFlow],
    trained_dataset = model_args.dataset.sst_2
)
```


```
api.list_models(filter=filt)[0]
```




    ModelInfo: {
    	modelId: distilbert-base-uncased-finetuned-sst-2-english
    	sha: 03b4d196c19d0a73c7e0322684e97db1ec397613
    	lastModified: 2021-02-09T07:59:22.000Z
    	tags: ['pytorch', 'tf', 'rust', 'distilbert', 'text-classification', 'en', 'dataset:sst-2', 'transformers', 'license:apache-2.0', 'infinity_compatible']
    	pipeline_tag: text-classification
    	siblings: [ModelFile(rfilename='.gitattributes'), ModelFile(rfilename='README.md'), ModelFile(rfilename='config.json'), ModelFile(rfilename='pytorch_model.bin'), ModelFile(rfilename='rust_model.ot'), ModelFile(rfilename='tf_model.h5'), ModelFile(rfilename='tokenizer_config.json'), ModelFile(rfilename='vocab.txt')]
    	config: None
    	private: False
    	downloads: 2858092
    	library_name: transformers
    	likes: 29
    }



And it finds the exact model we want, without having to get *too* complex with our setup!

This is done exactly in the same fashion for datasets as well. Below is a quick example of finding all English datasets for text classification:


```
filt = DatasetFilter(
    task_categories = "text-classification",
    languages = "en"
)
```


```
api.list_datasets(filt)[0]
```




    DatasetInfo: {
    	id: Abirate/english_quotes
    	lastModified: None
    	tags: ['annotations_creators:expert-generated', 'language_creators:expert-generated', 'language_creators:crowdsourced', 'languages:en', 'multilinguality:monolingual', 'source_datasets:original', 'task_categories:text-classification', 'task_ids:multi-label-classification']
    	private: False
    	author: Abirate
    	description: None
    	citation: None
    	cardData: None
    	siblings: None
    	gated: False
    	downloads: 5
    }



With these new supercharged searching capabilities, you now don't have to even leave your coding interface to go find the right model or dataset for your task!
