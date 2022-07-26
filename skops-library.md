---
title: "Introducing Skops"
thumbnail: /blog/assets/88_skops_library/introducing_skops.png
---

<h1>
    Introducing Skops
</h1>

<div class="blog-metadata">
    <small>Published N/A.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/skops-library.md">
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

## Introducing Skops

At Hugging Face, we are working on tackling various problems in open-source machine learning, including, hosting models securely and openly, enabling reproducibility, explainability and collaboration. We are thrilled to introduce you to our new library: Skops! With skops, you can host your sklearn models on Hugging Face Hub, create model cards with better reproducibility and collaborate with others. 

Models can be serialized for reproducibility with skops. You can later push your model to Hugging Face Hub as well.

```python
import os
import pickle
from tempfile import mkstemp, mkdtemp
from skops import hub_utils, card
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# let's save the model
_, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
with open(pkl_name, mode="bw") as f:
    pickle.dump(model, file=f)

# we will now initialize a local repository and put our model in
local_repo = mkdtemp(prefix="skops-")
hub_utils.init(
    model=pkl_name, requirements=[f"scikit-learn={sklearn.__version__}"], dst=local_repo
)
```

`init` not only saves the model, but also creates a configuration file containing the specifications of the environment model is trained. 

We will now create a model card. The content of the model card is determined by a jinja template. By default it uses [this template](https://github.com/skops-dev/skops/blob/main/skops/card/default_template.md). The default template consists of a markdown part and a metadata section. The keys to the metadata section is defined [here](https://huggingface.co/docs/hub/models-cards#model-card-metadata) and is used for discoverability of the models. 

We can add information and metadata using `add`. You can also pass a custom template's path to be used in model card using `add`.

```python
# create the card
model_card = card.Card(model)

# create some information we want to pass to the card
license = "mit"
limitations = "This model is not ready to be used in production."
model_description = (
    "This is a HistGradientBoostingClassifier model trained on breast cancer dataset."
)
model_card_authors = "skops_user"
get_started_code = (
    "import pickle\nwith open(dtc_pkl_filename, 'rb') as file:\nclf = pickle.load(file)"
)
citation_bibtex = "bibtex\n@inproceedings{...,year={2020}}"

# we can add the information using add
model_card.add(
    citation_bibtex=citation_bibtex,
    get_started_code=get_started_code,
    model_card_authors=model_card_authors,
    limitations=limitations,
    model_description=model_description,
)
```

We can also add any plot of our choice to the card using `add_plot` like below.

```python
# we will create a confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# save the plot
plt.savefig(f"{local_repo}/confusion_matrix.png")

# the plot will be written to the model card under the name confusion_matrix
# we pass the path of the plot itself
model_card.add_plot(confusion_matrix="confusion_matrix.png")
```

Let's save the model card in the local repository. 
```python
model_card.save((Path(local_repo) / "README.md"))
```

We can now push the repository to Hugging Face Hub. For this, we will use `push` from `hub_utils`.

```python
# if the repository doesn't exist remotely on the Hugging Face Hub, it will be created when we set create_remote to True
hub_utils.push(
    repo_id="skops-user/my-awesome-model",
    source=local_repo,
    token=token,
    commit_message="pushing files to the repo from the example!",
    create_remote=True,
)
```

Once we push the model to the Hub, anyone can use it, unless the repository is private. You can download the models using `download`.

```python
repo_copy = mkdtemp(prefix="skops")
hub_utils.download(repo_id="skops-user/my-awesome-model", dst=repo_copy)
print(os.listdir(repo_copy))
```

The repository also contains the model configuration as well as requirements of the environment.

```python
# We can get the requirements using get_requirements
print(hub_utils.get_requirements(path=repo_copy))
# We can also get the configuration using get_config
print(json.dumps(hub_utils.get_config(path=repo_copy), indent=2))
```

If the requirements of your project have changed, you can use `update_env` to update the environment.

```python
hub_utils.update_env(path=local_repo, requirements=["scikit-learn"])
```

We have prepared two notebooks to demonstrate how to save your models and use model card utilities, you can find them at resources section below.


## Resources
- [Model card tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_model_card.html)
- [hub_utils tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_hf_hub.html)
- [skops documentation](https://skops.readthedocs.io/en/latest/modules/classes.html)