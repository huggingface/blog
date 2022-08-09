---
title: "Introducing Skops"
thumbnail: /blog/assets/88_skops_library/introducing_skops.png
---

<h1>
    Introducing Skops
</h1>

<div class="blog-metadata">
    <small>Published August 15, 2022.</small>
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

At Hugging Face, we are working on tackling various problems in open-source machine learning, including, hosting models securely and openly, enabling reproducibility, explainability and collaboration. We are thrilled to introduce you to our new library: Skops! With Skops, you can host your sklearn models on the Hugging Face Hub, create model cards for model documentation and collaborate with others.

Let's go through an end-to-end example: train a model first, and see step-by-step how to leverage Skops for sklearn in production.

```python
# let's import the libraries first
import os
from pathlib import Path
import pickle
from skops import hub_utils, card
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, train_test_split

# Load the data and split
X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
param_grid = {
    "max_leaf_nodes": [5, 10, 15],
    "max_depth": [2, 5, 10],
}

model = HalvingGridSearchCV(
    estimator=HistGradientBoostingClassifier(),
    param_grid=param_grid,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)
```

We will first save our model and initialize the repository. The model name and the format is insignificant, we will save our model as `example.pkl` for simplicity. `init` creates a repository containing the model in the given path and the configuration file containing the specifications of the environment the model is trained in. The data and the task passed to the `init` will help Hugging Face Hub enable the inference widget on the model page.

```python
# let's save the model
model_path = "example.pkl"
local_repo = "my-awesome-model"
with open(model_path, mode="bw") as f:
    pickle.dump(model, file=f)

# we will now initialize a local repository
hub_utils.init(
    model=model_path, 
    requirements=[f"scikit-learn={sklearn.__version__}"], 
    dst=local_repo,
    task="tabular-classification",
    data=X_test,
)
```

The repository now contains the serialized model and the configuration file. The configuration contains the features of the model, the requirements of the model, an example input taken from `X_test` that we've passed, name of the model file and name of the task solved here.

We will now create the model card. The card should fit the format that Hugging Face Hub expects it to be: it consists of a markdown part and a metadata section. The keys to the metadata section are defined [here](https://huggingface.co/docs/hub/models-cards#model-card-metadata) and are used for discoverability of the models. 
The content of the model card is determined by a jinja template that has:
- yaml section on top for metadata (e.g. model license, library name and more),
- markdown section with free text and slots to be filled (e.g. simple description of the model),
Following sections are extracted by Skops to fill in the model card:
- Hyperparameters of the model,
- Interactive plot of the model,
- For metadata, library name and task identifier (e.g. tabular-classification) are filled.
We will walk you through how to programmatically pass information to fill the model card. Please take a look at the [default template](https://github.com/skops-dev/skops/blob/main/skops/card/default_template.md) used by Skops to see what the template expects.
You can add information and metadata using `add`.

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
    "import pickle \nwith open(dtc_pkl_filename, 'rb') as file: \nclf = pickle.load(file)"
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
plt.savefig(Path(local_repo) / "confusion_matrix.png")

# the plot will be written to the model card under the name confusion_matrix
# we pass the path of the plot itself
model_card.add_plot(confusion_matrix="confusion_matrix.png")
```

Let's save the model card in the local repository. The file name here should be `README.md` since it is what Hugging Face Hub expects.
```python
model_card.save(Path(local_repo) / "README.md")
```

We can now push the repository to Hugging Face Hub. For this, we will use `push` from `hub_utils`. Hugging Face Hub requires tokens for authentication, therefore you need to pass your token.

```python
token = os.environ["HF_HUB_TOKEN"]
# if the repository doesn't exist remotely on the Hugging Face Hub, it will be created when we set create_remote to True
repo_id = "skops-user/my-awesome-model"
hub_utils.push(
    repo_id=repo_id,
    source=local_repo,
    token=token,
    commit_message="pushing files to the repo from the example!",
    create_remote=True,
)
```

Once we push the model to the Hub, anyone can use it, unless the repository is private. You can download the models using `download`. Apart from the model file, the repository contains the model configuration and the requirements of the environment."

```python
download_repo = "downloaded-model"
hub_utils.download(repo_id=repo_id, dst=download_repo)
```

If the requirements of your project have changed, you can use `update_env` to update the environment.

```python
hub_utils.update_env(path=local_repo, requirements=["scikit-learn"])
```

We have prepared two examples to demonstrate how to save your models and use model card utilities, you can find them at resources section below.


## Resources
- [Model card tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_model_card.html)
- [hub_utils tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_hf_hub.html)
- [skops documentation](https://skops.readthedocs.io/en/latest/modules/classes.html)