---
title: "Introducing Skops"
thumbnail: /blog/assets/94_skops/introducing_skops.png
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
    <a href="/adrin">
        <img class="avatar avatar-user" src="https://huggingface.co/avatars/f40271d9ff5ac148aab4c512f8ae6402.svg" title="Gravatar">
        <div class="bfc">
            <code>adrin</code>
            <span class="fullname">Adrin Jalali</span>
        </div>
    </a>
    <a href="/BenjaminB">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1656685953025-62bf03d1e80cec527083cd66.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>BenjaminB</code>
            <span class="fullname">Benjamin Bossan</span>
        </div>
    </a>
</div>

## Introducing Skops

At Hugging Face, we are working on tackling various problems in open-source machine learning, including, hosting models securely and openly, enabling reproducibility, explainability and collaboration. We are thrilled to introduce you to our new library: Skops! With Skops, you can host your scikit-learn models on the Hugging Face Hub, create model cards for model documentation and collaborate with others.

Let's go through an end-to-end example: train a model first, and see step-by-step how to leverage Skops for sklearn in production.

```python
# let's import the libraries first
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the data and split
X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
model = DecisionTreeClassifier().fit(X_train, y_train)
```

You can use any model filename and serialization method, like `pickle` or `joblib`. At the moment, our backend uses `joblib` to load the model. `hub_utils.init` creates a local folder containing the model in the given path, and the configuration file containing the specifications of the environment the model is trained in. The data and the task passed to the `init` will help Hugging Face Hub enable the inference widget on the model page as well as discoverability features to find the model.

```python
from skops import hub_utils
import pickle

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

The repository now contains the serialized model and the configuration file. The configuration contains the features of the model, the requirements of the model, an example input taken from `X_test` that we've passed, name of the model file and name of the task to be solved here.

We will now create the model card. The card should match the expected Hugging Face Hub format: a markdown part and a metadata section, which is a `yaml` section at the top. The keys to the metadata section are defined [here](https://huggingface.co/docs/hub/models-cards#model-card-metadata) and are used for the discoverability of the models. 
The content of the model card is determined by a template that has a:
- yaml section on top for metadata (e.g. model license, library name and more),
- markdown section with free text and sections to be filled (e.g. simple description of the model),
Following sections are extracted by `skops` to fill in the model card:
- Hyperparameters of the model,
- Interactive diagram of the model,
- For metadata, library name and task identifier (e.g. tabular-classification), and information required by the inference widget are filled.

We will walk you through how to programmatically pass information to fill the model card. You can check out our documentation on default template provided by `skops` and it's sections [here](https://skops.readthedocs.io/en/latest/model_card.html) to see what the template expects and see how the template itself looks like [here](https://github.com/skops-dev/skops/blob/main/skops/card/default_template.md).

You can create the model card by instantiating the `Card` class from `skops`. During model serialization, the task name and library name are written to the configuration file. This information is also needed in the card's metadata, so you can use the `metadata_from_config` method to extract the metadata from the configuration file and pass it to the card when you create it. You can add information and metadata using `add`.

```python
from skops import card

# create the card 
model_card = card.Card(model, metadata=card.metadata_from_config(Path(destination_folder)))

limitations = "This model is not ready to be used in production."
model_description = "This is a DecisionTreeClassifier model trained on breast cancer dataset."
model_card_authors = "skops_user"
get_started_code = "import pickle \nwith open(dtc_pkl_filename, 'rb') as file: \n    clf = pickle.load(file)"
citation_bibtex = "bibtex\n@inproceedings{...,year={2020}}"

# we can add the information using add
model_card.add(
    citation_bibtex=citation_bibtex,
    get_started_code=get_started_code,
    model_card_authors=model_card_authors,
    limitations=limitations,
    model_description=model_description,
)

# we can set the metadata part directly
model_card.metadata.license = "mit"
```

We will now evaluate the model and add a description of the evaluation method with `add`. The metrics are added by `add_metrics`, which will be parsed into a table. 

```python
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                            accuracy_score, f1_score)
# let's make a prediction and evaluate the model
y_pred = model.predict(X_test)
# we can pass metrics using add_metrics and pass details with add
model_card.add(eval_method="The model is evaluated using test split, on accuracy and F1 score with macro average.")
model_card.add_metrics(accuracy=accuracy_score(y_test, y_pred))
model_card.add_metrics(**{"f1 score": f1_score(y_test, y_pred, average="micro")})
```

We can also add any plot of our choice to the card using `add_plot` like below.

```python
import matplotlib.pyplot as plt
from pathlib import Path
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

We can now push the repository to Hugging Face Hub. For this, we will use `push` from `hub_utils`. Hugging Face Hub requires tokens for authentication, therefore you need to pass your token. You can use `notebook_login` if you're logging in from notebook, or you can use `huggingface-cli login` if you're logging in from CLI, you can also pass your token directly.

```python
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

The inference widget is enabled to make predictions in the repository.

![Hosted Inference Widget](blog/assets/94_skops/skops_widget.png)

If the requirements of your project have changed, you can use `update_env` to update the environment.

```python
hub_utils.update_env(path=local_repo, requirements=["scikit-learn"])
```

You can see the example repository pushed with above code [here](https://huggingface.co/scikit-learn/skops-blog-example).
We have prepared two examples to show how to save your models and use model card utilities. You can find them in the resources section below.


## Resources
- [Model card tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_model_card.html)
- [hub_utils tutorial](https://skops.readthedocs.io/en/latest/auto_examples/plot_hf_hub.html)
- [skops documentation](https://skops.readthedocs.io/en/latest/modules/classes.html)