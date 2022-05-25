# Introducing pull requests and discussions

![Pull requests and discussions on the hub](https://res.cloudinary.com/picturesbase/image/upload/q_auto/v1653479595/Frame_30_mmccj0.jpg)

We are thrilled to announce the release of pull requests and discussions on the Hugging Face Hub!

Pull requests and discussions are available today under the [community tab](https://huggingface.co/gpt2/discussions) and for all repository types: models, datasets, and Spaces. Any member of the community can create and participate in discussions and pull requests.

It's the biggest update ever done to the Hub, and we can't wait to see the community members start collaborating with it  🤩.

The new "Community" tab also aligns with proposals in ethical AI throughout the years. Feedback and iterations have a central place in the development of ethical machine learning software. We really believe having it in the community's toolset will unlock new kinds of positive patterns in ML, collaborations, and progress.

## About discussions

![Discussions on the Hugging Face Hub](https://res.cloudinary.com/picturesbase/image/upload/v1653480977/image_9_ccvwj3.jpg)

[Discussions](https://huggingface.co/gpt2/discussions?type=discussion) let community members ask and answer questions as well as share their ideas and suggestions directly with the repository owners and the community. Anyone can create and participate in discussions in the community tab of a repository.

## About pull requests

![Pull requests on the Hugging Face Hub](https://res.cloudinary.com/picturesbase/image/upload/v1653480977/image_10_hdy9kv.jpg)

[Pull requests](https://huggingface.co/gpt2/discussions?type=pull_request) let community members open, comment, merge, or close pull requests directly from the website.

The easiest way to open a pull request is to use the "Collaborate" button in the "Files and versions" tab. It will let you do single file contributions very easily.

Under the hood, our Pull requests do not use forks and branches, but instead, custom "branches" called `refs` that are stored directly on the source repo. This approach to avoids the need to create a forks for each new version of the model/dataset.

## How is this different from other git hosts

At a high level, we aim to build a simpler version of other git hosts' (like GitHub's) PRs and Issues:
- no forks are involved, contributors push to a special `ref` branch directly on the source repo
- there's no hard distinction between issues and PRs they are essentially the same so we display them in the same lists
- they are streamlined for ML (i.e. models/datasets/spaces repos), not arbitrary repos

## What's next

Of course, it's only the beginning. We will listen to the community feedback to add new features and improve the community tab in the future. Today is the best time to open your first discussion/PR 🤗.
