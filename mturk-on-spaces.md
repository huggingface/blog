---
title: "How to collect a dataset with crowdworkers, all from a Hugging Face space"
thumbnail: /blog/assets/91_mturk_on_spaces/thumbnail.png
---

<h1>
    How to collect a dataset with crowdworkers, all from a Hugging Face space
</h1>
abidlabs
<div class="blog-metadata">
    <small>Published July 25, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/mturk-on-spaces.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/Tristan">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648247133961-61e9e3d4e2a95338e04c9f33.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>Tristan</code>
            <span class="fullname">Tristan Thrush</span>
        </div>
    </a>
    <a href="/abidlabs">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1621947938344-noauth.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>abidlabs</code>
            <span class="fullname">Abubakar Abid</span>
        </div>
    </a>
    <a href="/douwekiela">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1641847245435-61dc997715b47073db1620dc.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>douwekiela</code>
            <span class="fullname">Douwe Kiela</span>
        </div>
    </a>
</div>


### TLDR

Do you want to gather data from crowdworkers but don’t know where to start? This tutorial is for you! Here, you will learn to automate the "crowdworker to Hugging Face dataset" pipeline, in only a few lines of code. Specifically, you will learn how to:

- Run your Hugging Face space as a data-collection interface in Mechanical Turk (a popular data collection platform)
- Make your space store data in a Hugging Face dataset automatically
- Do Dynamic Adversarial Data Collection (DADC), to improve the quality of the collected data. In DADC, there is a model providing feedback to human crowdworkers.


## Overview
In this post, we showcase a space which collects a sentiment analysis dataset. Examples in this dataset are `text, target` pairs like this:

```
text: “the movie was great”, target: “positive”
```

Following the DADC paradigm, we also deploy a Hugging Face model in the space, so it can give crowdworkers feedback about whether they are creating hard examples which fool it.

We make the UI with [Gradio](https://huggingface.co/blog/gradio-spaces). Below is the space, embedded in an iframe. You can also access the space [here](https://huggingface.co/spaces/Tristan/dadc). In the space, assignments are saved in chunks of two examples. Try creating a couple examples. They will be saved in a dataset which you will see later. Afterwards, a “Submit Work” button will appear for you. For turkers, the “Submit Work” button will send a message to Mechanical Turk to give them credit for their two-example chunk, so they can get paid. For you, the “Submit Work” button will just refresh the space. On Mechanical Turk, these quick chunks of work are called Human Intelligence Tasks (HITs).

<iframe src="https://hf.space/embed/Tristan/dadc/+?__theme=light" frameBorder="0" width="100%" height="1400px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>


## How does the space store examples into a Hugging Face dataset?

The following snippet at [this line of app.py](https://huggingface.co/spaces/Tristan/dadc/blob/main/app.py#L102) writes a person’s work in a `.jsonl` file on the Hugging Face space machine:

```
# Write the HIT data to our local dataset because the person has
# submitted everything now.
with open(DATA_FILE, "a") as jsonlfile:
    json_data_with_assignment_id =\
        [json.dumps(dict({"assignmentId": state["assignmentId"]}, **datum)) for datum in state["data"]]
    jsonlfile.write("\n".join(json_data_with_assignment_id) + "\n")
```

The following code at [this line of app.py](https://huggingface.co/spaces/Tristan/dadc/blob/main/app.py#L31) is run asynchronously by the space every minute:

```
# This function pushes the HIT data written in data.jsonl to our Hugging Face
# dataset every minute. Adjust the frequency to suit your needs.
def asynchronous_push(f_stop):
    if repo.is_repo_clean():
        print("Repo currently clean. Ignoring push_to_hub")
    else:
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Auto commit by space")
        if FORCE_PUSH == "yes":
            force_git_push(repo)
        else:
            repo.git_push()
    if not f_stop.is_set():
        # call again in 60 seconds
        threading.Timer(60, asynchronous_push, [f_stop]).start()

f_stop = threading.Event()
asynchronous_push(f_stop)
```

Every minute, this function checks the `.jsonl` file for updates. If it was updated, then the space pushes the `.jsonl` to the dataset [here](https://huggingface.co/datasets/Tristan/dadc-data/blob/main/data.jsonl). If you entered two examples following the section above, then you should see your examples stored in this dataset after a minute.


## How do we run the space on Mechanical Turk?

To run this app on Mechanical Turk, clone the space locally with:

```
git clone https://huggingface.co/spaces/Tristan/dadc
```

Then, follow the `README.md`. Essentially, all you need to do is run [`collect.py`](https://huggingface.co/spaces/Tristan/dadc/blob/main/collect.py), after providing it with your API keys for Mechanical Turk.

`collect.py` launches jobs on Mechanical Turk which embed our space in an iframe. Then, crowdworkers can enter data in the same way that you did in the sections above.


## How does the space give crowdworkers credit for their work?

When a normal Hugging Face user accesses the space, the embedded URL looks like this:

```
https://hf.space/embed/Tristan/dadc/+?__theme=light
```

But when a crowdworker on Mechanical Turk accesses the space, the embedded URL looks like this:

```
https://hf.space/embed/Tristan/dadc/+?__theme=light&assignmentId=351SEKWQSDVW7WHGFS3HLCY3M80DMX
```

Mechanical Turk appends an Assignment ID to the URL of the space. An Assignment ID is a unique identifier for a crowdworker’s chunk of work.

Our space can access URL query parameters and stores the Assignment ID in its state. If the Assignment ID is not blank (i.e., if a Turker is using the space), then the space knows that a Turker is using it and takes an additional step when the “Submit Work” button is clicked. The space sends an HTML form with the Assignment ID to Mechanical Turk. Here is the code in our space, at [this line of app.py](https://huggingface.co/spaces/Tristan/dadc/blob/main/app.py#L140)

```
post_hit_js = """
    function(state) {
        // If there is an assignmentId, then the submitter is on mturk
        // and has accepted the HIT. So, we need to submit their HIT.
        const form = document.createElement('form');
        form.action = 'https://workersandbox.mturk.com/mturk/externalSubmit';
        form.method = 'post';
        for (const key in state) {
            const hiddenField = document.createElement('input');
            hiddenField.type = 'hidden';
            hiddenField.name = key;
            hiddenField.value = state[key];
            form.appendChild(hiddenField);
        };
        document.body.appendChild(form);
        form.submit();
        return state;
    }
    """

submit_hit_button.click(
    lambda state: state,
    inputs=[state],
    outputs=[state],
    _js=post_hit_js,
)
```

In order for Mechanical Turk to notice the form submission, it must happen from the frontend of app.py. This is fine, because Gradio allows us to run javascript when a button is clicked.


## Conclusion

Now, you have learned how to create a “living” Hugging Face dataset with Dynamic Adversarial Data Collection on Mechanical Turk. You can launch a data-collection space and refresh your dataset page to see new data coming in every minute!

Link to space:
[https://huggingface.co/spaces/Tristan/dadc](https://huggingface.co/spaces/Tristan/dadc)

Link to generated dataset:
[https://huggingface.co/datasets/Tristan/dadc-data/blob/main/data.jsonl](https://huggingface.co/datasets/Tristan/dadc-data/blob/main/data.jsonl)

Questions/Suggestions/Comments? Feel free to open a discussion on the space:
[https://huggingface.co/spaces/Tristan/dadc/discussions](https://huggingface.co/spaces/Tristan/dadc/discussions)


### Ethical considerations

Note that crowdworker platforms offer very limited protections for their workers. It is up to you as the task owner to ensure that you pay crowdworkers enough and do not exploit them in any other ways. For example, see [https://arxiv.org/abs/1712.05796](https://arxiv.org/abs/1712.05796).
