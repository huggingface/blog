---
title: "Synthetic data: saving money and carbon with open source" 
thumbnail: blog/assets/176_synthetic-data-save-costs/synthetic_data_thumbnail.svg
authors:
- user: MoritzLaurer
---


# Synthetic data: save money, time and carbon with open source

## tl;dr

Should you fine-tune your own model or use an LLM API? Creating your own model puts you in full control, but requires expertise in data collection, training and deployment. LLM APIs are much easier to use, but force you to send your data to a third party and create costly dependencies on LLM providers. This blog post shows how you can combine the convenience of LLMs with the control and efficiency of customized models.

In a case-study on identifying investor sentiment in news, we shows how you can use an open-source LLM to create synthetic data to train your own customized model in a few steps. Our resulting custom RoBERTa model can analyze a large news corpus for around $2.7 compared to $3061 with GPT4; emits around 0.12 kg CO2 compared to 865 kg CO2 with GPT4; with a latency of 0.13 seconds compared to often multiple seconds with GPT4; while performing on-par with GPT4 at identifying investor sentiment (both 94% accuracy and 0.94 F1 macro).

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/table_pros_cons.png?raw=true" alt="table_pros_cons" width=85%>
</p>


## 1. The problem: There is no data for your use-case

Imagine your boss asks you to build a sentiment analysis system for your company. You will find 100 000+ datasets on the Hugging Face hub, 480~ of which have the word “sentiment” in the title, covering sentiment on Twitter, in poems or in Hebrew. This is great, but if, for example, you work in a financial institution and you need to track sentiment towards the specific brands in your portfolio, none of these datasets are useful for your task. With the millions of tasks companies could tackle with machine learning, it’s unlikely that someone already collected and published data on the same use-case your company is trying to solve. 

Given this lack of task-specific datasets and models, many people turn to general purpose LLMs. These models are so large and general that they can tackle most tasks with impressive accuracy out of the box. Their easy-to-use APIs eliminate the need for expertise in fine-tuning and deployment. Their main disadvantages are size and control: with hundreds of billions or trillions of parameters these models are incredibly inefficient and only run on compute clusters controlled by a few companies.

## 2. The solution: synthetic data to teach efficient students

In 2023, one development fundamentally changed the machine learning landscape: LLMs started reaching parity with human data annotators. There is now ample evidence showing that the best LLMs outperform crowd workers and are reaching parity with experts in creating quality (synthetic) data (e.g. [Zheng et al. 2023](https://arxiv.org/pdf/2306.05685.pdf), [Gilardi et al. 2023](https://arxiv.org/pdf/2303.15056.pdf), [He et al. 2023](https://arxiv.org/pdf/2303.16854.pdf)). It is hard to overstate the importance of this development. The key bottleneck for creating tailored models was the money, time and expertise required to recruit and coordinate human workers to create tailored training data. With LLMs starting to reach human parity, high-quality annotation labour is now available through APIs, reproducible annotation instructions can be sent as prompts, and synthetic data is returned almost instantaneously with compute as the only bottleneck.

In 2024, this approach will become commercially viable and boost the value of open-source for small and large businesses. For most of 2023, commercial use of LLMs for annotation labour was blocked due to restrictive business terms by LLM API providers. With models like [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) by [Mistral](https://mistral.ai/), LLM annotation labour and synthetic data now becomes open for commercial use. [Mixtral performs on-par with GPT3.5](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) and thanks to its Apache 2.0 license its synthetic data outputs can be used as training data for smaller, specialized models (the “students”) for commercial use-cases. This blog post provides an example of how this will significantly speed up the creation of your own tailored models while drastically reducing long-term inference costs.

## 3. Case-study: monitoring financial sentiment

Imagine you are a developer in a large investment firm tasked with monitoring economic news sentiment towards companies in your investment portfolio. Until recently, you had two main options: (1) First, you could fine-tune your own model. This requires writing annotation instructions, creating an annotation interface, recruiting (crowd) workers, introducing quality assurance measures to handle low-quality data, fine-tuning a model on this data and deploying it. (2) Second, you could send your data with instructions to an LLM API. You skip fine-tuning and deployment entirely, and you reduce the data creation process to writing instructions (a prompt) and sending them to an “LLM annotator” behind an API. Although option 2 is more expensive at inference time and requires you to send sensitive data to a third party, it is significantly easier to set up than option 1 and therefore used by many developers. 

In 2024, synthetic data now provides a third option, combining the cost benefits of option 1 with the ease-of-use of option 2. Simply put: you can use an LLM (the “teacher”) to annotate a small sample of data for you, and then you fine-tune a smaller, more efficient LM (the “student”) on this data. This approach can be implemented in a few simple steps.

### 3.1 Prompt an LLM to annotate your data

We use the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) sentiment dataset as a running example, but you can adapt the code for any other use-case. The financial_phrasebank task is a 3-class classification task, where 16 experts annotated sentences from financial news on Finnish companies as “positive” / “negative” / “neutral” from an investor perspective ([Malo et al. 2013](https://arxiv.org/pdf/1307.5336.pdf)). For example, the dataset contains the sentence “For the last quarter of 2010, Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier”, which was categorized as “positive” from an investor perspective by annotators. We start by downloading the dataset with its expert annotations.

```python
from datasets import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree", split='train')

# create a new column with the numeric label verbalised as label_text (e.g. "positive" instead of "0")
label_map = {i: label_text for i, label_text in enumerate(dataset.features["label"].names)}

def add_label_text(example):
    example["label_text"] = label_map[example["label"]]
    return example

dataset = dataset.map(add_label_text)

print(dataset)
# Dataset({
#    features: ['sentence', 'label', 'label_text'],
#    num_rows: 2264
#})
```

We can now write a short annotation instruction tailored to the `financial_phrasebank` task and format it as an LLM prompt. This prompt is analogous to the instructions you normally provide to crowd workers.

```python
prompt_financial_sentiment = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to analyze the sentiment in the TEXT below from an investor perspective and label it with only one the three labels:
positive, negative, or neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company. 

Do not provide any explanations and only respond with one of the labels as one word: negative, positive, or neutral

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
Label: positive
Text: The company generated net sales of 11.3 million euro this year.
Label: neutral
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
Label: negative

Your TEXT to analyse:
TEXT: {text}
Label: """
```

This annotation instruction can now be passed to an LLM API. For this example, we use `Mixtral-8x7B-Instruct-v0.1` via the free HF [serverless Inference Endpoints API](https://huggingface.co/docs/api-inference/index). The API is ideal for testing popular models. We define a simple `generate_text` function for sending our prompt and data to the API. 

```python
import os
import requests

# Choose your LLM annotator
# find available LLMs available via the free API via:  client.list_deployed_models("text-generation-inference")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

# your huggingface token for using the LLM API. Create a HF account to get your token.
HF_TOKEN = os.environ["HF_TOKEN"]  # note that writing your token in scripts is not good practice. 

# docs on different parameters: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
generation_params = dict(
    top_p=0.90,
    temperature=0.8,
    max_new_tokens=128,
		return_full_text=False,
		use_cache=False
)

def generate_text(prompt=None, generation_params=None):
    payload = {
        "inputs": prompt, 
        "parameters": {**generation_params}
    }
    response = requests.post(
				API_URL, 
				headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
				json=payload
		)
    return response.json()[0]["generated_text"]
```

As the LLM might not always return the labels in exactly the same harmonized format, we also define a short `clean_output` function, which maps the string output from the LLM to our three possible labels.  

```python
labels = ["positive", "negative", "neutral"]

def clean_output(string, random_choice=True):
    for category in labels:
        if category.lower() in string.lower():
            return category
    # if the output string cannot be mapped to one of the categories, we either return "FAIL" or choose a random label
    if random_choice:
        return random.choice(labels)
    else:
        return "FAIL"
```

We can now send our texts to the LLM for annotation. The code below sends each text to the LLM API and maps the text output from the LLM to our three clean categories. Note: iterating over each text and sending them to an API separately is inefficient in practice. APIs can usually process multiple texts at the same time and you can significantly speed up your API calls by sending batches of text to the API asynchronously. You can find optimized code in the [reproduction repository](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) of this blog post. 

```python
output_simple = []
for text in dataset["sentence"]:
		# add text into the prompt template
    prompt_formatted = prompt_financial_sentiment.format(text=text)
    # send text to API
    output = generate_text(
        prompt=prompt_formatted, generation_params=generation_params
    )
		# clean output
    output_cl = clean_output(output, random_choice=True)
    output_simple.append(output_cl)
```

Based on this output, we can now calculate metrics to see how accurately the model did the task without being trained on the task.

```python
from sklearn.metrics import classification_report

def compute_metrics(label_experts, label_pred):
		# classification report gives us both aggregate and per-class metrics 
    metrics_report = classification_report(
        label_experts, label_pred, digits=2, output_dict=True, zero_division='warn'
    )
    return metrics_report

label_experts = dataset["label_text"]
label_pred = output_simple

metrics = compute_metrics(label_experts, label_pred)
```

Based on the simple prompt, the LLM correctly classified 91.6% of texts (0.916 accuracy and 0.916 F1 macro). That’s pretty good, given that it was not trained to do this specific task. 

We can further improve this by using two simple prompting techniques: Chain-of-Thought (CoT) and Self-Consistency (SC). CoT asks the model to first reason about the correct label and then take the labeling decision, instead of immediately deciding on the correct label. SC simply means sending the same prompt with the same text to the same LLM multiple times. SC effectively gives the LLM multiple attempts per texts with different reasoning paths and if the LLM then responds “positive” twice and “neutral” once, we choose the majority (”positive”) as the correct label. Here is our updated prompt for CoT and SC:

```python
prompt_financial_sentiment_cot = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to briefly analyze the sentiment in the TEXT below from an investor perspective and then label it with only one the three labels:
positive, negative, neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company. 

You first reason step by step about the correct label and then return your label.

You ALWAYS respond only in the following JSON format: {{"reason": "...", "label": "..."}}
You only respond with one single JSON response. 

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
JSON response: {{"reason": "An increase in operating profit is positive for investors", "label": "positive"}}
Text: The company generated net sales of 11.3 million euro this year.
JSON response: {{"reason": "The text only mentions financials without indication if they are better or worse than before", "label": "neutral"}}
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
JSON response: {{"reason": "A decrease in profit is negative for investors", "label": "negative"}}

Your TEXT to analyse:
TEXT: {text}
JSON response: """
```

This is a JSON prompt, where we ask the LLM to return a structured JSON string with its “reason” as one key and the “label” as another key. The main advantage of JSON is that we can parse it to a Python dictionary and then extract the “label”. We can also extract the “reason” if we want to understand the reasoning why the LLM chose this label.

The `process_output_cot` function parses the JSON string returned by the LLM and, in case the LLM does not return valid JSON, it tries to identify the label with a simple string match from our `clean_output` function defined above.

```python
import ast 

def process_output_cot(output):
    try: 
        output_dic = ast.literal_eval(output) 
        return output_dic
    except Exception as e:
        # if json/dict parse fails, do simple search for occurance of first label term
        print(f"Parsing failed for output: {output}, Error: {e}")
        output_cl = clean_output(output, random_choice=False)
        output_dic = {"reason": "FAIL", "label": output_cl}
        return output_dic
```

We can now reuse our `generate_text` function from above with the new prompt, process the JSON Chain-of-Thought output with `process_output_cot` and send each prompt multiple times for Self-Consistency. 

```python
SELF_CONSISTENCY_ITERATIONS = 3

output_cot_multiple = []
for _ in range(SELF_CONSISTENCY_ITERATIONS):
    output_lst_step = []
    for text in tqdm(dataset["sentence"]):
        prompt_formatted = prompt_financial_sentiment_cot.format(text=text)
        output = generate_text(
            prompt=prompt_formatted, generation_params=generation_params
        )
        output_dic = process_output_cot(output)
        output_lst_step.append(output_dic["label"])

    output_cot_multiple.append(output_lst_step)
```

For each text, we now have three attempts by our LLM annotator to identify the correct label with three different reasoning paths. The code below selects the majority label from the three paths. 

```python
import pandas as pd
from collections import Counter

def find_majority(row):
    # Count occurrences
    count = Counter(row)
    # Find majority
    majority = count.most_common(1)[0]
    # Check if it's a real majority or if all labels are equally frequent
    if majority[1] > 1:
        return majority[0]
    else: # in case all labels appear with equal frequency
        return random.choice(labels)

df_output = pd.DataFrame(data=output_cot_multiple).T

df_output['label_pred_cot_multiple'] = df_output.apply(find_majority, axis=1)
```

Now we can compare our improved LLM labels with the expert labels again and calculate metrics. 

```python
label_experts = dataset["label_text"]
label_pred_cot_multiple = df_output['label_pred_cot_multiple']

metrics_cot_multiple = compute_metrics(label_experts, label_pred_cot_multiple)
```

With CoT and SC, performance increased to 94.0% accuracy and 0.94 F1 macro. By giving the model time to think about its label decision and giving it multiple attempts, we can further improve performance. Note that CoT and SC cost additional compute. We are essentially buying annotation accuracy with compute. 

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/fig_mixtral.png?raw=true" alt="fig_mixtral" width=95%>
</p>

With these LLM API calls we have now created a synthetic training dataset. We have labeled each text, by making the LLM try three different reasoning paths before taking the label decision. The result are labels with high agreement with human experts and a good quality dataset we can use for training a more efficient and specialized model. 

```python
df_train = pd.DataFrame({
    "text": dataset["sentence"],
    "labels": df_output['label_pred_cot_multiple']
})

df_train.to_csv("df_train.csv")
```

Note that in the [full reproduction script](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) for this blog post, we also create a test split purely based on the expert annotations to assess the quality of all models. All metrics are always based on this human expert test split. 

### 3.2. Comparing the open source model to proprietary models

The main advantage of this data created with the open-source Mixtral model is that the data is fully commercially usable without legal uncertainty. Data created with the OpenAI API, for example, is subject to the [OpenAI Business Terms](https://openai.com/policies/business-terms), which explicitly prohibit using model outputs for training models that compete with their products and services. The legal value and meaning of these Terms are unclear, but they introduce legal uncertainty for the commercial use of models trained on synthetic data from OpenAI models. Any smaller, efficient model trained on synthetic data could be considered as competing, as it reduces dependency on the API service. 

How does the quality of synthetic data compare between Mistral’s open-source `Mixtral-8x7B-Instruct-v0.1` and OpenAI’s GPT3.5 and GPT4? We ran the identical pipeline and prompts explained above with `gpt-3.5-turbo-0613` and `gpt-4-0125-preview` and report the results in the table below. We see that Mixtral performs better than GPT3.5 and on-par with GPT4 for this task, depending on the type of prompt. (We don’t display the results for the newer gpt-3.5-turbo-0125 here, because for some reason the performance with this model was worse than with the older default gpt-3.5-turbo-0613)

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/fig_mixtral_gpt.png?raw=true" alt="fig_mixtral_gpt" width=95%>
</p>

Note that this does not mean that Mixtral is always better than GPT3.5 and on-par with GPT4. GPT4 performs better on several benchmarks. The main message is that open-source models are now capable of creating high-quality synthetic data. 

### 3.3 Understand and validate your (synthetic) data

What does all this mean in practice? The result so far is just data that is annotated by some blackbox LLM. We could also only calculate metrics because we have expert annotated reference data from our example dataset. How can we trust the LLM annotations if we do not have expert annotations in a real-world scenario?

In practice, whatever annotator you are using (human annotators or LLMs), you can only trust data that you have validated yourself. Instructions/prompts always contain a degree of ambiguity and even a perfectly intelligent annotator can make mistakes and needs to make unclear decisions when faced with often ambiguous real-world data. 

Fortunately, data validation has become significantly easier over the past years with open-source tools: [Argilla](https://argilla.io/) provides a free interface for validating and cleaning unstructured LLM outputs; [LabelStudio](https://labelstud.io/) enables you to annotate data in many modalities; and [CleanLab](https://cleanlab.ai/) provides an interface for annotating and automatically cleaning structured data; for quick and simple validation, it can also be fine to just annotate in a simple Excel file. 

Its essential to spend some time annotating texts to get a feeling for the data and its ambiguities. You will quickly learn that the model made some mistakes, but there will also be several examples where the correct label is unclear and some texts where you agree more with the decision of the LLM than with the experts who created the dataset. These mistakes and ambiguities are a normal part of dataset creation. In fact, there are actually only very few real-world tasks where the human expert baseline is 100% agreement. It’s an old insight recently “rediscovered” by the machine learning literature that human data is a faulty gold standard ([Krippendorf 2004](https://books.google.de/books/about/Content_Analysis.html?id=q657o3M3C8cC&redir_esc=y) , [Hosking et al. 2024](https://arxiv.org/pdf/2309.16349.pdf)). 

After less than an hour in the the annotation interface, we have gained a better understanding of our data and corrected some mistakes. For reproducibility and to demonstrate the quality of purely synthetic data, however, we continue using the uncleaned LLM annotations in the next step. 

### 3.3 Tuning your efficient & specialized model with AutoTrain

So far, this has been a standard workflow of prompting an LLM through an API and validating the outputs. Now comes the additional step to enable significant resource savings: we fine-tune a smaller, more efficient and specialized LM on the LLM’s synthetic data. This process is also called “distillation”, where the output from a larger model (the “teacher”) is used to train a smaller model (the “student”). While this sounds fancy, it essentially only means that we take our original `text` from the dataset and treat the predictions from the LLM as our `labels` for fine-tuning. If you have trained a classifier before, you know that these are the only two columns you need to train a classifier, be it with `transformers`, `sklearn` or any other library. 

To make this process even easier, we use the Hugging Face [AutoTrain](https://huggingface.co/autotrain) solution. AutoTrain is a no-code interface, which enables you to upload a `.csv` file with labelled data, which the service then uses to fine-tune a model for you automatically. This removes the need for coding or in-depth fine-tuning expertise for training your own model.

On the Hugging Face website, we first click on “Spaces” at the top and then “Create new Space”. We then select “Docker” > “AutoTrain” and choose a small A10G GPU, which costs $1.05 per hour. The space for AutoTrain will then initialize. We can then upload or synthetic training data and expert test data via the interface and adjust the different fields as shown in the screenshot below. Once everything is filled in, we can click on “Start Training” and you can follow the training process in the Space’s logs. Training a small RoBERTa-base model (~0.13 B parameters) on just 1811 data points is very fast and should not take more than a few minutes. Once training is done, the model is automatically uploaded to your HF profile. The Space stops once training is finished and the whole process should not take longer than 15 minutes and cost less than $1. 

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/autotrain.png?raw=true" alt="autotrain" width=95%>
</p>

If you want, you can also use AutoTrain entirely locally on your own hardware, see our [documentation](https://huggingface.co/docs/autotrain/index). Advanced users can of course always write their own training scripts, but with these default hyperparameters the results with AutoTrain should be sufficient for many classification tasks. 

### 3.4 Performance and cost comparison

How do these different approaches compare? The table below displays the trade-offs across different factors and we discuss different metrics based on our example dataset underneath. 

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/table_pros_cons.png?raw=true" alt="table_pros_cons" width=85%>
</p>

Let’s start with task performance. How does the small fine-tuned ~0.13B parameter RoBERTa-base model compare to the much larger LLMs? The bar chart below shows that the custom model fine-tuned on 1811 texts performs on-par with the LLMs and it would be trivial to create more training data. A small model could never compete with a much larger LLM out-of-the-box, but fine-tuning it on some high-quality data brings it to the same level of performance. The fine-tuned model can only do the one specific task we have trained it to do, but it does it very well. 

<p align="center">
    <img src="https://github.com/huggingface/blog/blob/moritzlaurer/synthetic-data-save-costs/assets/176_synthetic-data-save-costs/fig_mixtral_gpt_roberta.png?raw=true" alt="fig_mixtral_gpt_roberta" width=95%>
</p>

Second, compute costs and inference speed. The main compute costs in practice will be inference, i.e. running the model after it has been trained. Let’s assume that in your production use-case, you need to process 1 million sentences in a given time period. Our fine-tuned RoBERTa-base model runs efficiently on a small T4 GPU with 16GB RAM, which costs $0.6 per hour on a HF Inference Endpoint. It has a latency of 0.13 seconds and a throughput of 61 sentences per second with batch_size=8. This leads to a total cost of $2.7 for processing 1 million sentences. 

With GPT models, we can calculate inference costs by counting tokens. Processing the tokens in 1 million sentences would cost ~$153 with GPT3.5 and ~$3061 with GPT4. The latency and throughput for these models is harder to calculate as it varies throughout the day depending on the current server load, but anyone working with GPT4 knows that latency can often be multiple seconds and is rate limited. Note that speed is an issue for any LLM (API), including open-source LLMs. Generative LLMs are simply too large to be fast.

Training compute costs tend to be less relevant, as LLMs can often be used out-of-the box without fine-tuning and fine-tuning costs of smaller models are relatively small (fine-tuning RoBERTa-base costs less than $1). Only in very few cases do you need to invest in pre-training a model from scratch. Training costs can become relevant when fine-tuning a larger generative LLM to specialize it in a specific generative task.

Third, required investments in time and expertise. This is the main strong point of LLM APIs. It is significantly easier to send instructions to an API than to collect data, fine-tune a custom model and deploy it. This is exactly where using an LLM API to create synthetic data becomes important. Creating good training data becomes significantly easier. Fine-tuning and deployment can then be handled by services like AutoTrain and dedicated Inference Endpoints. 

Fourth, control. This is probably the main disadvantage of LLM APIs. By design, LLM APIs make you dependent on the LLM API provider. You need to send your sensitive data to someone else’s servers and you cannot control the reliability and speed of your system. Training your own model lets you choose how and where to deploy it. 

Lastly, environmental impact. One query with GPT4 is estimated to consume [roughly 0.002 KWh of energy](https://towardsdatascience.com/chatgpts-energy-use-per-query-9383b8654487). 1 million queries will therefore cost roughly 2000 KWh, which is equivalent to 865 kg of CO2, or 2218 miles driven by an average car (see [EPA equivalence calculator](https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator)). Analysing 1 million sentences with our custom model, takes roughly 4.52 hours on a T4 GPU and emits around 0.12 kg of CO2 on AWS servers in US east N. Virginia (see [ML CO2 Impact calculator](https://mlco2.github.io/impact/)). Running a general-purpose LLM like GPT4 with (allegedly) 8x220B parameters is ridiculously inefficient compared to a specialized model with ~0.13B parameters. 

## Conclusion

We have shown the enormous benefits of using an LLM to create synthetic data to train a smaller, more efficient model. While this example only treats investor sentiment classification the same pipeline could be applied to many other tasks, from other classification tasks (e.g. customer intent detection or harmful content detection), to token-classification (e.g. named entity recognition or PII detection) to generative tasks (e.g. summarization or question answering). 

In 2024, it has never been easier for companies to create their own efficient models, control their own data and infrastructure, reduce CO2 emissions, and save compute costs and time without having to compromise on accuracy.

Now try it out yourself! You can find the full reproduction code for all numbers in this blog post, as well as more efficient asynchronous functions with batching for API calls in the [reproduction repository](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main). We invite you to copy and adapt our code to your use-cases!

