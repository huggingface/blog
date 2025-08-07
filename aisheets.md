---
title: "Introducing AI Sheets: a tool to work with datasets using open AI models!"
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/MeN-bF32JJvJhP-QY-0Eo.gif
authors:
- user: dvilasuero
- user: Ameeeee
- user: frascuchon
- user: damianpumar
- user: lvwerra
- user: thomwolf
---

# Introducing AI Sheets: a tool to work with datasets using open AI models!

**🧭TL;DR** 

Hugging Face AI Sheets is a new, open-source tool for building, enriching, and transforming datasets using AI models with no code. The tool can be deployed locally or on the Hub. It lets you use thousands of open models from the Hugging Face Hub via Inference Providers or local models, including `gpt-oss` from OpenAI! 

## Useful links

Try the tool for free (no installation required): [https://huggingface.co/spaces/aisheets/sheets](https://huggingface.co/spaces/aisheets/sheets)  
Install and run locally: [https://github.com/huggingface/sheets](https://github.com/huggingface/aisheets)


## What is AI Sheets

***AI Sheets*** is a no-code tool for building, transforming, and enriching datasets using (open) AI models. It’s tightly integrated with the Hub and the open-source AI ecosystem. 

AI Sheets uses an easy-to-learn user interface, similar to a spreadsheet. The tool is built around quick experimentation, starting with small datasets before running long/costly data generation pipelines. 

In AI Sheets, new columns are created by writing prompts, and you can iterate as many times as you need and edit the cells/validate cells to teach the model what you want. But more on this later\!

## What can I use it for

You can use AI Sheets to:

**Compare and vibe test models.** Imagine you want to test the latest models on your data. You can import a dataset with prompts/questions, and create several columns (one per model) with a prompt like this: `Answer the following: {{prompt}}`, where `prompt` is a column in your dataset. You can validate the results manually or create a new column with an LLM as a judge prompt like this: `Evaluate the responses to the following question: {{prompt}}. Response 1: {{model1}}. Response 2: {{model2}}`, where `model1` and `model2` are columns in your dataset with different model responses.

**Transform a dataset.** Imagine you want to clean up a column of your dataset. You can add a new column with a prompt like this: `Remove extra punctuation marks from the following text: {{text}}`, where `text` is a column in your dataset containing the texts you want to clean up.

**Classify a dataset.** Imagine you want to classify some content in your dataset. You can add a new column with a prompt like this: `Categorize the following text: {{text}}`, where `text` is a column in your dataset containing the texts you want to categorize.

**Analyze a dataset.** Imagine you want to extract the main ideas in your dataset. You can add a new column with a prompt like this: `Extract the most important ideas from the following: {{text}}`, where `text` is a column in your dataset containing the texts you want to analyze.

**Enrich a dataset.** Imagine you have a dataset with addresses that are missing zip codes. You can add a new column with a prompt like this: `Find the zip code of the following address: {{address}}` (in this case, you must enable the "Search the web" option to ensure accurate results).

**Generate a synthetic dataset.** Imagine you need a dataset with realistic emails, but the data is not available for data privacy reasons. You can create a dataset with a prompt like this: `Write a short description of a professional in the field of pharma companies` and name the column `person_bio`. Then you can create another column with a prompt like this `Write a realistic professional email as it was written by the following person: {{person_bio}}`.

Now let’s dive into how to use it\!

## How to use it

AI Sheets gives you two ways to start: import existing data or generate a dataset from scratch. Once your data is loaded, you can refine it by adding columns, editing cells, and regenerating content.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/jLZPLa0x2EC9Xw4r3PXa6.png)

### Getting started

To get started, you need to import a dataset or create one from scratch.

#### Import your dataset (recommended)

***Best for:** Most use cases where you want to transform, classify, enrich, and analyze real-world data.*

*This is recommended for most use cases, as importing real data gives you more control and flexibility than starting from scratch.*

***When to use this:***

* *You have existing data to transform or enrich using AI models*  
* *You want to generate synthetic data, and accuracy and diversity are important*

***How it works:***

1. *Upload your data in XLS, TSV, CSV, or Parquet format*  
2. *Ensure your file includes at least one column name and one row of data*  
3. *Upload up to 1,000 rows (unlimited columns)*  
4. *Your data appears in a familiar spreadsheet format*

***Pro tip:** If your file contains minimal data, you can manually add more entries by typing directly into the spreadsheet.*

#### Generate Dataset from Scratch 

**Best for:** Familiarizing with AI Sheets, brainstorming, rapid experiments, and creating test datasets.

Think of this as an auto-dataset or prompt-to-dataset feature—you describe what you want, and AI Sheets creates the entire dataset structure and content for you.

**When to use this:**

* You're exploring AI Sheets for the first time  
* You need synthetic data for testing or prototyping  
* Data accuracy and diversity are not critical (e.g., brainstorming use cases, quick research, generating test datasets)  
* You want to experiment with ideas quickly

**How it works:**

1. Describe the dataset you want in the prompt area  
   * Example: "A list of fictional startups with name, industry, and slogan"  
2. AI Sheets generates the schema and creates 5 sample rows  
3. Extend to up to 1,000 rows or modify the prompt to change structure


### Working with your dataset 

Once your data is loaded (regardless of how you started), you'll see it in an editable spreadsheet interface. Here's what you need to know:

**Understanding AI Sheets**

* **Imported cells:** Manually editable but can't be modified by AI prompts  
* **AI-generated cells:** Can be regenerated and refined using prompts and your feedback (edits \+ thumbs-up)  
* **New columns:** Always AI-powered and fully customizable

**Getting Started with AI columns**

1. Click the "+" button to add a new column  
2. Choose from recommended actions:  
   * Extract specific information  
   * Summarize long text  
   * Translate content  
   * Or write custom prompts with "Do something with {{column}}"

## Refining and expanding the dataset

Now that you have AI columns, you can improve their results and expand your data. You can improve results by providing feedback through manual edits and likes or by adjusting the column configuration. Both require regeneration to take effect.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/nlVWiRnUKCi308kZ-Fxv0.png)

**1\. How to add more cells**

* **Drag down:** From the last cell in a column to generate additional rows immediately  
* No regeneration needed \- new cells are created instantly  
* You can use this to regenerate errored cells too

**2\. Manual editing and feedback**

* **Edit cells:** Click any cell to edit content directly \- this gives the model examples of your preferred output  
* **Like results:** Use thumbs-up to mark examples of good output  
* Regenerate to apply feedback to other cells in the column

**3\. Adjust column configuration** Change the prompt, switch models or providers, or modify settings, then regenerate to get better results.

**Rewrite the prompt**

* Each column has its generation prompt  
* Edit anytime to change or improve output  
* Column regenerates with new results

**Switch models / providers**

* Try different models for different performance or compare them.  
* Some are more accurate, creative, or structured than others for specific tasks.  
* Some providers have faster inference and different context lengths; test different providers for the selected model. 

**Toggle Search**

* Enable: Model pulls up-to-date information from the web   
* Disable: Offline, model-only generation

## Examples

This section provides examples of datasets you can build with AI Sheets to inspire your next project.

### Vibe testing and comparing models
AI Sheets is your perfect companion if you want to test the latest models on different prompts and data you care about.

You just need to import a dataset (or create one from scratch) and then add different columns with the models you want to test.

Then, you can either inspect the results manually or add a column to use LLMs to judge the quality of each model.

Below is an example, comparing open frontier models for mini web apps. AI Sheet lets you see the interactive results and play with each app. Additionally, the dataset includes several columns using LLM to judge and compare the quality of the apps.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/ArLkbk5trsp1CzehDw45N.png)

Example dataset: https://huggingface.co/datasets/dvilasuero/jsvibes-qwen-gpt-oss-judged

Config:

```yml
columns:
  gpt-oss:
    modelName: openai/gpt-oss-120b
    modelProvider: groq
    userPrompt: Create a complete, runnable HTML+JS file implementing {{description}}
    searchEnabled: false
    columnsReferences:
      - description
  eval-qwen-coder:
    modelName: Qwen/Qwen3-Coder-480B-A35B-Instruct
    modelProvider: cerebras
    userPrompt: "Please compare the two apps and tell me which one is better and why:\n\nApp description:\n\n{{description}}\n\nmodel 1:\n\n{{qwen3-coder}}\n\nmodel 2:\n\n{{gpt-oss}}\n\nKeep it very short and focus on whether they work well for the purpose, make sure they work and are not incomplete, and the code quality, not on visual appeal and unrequested features. Assume the models might provide non working solutions, so be careful to assess that\n\nRespond with:\n\nchosen: {model 1, model 2}\n\nreason: ..."
    searchEnabled: false
    columnsReferences:
      - gpt-oss
      - description
      - qwen3-coder
  eval-gpt-oss:
    modelName: openai/gpt-oss-120b
    modelProvider: groq
    userPrompt: "Please compare the two apps and tell me which one is better and why:\n\nApp description:\n\n{{description}}\n\nmodel 1:\n\n{{qwen3-coder}}\n\nmodel 2:\n\n{{gpt-oss}}\n\nKeep it very short and focus on whether they work well for the purpose, make sure they work and are not incomplete, and the code quality, not on visual appeal and unrequested features. Assume the models might provide non working solutions, so be careful to assess that\n\nRespond with:\n\nchosen: {model 1, model 2}\n\nreason: ..."
    searchEnabled: false
    columnsReferences:
      - gpt-oss
      - description
      - qwen3-coder
  eval-kimi:
    modelName: moonshotai/Kimi-K2-Instruct
    modelProvider: groq
    userPrompt: "Please compare the two apps and tell me which one is better and why:\n\nApp description:\n\n{{description}}\n\nmodel 1:\n\n{{qwen3-coder}}\n\nmodel 2:\n\n{{gpt-oss}}\n\nKeep it very short and focus on whether they work well for the purpose, make sure they work and are not incomplete, and the code quality, not on visual appeal and unrequested features. Assume the models might provide non working solutions, so be careful to assess that\n\nRespond with:\n\nchosen: {model 1, model 2}\n\nreason: ..."
    searchEnabled: false
    columnsReferences:
      - gpt-oss
      - description
      - qwen3-coder
```

### Add categories to a Hub dataset
AI Sheets can also augment existing datasets and help you with quick data analysis and data science projects that involve analyzing text datasets.

Here's an example of adding categories to an existing Hub dataset.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/K3QMIBf0fSeJFUEA3oI5H.png)

A cool feature is that you can validate or edit manually the initial categorization outputs and regenerate the full column to improve the results, as seen below:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/fp7FE4wpCPP9zU48Cyd6S.png)

Example dataset: 
Config:
```yml
columns:
  category:
    modelName: moonshotai/Kimi-K2-Instruct
    modelProvider: groq
    userPrompt: |-
      Categorize the main topics of the following question:

      {{question}}
    prompt: "

      You are a rigorous, intelligent data-processing engine. Generate only the
      requested response format, with no explanations following the user
      instruction. You might be provided with positive, accurate examples of how
      the user instruction must be completed.

      # Examples

      The following are correct, accurate example outputs with respect to the
      user instruction:

      ## Example

      ### Input

      question: Given the area of a parallelogram is 420 square centimeters and
      its height is 35 cm, find the corresponding base. Show all work and label
      your answer.

      ### Output

      Mathematics – Geometry

      ## Example

      ### Input

      question: What is the minimum number of red squares required to ensure
      that each of $n$ green axis-parallel squares intersects 4 red squares,
      assuming the green squares can be scaled and translated arbitrarily
      without intersecting each other?

      ### Output

      Geometry, Combinatorics
      # User instruction

      Categorize the main topics of the following question:

      {{question}}

      # Your response
      "
    searchEnabled: false
    columnsReferences:
      - question
```

### Evaluate models with LLMs-as-Judge 
Another use case is evaluating the outputs of models using an LLM as a judge approach. This can be useful for comparing models or assessing the quality of an existing dataset, for example, fine-tuning a model on an existing dataset on the Hugging Face Hub.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/uiP176LVIcfSKHC-RUH6r.png)

Example dataset: https://huggingface.co/datasets/dvilasuero/jsvibes-qwen-gpt-oss-judged

Config:

```yml
columns:
  object_name:
    modelName: meta-llama/Llama-3.3-70B-Instruct
    modelProvider: groq
    userPrompt: Generate the name of a common day to day object
    searchEnabled: false
    columnsReferences: []
  object_description:
    modelName: meta-llama/Llama-3.3-70B-Instruct
    modelProvider: groq
    userPrompt: Describe a {{object_name}} with adjectives and short word groups separated by commas. No more than 10 words
    searchEnabled: false
    columnsReferences:
      - object_name
  object_image_with_desc:
    modelName: multimodalart/isometric-skeumorphic-3d-bnb
    modelProvider: fal-ai
    userPrompt: RBNBICN, icon, white background, isometric perspective, {{object_name}} , {{object_description}}
    searchEnabled: false
    columnsReferences:
      - object_description
      - object_name
  object_image_without_desc:
    modelName: multimodalart/isometric-skeumorphic-3d-bnb
    modelProvider: fal-ai
    userPrompt: "RBNBICN, icon, white background, isometric perspective, {{object_name}} "
    searchEnabled: false
    columnsReferences:
      - object_name
  glowing_colors:
    modelName: multimodalart/isometric-skeumorphic-3d-bnb
    modelProvider: fal-ai
    userPrompt: "RBNBICN, icon, white background, isometric perspective, {{object_name}}, glowing colors "
    searchEnabled: false
    columnsReferences:
      - object_name
  flux:
    modelName: black-forest-labs/FLUX.1-dev
    modelProvider: fal-ai
    userPrompt: Create an isometric icon for the object {{object_name}} based on {{object_description}}
    searchEnabled: false
    columnsReferences:
      - object_description
      - object_name
```

## Next steps
You can get started with AI Sheets [for free without installing anything](https://huggingface.co/spaces/aisheets/sheets) or download and deploy it locally from the [GitHub repo](https://github.com/huggingface/aisheets).

If you have questions or suggestions, let us know in the [Community tab](https://huggingface.co/spaces/aisheets/sheets/discussions) or by opening an issue on [GitHub](https://github.com/huggingface/aisheets).

