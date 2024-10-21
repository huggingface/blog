---
title: "HF Expert Support story: Bolstering a RAG app with LLM-as-a-Judge"
thumbnail: /blog/assets/digital-gren-llm-judge/thumbnail.png
authors:
  - user: m-ric
---
# HF Expert Support story: Bolstering a RAG app with LLM-as-a-Judge

> [!NOTE] This is a guest blog post by the Digital Green team.

There are an estimated 500 million smallholder farmers globally: they play a critical role in global food security. Timely access to accurate information is essential for these farmers to make informed decisions and improve their yields.

An “agricultural extension service” offers technical advice on agriculture to farmers, and also supplies them with the necessary inputs and services to support their agricultural production.   
Agriculture extension agents are 300K in India alone, they provide necessary information about improved agriculture practice and help in decision making for the smallholder farmers.

But although their number is impressive, extension workers are not in large enough numbers to cope with all the demand: they interact with farmers at typically in the ratio of 1:1000. Reaching the agriculture extension workers and farmers through partnership and technology remains the key.

Enter project GAIA, a collaborative initiative pioneered by [CGIAR](https://www.cgiar.org/).  
It brought together [Hugging Face](https://huggingface.co/) as mentor through the [Expert Support program](https://huggingface.co/support), and [Digital Green](http://digitalgreen.org) as project partner.

GAIA has a lofty goal to bring years of agriculture knowledge in the form of research papers meticulously maintained in [GARDIAN portal](https://gardian.bigdata.cgiar.org/#/) in the hands of the farmers. There are close to 46000 research papers and reports that have agricultural knowledge globally carried over multiple decades across different crops.

[Digital Green](http://www.digitalgreen.org) immediately saw the potential of developing intelligent chatbots powered by Retrieval-Augmented Generation (RAG) on approved, curated information. Thus they decided to develop [farmer.chat](https://farmerchat.digitalgreen.org/), a chatbot that leverages the capabilities of large language models (LLMs) to deliver personalized and reliable agricultural advice to the farmers and front line extension workers.

Creating such a chatbot for a huge variety of languages, geographies, crops, and use cases, is a gigantic challenge:  information disseminated has to be contextual to the local level details about the farm, in the language and tone that farmers can understand and accurate (grounded in trustworthy sources) for farmers to act on it. To evaluate the performance of the system, the CGIAR team and HF expert collaborated to set up a strong evaluation suite, in the form of an LLM-as-a-judge system.

Let’s take a look at how they tackled this challenge!

## System architecture

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*L8epmSjWRweoVhQoLlAbuQ.png" alt="System architecture" width=100%>
</p>

The full system uses many components in order to provide chatbot answers grounded in several tools and external knowledge. It has several key elements:

* **Knowledge base:**  
  * Preprocessing: The first step was to ingest the pdf documents into the farmer.chat pipeline with the help of APIs maintained by [Scio](https://scio.systems/). The in the knowledge base, topics were auto categorized for relevant geographic areas and semantically grouped together.  
  * Semantic chunking: the organized files with metadata are processed with sentences similar in meaning grouped together in text chunks. The function uses small-text embedding currently for cosine similarity  
  * Conversion into VectorDB format: each text chunk is converted into vector representation using an embedding model using which the vector representation is stored in QdrantDB.  
* **RAG pipeline:** It is what ensures that the information delivered is grounded in the content and not outside. It consists in two parts:  
  * Information retrieval: Searching the knowledge base for relevant information that matches the user’s query. This involves calling the vector database API created in knowledge base builder to get necessary text chunks.  
  * Generation: Using the retrieved information in the text chunks and the user query, the generator calls LLM and generates a human-like response that addresses the user’s needs.  
* **User-facing Agent:** The planning agent leverages GPT-4o under the hood.  
  * Its task is to:  
    * Understand the user intent  
    * Based on the user intent and tools description decide what more information is required  
    * Ask that information from the user till the ask is clear  
    * Once the ask is clear, call the execution agent   
    * Get the response form the execution agent and generate response  
  * The agent runs a ReAct based prompt to think in step by step manner and call the respective tools and analyze the responses. Then it can leverage its tools to answer: Currently, the agent uses the following set of tools:  
    * Converse more  
    * RAG QA endpoint  
    * Video retrieval endpoint  
    * Weather endpoint  
    * Crop table

Now this system has many moving parts, and each part has a radical impart on some aspects of performance. So we need to carefully run performance evaluation.

In the last one year, the usage of farmer.chat has grown to service more than 20k farmers handling over 340k queries. How can we evaluate the performance of the system at this scale?

Here, we enter the exciting realm of using LLMs as judges to assess the performance of these AI systems. 

During weekly brainstorming sessions, Hugging Face provided a source to their  article on [LLM as a Judge](https://huggingface.co/learn/cookbook/en/llm_judge): it was discussed in detail, and what followed became a practice that has helped navigating farmer.chat’s development. LLM as judge helped create a metrics driven approach to measure the accuracy and impact of [farmer.chat](https://farmerchat.digitalgreen.org/) in a more objective and data-driven way. 

## The Power of LLMs-as-Judges

The use of LLMs-as-Judges lies at the heart of metrics-driven development that becomes necessary in the rapidly evolving landscape. It empowers us to go beyond simply measuring how many questions a chatbot can answer or how quickly it responds. Instead, we can delve deeper into the quality of the responses and understand the user experience in a more nuanced way. The results were published recently in this [scientific article](https://arxiv.org/abs/2409.08916), focusing on the quantitative study of user research.

Farmer.Chat employs a sophisticated Retrieval-Augmented Generation (RAG) pipeline to deliver accurate and relevant information to farmers that is grounded in the knowledge base. The RAG pipeline uses an LLM to retrieve information from a vast knowledge base and then generate a concise and informative response. But how do we measure the effectiveness of this pipeline?

Here, LLMs step in as judges. The research team behind Farmer.Chat leverages the capabilities of LLMs to evaluate several crucial metrics:

* **Prompt Clarity Score**: This metric evaluates how well users can articulate their questions. LLMs are trained to assess the clarity of user intent, topic specificity, and entity-attribute identification, providing insights into how effectively users can communicate their needs.  
* **Question Type**: This metric classifies user questions into different categories based on their cognitive complexity. LLMs analyze the user's query and assign it to one of six categories, such as "remember," "understand," "apply," "analyze," "evaluate," and "create," helping us understand the cognitive demands of user interactions.  
* **Answered Queries**: This metric tracks the percentage of questions answered by the chatbot, providing insights into the breadth of the knowledge base and the platform's ability to address a wide range of queries.  
* **RAG Accuracy**: This metric assesses the faithfulness and relevance of the information retrieved by the RAG pipeline. The LLM acts as a judge, comparing the retrieved information to the user's query and evaluating whether the response is accurate and relevant.

What is interesting about the metrics is how versatile LLM as Judge can be. For example, you can use it to evaluate the clarity of a prompt as follows:

```
Given a user input about agriculture, you are supposed to score on the clarity.
You are supposed to score on a scale of 1 to 3: 1 being lowest and 3 highest on overall clarity of user input.
Think step by step and rate the user input on all three criteria and give a score for each:   
a) the intent and ask is clear,   
b) it is about a specific topic, and   
c) it is about a specific entity and its attribute.

After creating a score for each three, take the average and round it off to the nearest integer which becomes the final score.  

Example:
user input is "tell the benefits of batian coffee variety"
scores 3 for first criteria as the intent is clear about knowing about batian variety of coffee and ask is clear want to summarize the benefits,   
scores 3 on second criteria as it is about coffee varieties and   
scores 2 on third criteria as it is about batian variety but not about specific attribute like disease resistant or high yielding.”
```

As mentioned in [this article that we referred earlier](https://huggingface.co/learn/cookbook/en/llm_judge), the key to use LLM-as-a-judge is to define the task, define the criteria and define the rating on a certain scale. 

On the other hand, for RAG accuracy we use LLM-as-a-judge to evaluate on a binary scale: zero or one. But the way the task is broken down leads to a well established process that comes up with a score that we tested with human evaluators on roughly 360 questions and answers and found to actually do a better job and high correlation\!\!

Here is the prompt, which has been inspired from the RAGAS library.

```  
You are a natural language inference engine. You will be presented with a set of factual statements and context. You are supposed to analyze if each statement is factually correct given the context. You can come up with the scores of 'Yes' (1) and 'No' (0) as verdict.

Use the following rules:  
If the statement can be derived from the context, give a score of 1  
If there is no statement and there is no context, give a score of 1  
If the statement can’t be derived from the context, give a score of 0  
If there is no context but there is a statement, give a score of 0

#### Input :

##
context : {context}

##
statements : {statements}
```
The context variable above is the input chunks given for generating the answers while statements are the atomic factual statements generated by another LLM call.

This was a very important step as it enables evaluation at scale which is important when dealing with large numbers of documents and queries. The LLM-as-a-judge at core leads to metrics that act as a compass navigating the various options available for AI pipeline.

### Results: benchmarking LLMs for RAG

We created a sample dataset of \> 700 user queries randomized across different value chains (crops) and date (months). While this upgrade itself had 11 different versions that evaluated using RAG accuracy and percentage answered, the same approach was used to measure performance of the leading LLMs without any prompt changes in each LLM call. For this experiment, we selected GPT-4 by OpenAI, Gemini-1.5 in Pro and Flash versions, and Llama-3-70B-Instruct.

| LLM | Faithful | Relevance | Answered \* Relevance | Answered \*  Faithful | %  Unanswered |
| :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4-turbo | 88% | 75% | 59% | 69% | 21.95% |
| Llama3-70B | 78% | 76% | 76% | 78% | 0.28% |
| Gemini-1.5-Pro | 91% | 88% | 71% | 73% | 19.37% |
| Gemini-1.5-Flash | 89% | 78% | 74 % | 85% | 4.50% |

What we see is that amongst the four models, the highest level of factually correct answers (“Faithful” column) is obtained with Gemini-1.5-pro, followed very closely by Gemini 1.5-flash and GPT-4-turbo.

We also decided to score the performance of the LLM based on percentage answered times average relevance and percentage answered time faithful, which are columns 4th and 5th in the table above. While Llama-3-70B comes out on top in the first one, it is distant second on the other. 

What we found was that purely on the basis of accuracy (a response is deemed accurate if faithful \+ relevant to the question), Gemini-1.5-pro supersedes the other models followed by gpt4 at the time of testing. But if we also put the criteria of percentage answered, Llama3-70B and Gemini 1.5-flash perform better simply because they tend to answer even if the context is empty. 

### Conclusion

By leveraging LLMs as judges, we gain a deeper understanding of user behavior and the effectiveness of AI-powered tools in the agricultural context. This data-driven approach is crucial for:

* **Improving user experience**: By identifying areas where users struggle to articulate their needs or where the RAG pipeline is not performing as expected, we can improve the design and functionality of the platform.  
* **Optimizing the knowledge base**: The analysis of unanswered queries helps us identify gaps in the knowledge base and prioritize content development.  
* **Selecting the right LLMs**: By benchmarking different LLMs on key metrics, we can make informed decisions about which models are best suited for specific tasks and contexts.

In the rapidly evolving landscape of generative AI, LLMs are becoming increasingly powerful and versatile. Their ability to act as judges in evaluating AI performance is a game-changer. It allows us to measure the impact of these systems in a more objective and data-driven way, ultimately leading to the development of more robust, effective, and user-friendly AI tools for agriculture.

In the span of over a year, we have continuously evolved our product. In this small timeframe we have been able to:

* Reach more than 20k farmers  
* Answer \> 340k questions  
* Serve \> 6 languages, for 50 value chain crops  
* Maintain close to zero biases or toxic responses

<p align="center">
    <img src="https://cdn.prod.website-files.com/659d11eefb40654676991482/65e612f59aa42d0216e15236_Map%20and%20data%20(3).gif" alt="System demo" width=80%>
</p>

_If you are interested in the Hugging Face Expert Support program for your company, don't hesitate to contact us [here](https://huggingface.co/contact/sales?from=support) - our sales team will get in touch to discuss your needs!_