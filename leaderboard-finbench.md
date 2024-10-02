---
title: "Introducing the Open FinLLM Leaderboard - Selecting the best AI models for finance"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail.png
authors:
- user: QianqianXie1994
  org: TheFinAI
  guest: true
- user: jiminHuang
  org: TheFinAI
  guest: true
- user: Effoula
  org: TheFinAI
  guest: true
- user: yanglet
  guest: true
- user: alejandroll10
  guest: true
- user: Benyou
  guest: true
- user: ldruth
  org: TheFinAI
  guest: true
- user: xiangr
  org: TheFinAI
  guest: true
- user: Me1oy
  org: TheFinAI
  guest: true
- user: ShirleyY
  guest: true
- user: mirageco
  guest: true
- user: blitzionic
  guest: true
- user: clefourrier
---

# Introducing the Open FinLLM Leaderboard - Selecting the best AI models for finance

The growing complexity of financial language models (LLMs) necessitates evaluations that go beyond general NLP benchmarks. While traditional leaderboards focus on broader NLP tasks like translation or summarization, they often fall short in addressing the specific needs of the finance industry. Financial tasks, such as predicting stock movements, assessing credit risks, and extracting information from financial reports, present unique challenges that require models with specialized skills. This is why we decided to create the **Open FinLLM Leaderboard** 

The leaderboard provides a **specialized evaluation framework** tailored specifically to the financial sector. We hope it fills this critical gap, by providing a transparent framework that assesses model readiness for real-world use with a one-stop solution. The leaderboard is designed to highlight a model's **financial skill** by focusing on tasks that matter most to finance professionals—such as information extraction from financial documents, market sentiment analysis, and forecasting financial trends.

* **Comprehensive Financial Task Coverage:** The leaderboard evaluates models only on tasks that are directly relevant to finance. These tasks include **information extraction**, **sentiment analysis**, **credit risk scoring**, and **stock movement forecasting**, which are crucial for real-world financial decision-making.  
* **Real-World Financial Relevance:** The datasets used for the benchmarks represent real-world challenges faced by the finance industry. This ensures that models are actually assessed on their ability to handle complex financial data, making them suitable for industry applications.  
* **Focused Zero-Shot Evaluation:** The leaderboard employs a **zero-shot evaluation** method, testing models on unseen financial tasks without any prior fine-tuning. This approach reveals a model’s ability to generalize and perform well in financial contexts, such as predicting stock price movements or extracting entities from regulatory filings, without being explicitly trained on those tasks.


## **Key Features of the Open Financial LLM Leaderboard**

* **Diverse Task Categories:** The leaderboard covers tasks across seven categories: Information Extraction (IE), Textual Analysis (TA), Question Answering (QA), Text Generation (TG), Risk Management (RM), Forecasting (FO), and Decision-Making (DM).  
* **Evaluation Metrics:** Models are assessed using a variety of metrics, including Accuracy, F1 Score, ROUGE Score, and Matthews Correlation Coefficient (MCC). These metrics provide a multidimensional view of model performance, helping users identify the strengths and weaknesses of each model.


## Supported Tasks and Metric

The **Open Financial LLM Leaderboard (OFLL)** evaluates financial language models across a diverse set of categories that reflect the complex needs of the finance industry. Each category targets specific capabilities, ensuring a comprehensive assessment of model performance in tasks directly relevant to finance.

### Categories

The selection of task categories in OFLL is intentionally designed to capture the full range of capabilities required by financial models. This approach is influenced by both the diverse nature of financial applications and the complexity of the tasks involved in financial language processing.

* **Information Extraction (IE):** The financial sector often requires structured insights from unstructured documents such as regulatory filings, contracts, and earnings reports. Information extraction tasks include **Named Entity Recognition (NER)**, **Relation Extraction**, and **Causal Classification**. These tasks evaluate a model’s ability to identify key financial entities, relationships, and events, which are crucial for downstream applications such as fraud detection or investment strategy.  
* **Textual Analysis (TA):** Financial markets are driven by sentiment, opinions, and the interpretation of financial news and reports. Textual analysis tasks such as **Sentiment Analysis**, **News Classification**, and **Hawkish-Dovish Classification** help assess how well a model can interpret market sentiment and textual data, aiding in tasks like investor sentiment analysis and policy interpretation.  
* **Question Answering (QA):** This category addresses the ability of models to interpret complex financial queries, particularly those that involve numerical reasoning or domain-specific knowledge. The QA tasks, such as those derived from datasets like **FinQA** and **TATQA**, evaluate a model’s capability to respond to detailed financial questions, which is critical in areas like risk analysis or financial advisory services.  
* **Text Generation (TG):** Summarization of complex financial reports and documents is essential for decision-making. Tasks like **ECTSum** and **EDTSum** test models on their ability to generate concise and coherent summaries from lengthy financial texts, which is valuable in generating reports or analyst briefings.  
* **Forecasting (FO):** One of the most critical applications in finance is the ability to forecast market movements. Tasks under this category evaluate a model’s ability to predict stock price movements or market trends based on historical data, news, and sentiment. These tasks are central to tasks like portfolio management and trading strategies.  
* **Risk Management (RM):** This category focuses on tasks that evaluate a model’s ability to predict and assess financial risks, such as **Credit Scoring**, **Fraud Detection**, and **Financial Distress Identification**. These tasks are fundamental for credit evaluation, risk management, and compliance purposes.  
* **Decision-Making (DM):** In finance, making informed decisions based on multiple inputs (e.g., market data, sentiment, and historical trends) is crucial. Decision-making tasks simulate complex financial decisions, such as **Mergers & Acquisitions** and **Stock Trading**, testing the model’s ability to handle multimodal inputs and offer actionable insights.

#### Metrics

* **F1-score**, the harmonic mean of precision and recall, offers a balanced evaluation, especially important in cases of class imbalance within the dataset. Both metrics are standard in classification tasks and together give a comprehensive view of the model's capability to discern sentiments in financial language.  
* **Accuracy** measures the proportion of correctly classified instances out of all instances, providing a straightforward assessment of overall performance.  
* **RMSE** provides a measure of the average deviation between predicted and actual sentiment scores, offering a quantitative insight into the accuracy of the model's predictions.   
* **Entity F1 Score (EntityF1)**. This metric assesses the balance between precision and recall specifically for the recognized entities, providing a clear view of the model's effectiveness in identifying relevant financial entities. A high EntityF1 indicates that the model is proficient at both detecting entities and minimizing false positives, making it an essential measure for applications in financial data analysis and automation.  
* **Exact Match Accuracy (EmAcc)** measures the proportion of questions for which the model’s answer exactly matches the ground truth, providing a clear indication of the model's effectiveness in understanding and processing numerical information in financial contexts. A high EmAcc reflects a model's capability to deliver precise and reliable answers, crucial for applications that depend on accurate financial data interpretation.  
* **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to assess the quality of summaries by comparing them to reference summaries. It focuses on the overlap of n-grams between the generated summaries and the reference summaries, providing a measure of content coverage and fidelity.  
* **BERTScore** utilizes contextual embeddings from the BERT model to evaluate the similarity between generated and reference summaries. By comparing the cosine similarity of the embeddings for each token, BERTScore captures semantic similarity, allowing for a more nuanced evaluation of summary quality.  
* **BARTScore** is based on the BART (Bidirectional and Auto-Regressive Transformers) model, which combines the benefits of both autoregressive and autoencoding approaches for text generation. It assesses how well the generated summary aligns with the reference summary in terms of coherence and fluency, providing insights into the overall quality of the extraction process.  
* **Matthews Correlation Coefficient (MCC)** takes into account true and false positives and negatives, thereby offering insights into the model's effectiveness in a binary classification context. Together, these metrics ensure a comprehensive assessment of a model's predictive capabilities in the challenging landscape of stock movement forecasting.  
* **Sharpe Ratio (SR).** The Sharpe Ratio measures the model's risk-adjusted return, providing insight into how well the model's trading strategies perform relative to the level of risk taken. A higher Sharpe Ratio indicates a more favorable return per unit of risk, making it a critical indicator of the effectiveness and efficiency of the trading strategies generated by the model. This metric enables users to gauge the model’s overall profitability and robustness in various market conditions.

### Individual Tasks

We use 40 tasks on this leaderboard, across these categories:  
- **Information Extraction(IE)**: NER, FiNER-ORD, FinRED, SC, CD, FNXL, FSRL  
- **Textual Analysis(TA)**: FPB, FiQA-SA, TSA, Headlines, FOMC, FinArg-ACC, FinArg-ARC, MultiFin, MA, MLESG  
- **Question Answering(QA)**: FinQA, TATQA, Regulations, ConvFinQA  
- **Text Generation(TG)**: ECTSum, EDTSum  
- **Risk Management(RM)**: German, Australian, LendingClub, ccf, ccfraud, polish, taiwan, ProtoSeguro, travelinsurance  
- **Forecasting(FO)**: BigData22, ACL18, CIKM18  
- **Decision-Making(DM)**: FinTrade  
- **Spanish**: MultiFin-ES, EFP ,EFPA ,FinanceES, TSA-Spanish

For a detailed explanation or each task, please read the next section

#### Explaination of Tasks

1. **FPB (Financial PhraseBank Sentiment Classification)**  
   **Description:** Sentiment analysis of phrases in financial news and reports, classifying into positive, negative, or neutral categories.  
   **Metrics:** Accuracy, F1-Score								Source: [https://huggingface.co/datasets/ChanceFocus/en-fpb](https://huggingface.co/datasets/ChanceFocus/en-fpb)  
2. **FiQA-SA (Sentiment Analysis in Financial Domain)**  
   **Description:** Sentiment analysis in financial media (news, social media). Classifies sentiments into positive, negative, and neutral, aiding in market sentiment interpretation.  
   **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/ChanceFocus/flare-fiqasa](https://huggingface.co/datasets/ChanceFocus/flare-fiqasa)  
3. **TSA (Sentiment Analysis on Social Media)**  
   **Description:** Sentiment classification for financial tweets, reflecting public opinion on market trends. Challenges include informal language and brevity.			**Metrics:** F1-Score, RMSE								Source: [https://huggingface.co/datasets/ChanceFocus/flare-fiqasa](https://huggingface.co/datasets/ChanceFocus/flare-fiqasa)  
4. **Headlines (News Headline Classification)**  
   **Description:** Classification of financial news headlines into sentiment or event categories. Critical for understanding market-moving information.  
   **Metrics:** AvgF1									Source: [https://huggingface.co/datasets/ChanceFocus/flare-headlines](https://huggingface.co/datasets/ChanceFocus/flare-headlines)

5. **FOMC (Hawkish-Dovish Classification)**  
   **Description:** Classification of FOMC statements as hawkish (favoring higher interest rates) or dovish (favoring lower rates), key for monetary policy predictions.  
   **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-fomc](https://huggingface.co/datasets/ChanceFocus/flare-fomc)  
6. **FinArg-ACC (Argument Unit Classification)**  
   **Description:** Identifies key argument units (claims, evidence) in financial texts, crucial for automated document analysis and transparency.  
   **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-finarg-ecc-auc](https://huggingface.co/datasets/ChanceFocus/flare-finarg-ecc-auc)  
7. **FinArg-ARC (Argument Relation Classification)**  
   **Description:** Classification of relationships between argument units (support, opposition) in financial documents, helping analysts construct coherent narratives.  
   **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-finarg-ecc-arc](https://huggingface.co/datasets/ChanceFocus/flare-finarg-ecc-arc)  
8. **MultiFin (Multi-Class Sentiment Analysis)**  
   **Description:** Classification of diverse financial texts into sentiment categories (bullish, bearish, neutral), valuable for sentiment-driven trading.  
   **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-es-multifin](https://huggingface.co/datasets/ChanceFocus/flare-es-multifin)  
9. **MA (Deal Completeness Classification)**  
   **Description:** Classifies mergers and acquisitions reports as completed, pending, or terminated. Critical for investment and strategy decisions.  
   **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-ma](https://huggingface.co/datasets/ChanceFocus/flare-ma)  
10. **MLESG (ESG Issue Identification)**  
    **Description:** Identifies Environmental, Social, and Governance (ESG) issues in financial documents, important for responsible investing.  
    **Metrics:** F1-Score, Accuracy								Source: [https://huggingface.co/datasets/ChanceFocus/flare-mlesg](https://huggingface.co/datasets/ChanceFocus/flare-mlesg)  
11. **NER (Named Entity Recognition in Financial Texts)**  
    **Description:** Identifies and categorizes entities (companies, instruments) in financial documents, essential for information extraction.  
    **Metrics:** Entity F1-Score								Source: [https://huggingface.co/datasets/ChanceFocus/flare-ner](https://huggingface.co/datasets/ChanceFocus/flare-ner)  
12. **FINER-ORD (Ordinal Classification in Financial NER)**  
    **Description:** Extends NER by classifying entity relevance within financial documents, helping prioritize key information.  
    **Metrics:** Entity F1-Score								Source: [https://huggingface.co/datasets/ChanceFocus/flare-finer-ord](https://huggingface.co/datasets/ChanceFocus/flare-finer-ord)  
13. **FinRED (Financial Relation Extraction)**  
    **Description:** Extracts relationships (ownership, acquisition) between entities in financial texts, supporting knowledge graph construction.  
    **Metrics:** F1-Score, Entity F1-Score							Source: [https://huggingface.co/datasets/ChanceFocus/flare-finred](https://huggingface.co/datasets/ChanceFocus/flare-finred)  
14. **SC (Causal Classification)**  
    **Description:** Classifies causal relationships in financial texts (e.g., "X caused Y"), aiding in market risk assessments.  
    **Metrics:** F1-Score, Entity F1-Score							Source: [https://huggingface.co/datasets/ChanceFocus/flare-causal20-sc](https://huggingface.co/datasets/ChanceFocus/flare-causal20-sc)  
15. **CD (Causal Detection)**  
    **Description:** Detects causal relationships in financial texts, helping in risk analysis and investment strategies.  
    **Metrics:** F1-Score, Entity F1-Score							Source: [https://huggingface.co/datasets/ChanceFocus/flare-cd](https://huggingface.co/datasets/ChanceFocus/flare-cd)  
16. **FinQA (Numerical Question Answering in Finance)**  
    **Description:** Answers numerical questions from financial documents (e.g., balance sheets), crucial for automated reporting and analysis.  
    **Metrics:** Exact Match Accuracy (EmAcc)						Source: [https://huggingface.co/datasets/ChanceFocus/flare-finqa](https://huggingface.co/datasets/ChanceFocus/flare-finqa)  
17. **TATQA (Table-Based Question Answering)**  
    **Description:** Extracts information from financial tables (balance sheets, income statements) to answer queries requiring numerical reasoning.  
    **Metrics:** F1-Score, EmAcc								Source: [https://huggingface.co/datasets/ChanceFocus/flare-tatqa](https://huggingface.co/datasets/ChanceFocus/flare-tatqa)  
18. **ConvFinQA (Multi-Turn QA in Finance)**  
    **Description:** Handles multi-turn dialogues in financial question answering, maintaining context throughout the conversation.  
    **Metrics:** EmAcc									Source: [https://huggingface.co/datasets/ChanceFocus/flare-convfinqa](https://huggingface.co/datasets/ChanceFocus/flare-convfinqa)  
19. **FNXL (Numeric Labeling)**  
    **Description:** Labels numeric values in financial documents (e.g., revenue, expenses), aiding in financial data extraction.  
    **Metrics:** F1-Score, EmAcc								Source: [https://huggingface.co/datasets/ChanceFocus/flare-fnxl](https://huggingface.co/datasets/ChanceFocus/flare-fnxl)  
20. **FSRL (Financial Statement Relation Linking)**  
    **Description:** Links related information across financial statements (e.g., revenue in income statements and cash flow data).  
    **Metrics:** F1-Score, EmAcc								Source: [https://huggingface.co/datasets/ChanceFocus/flare-fsrl](https://huggingface.co/datasets/ChanceFocus/flare-fsrl)  
21. **EDTSUM (Extractive Document Summarization)**  
    **Description:** Summarizes long financial documents, extracting key information for decision-making.  
    **Metrics:** ROUGE, BERTScore, BARTScore						Source: [https://huggingface.co/datasets/ChanceFocus/flare-edtsum](https://huggingface.co/datasets/ChanceFocus/flare-edtsum)  
22. **ECTSUM (Extractive Content Summarization)**  
    **Description:** Summarizes financial content, extracting key sentences or phrases from large texts.  
    **Metrics:** ROUGE, BERTScore, BARTScore							Source: [https://huggingface.co/datasets/ChanceFocus/flare-ectsum](https://huggingface.co/datasets/ChanceFocus/flare-ectsum)  
23. **BigData22 (Stock Movement Prediction)**  
    **Description:** Predicts stock price movements based on financial news, using textual data to forecast market trends.  
    **Metrics:** Accuracy, MCC								Source: [https://huggingface.co/datasets/TheFinAI/en-forecasting-bigdata](https://huggingface.co/datasets/TheFinAI/en-forecasting-bigdata)  
24. **ACL18 (Financial News-Based Stock Prediction)**  
    **Description:** Predicts stock price movements from news articles, interpreting sentiment and events for short-term forecasts.  
    **Metrics:** Accuracy, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/flare-sm-acl](https://huggingface.co/datasets/ChanceFocus/flare-sm-acl)  
25. **CIKM18 (Financial Market Prediction Using News)**  
    **Description:** Predicts broader market movements (indices) from financial news, synthesizing information for market trend forecasts.  
    **Metrics:** Accuracy, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/flare-sm-cikm](https://huggingface.co/datasets/ChanceFocus/flare-sm-cikm)  
26. **German (Credit Scoring in Germany)**  
    **Description:** Predicts creditworthiness of loan applicants in Germany, important for responsible lending and risk management.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/flare-german](https://huggingface.co/datasets/ChanceFocus/flare-german)  
27. **Australian (Credit Scoring in Australia)**  
    **Description:** Predicts creditworthiness in the Australian market, considering local economic conditions.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/flare-australian](https://huggingface.co/datasets/ChanceFocus/flare-australian)  
28. **LendingClub (Peer-to-Peer Lending Risk Prediction)**  
    **Description:** Predicts loan default risk for peer-to-peer lending, helping lenders manage risk.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/cra-lendingclub](https://huggingface.co/datasets/ChanceFocus/cra-lendingclub)  
29. **ccf (Credit Card Fraud Detection)**  
    **Description:** Identifies fraudulent credit card transactions, ensuring financial security and fraud prevention.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/cra-ccf](https://huggingface.co/datasets/ChanceFocus/cra-ccf)  
30. **ccfraud (Credit Card Transaction Fraud Detection)**  
    **Description:** Detects anomalies in credit card transactions that indicate fraud, while handling imbalanced datasets.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/cra-ccfraud](https://huggingface.co/datasets/ChanceFocus/cra-ccfraud)  
31. **Polish (Credit Risk Prediction in Poland)**  
    **Description:** Predicts credit risk for loan applicants in Poland, assessing factors relevant to local economic conditions.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/ChanceFocus/cra-polish](https://huggingface.co/datasets/ChanceFocus/cra-polish)  
32. **Taiwan (Credit Risk Prediction in Taiwan)**  
    **Description:** Predicts credit risk in the Taiwanese market, helping lenders manage risk in local contexts.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/TheFinAI/cra-taiwan](https://huggingface.co/datasets/TheFinAI/cra-taiwan)  
33. **Portoseguro (Claim Analysis in Brazil)**  
    **Description:** Predicts the outcome of insurance claims in Brazil, focusing on auto insurance claims.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/TheFinAI/en-forecasting-portoseguro](https://huggingface.co/datasets/TheFinAI/en-forecasting-portoseguro)  
34. **Travel Insurance (Claim Prediction)**  
    **Description:** Predicts the likelihood of travel insurance claims, helping insurers manage pricing and risk.  
    **Metrics:** F1-Score, MCC								Source: [https://huggingface.co/datasets/TheFinAI/en-forecasting-travelinsurance](https://huggingface.co/datasets/TheFinAI/en-forecasting-travelinsurance)  
35. **MultiFin-ES (Sentiment Analysis in Spanish)**  
    **Description:** Classifies sentiment in Spanish-language financial texts (bullish, bearish, neutral).  
    **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/ChanceFocus/flare-es-multifin](https://huggingface.co/datasets/ChanceFocus/flare-es-multifin)  
36. **EFP (Financial Phrase Classification in Spanish)**  
    **Description:** Classifies sentiment or intent in Spanish financial phrases (positive, negative, neutral).  
    **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/ChanceFocus/flare-es-efp](https://huggingface.co/datasets/ChanceFocus/flare-es-efp)  
37. **EFPA (Argument Classification in Spanish)**  
    **Description:** Identifies claims, evidence, and counterarguments in Spanish financial texts.  
    **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/ChanceFocus/flare-es-efpa](https://huggingface.co/datasets/ChanceFocus/flare-es-efpa)  
38. **FinanceES (Sentiment Classification in Spanish)**  
    **Description:** Classifies sentiment in Spanish financial documents, understanding linguistic nuances.  
    **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/ChanceFocus/flare-es-financees](https://huggingface.co/datasets/ChanceFocus/flare-es-financees)  
39. **TSA-Spanish (Sentiment Analysis in Spanish Tweets)**  
    **Description:** Sentiment analysis of Spanish tweets, interpreting informal language in real-time market discussions.  
    **Metrics:** F1-Score									Source: [https://huggingface.co/datasets/TheFinAI/flare-es-tsa](https://huggingface.co/datasets/TheFinAI/flare-es-tsa)  
40. **FinTrade (Stock Trading Simulation)**  
    **Description:** Evaluates models on stock trading simulations, analyzing historical stock prices and financial news to optimize trading outcomes.  
    **Metrics:** Sharpe Ratio (SR)								Source: [https://huggingface.co/datasets/TheFinAI/FinTrade\_train](https://huggingface.co/datasets/TheFinAI/FinTrade_train)  

## How to Use the Open Financial LLM Leaderboard

When you first visit the OFLL platform, you are greeted by the **main page**, which provides an overview of the leaderboard, including an introduction to the platform's purpose and a link to submit your model for evaluation.

At the top of the main page, you'll see different tabs:

* **LLM Benchmark:** The core page where you evaluate models.  
* **Submit here:** A place to submit your own model for automatic evaluation.  
* **About:** More details about the benchmarks, evaluation process, and the datasets used.

### Selecting Tasks to Display

To tailor the leaderboard to your specific needs, you can select the financial tasks you want to focus on under the **"Select columns to show"** section. These tasks are divided into several categories, such as:

* **Information Extraction (IE)**  
* **Textual Analysis (TA)**  
* **Question Answering (QA)**  
* **Text Generation (TG)**  
* **Risk Management (RM)**  
* **Forecasting (FO)**  
* **Decision-Making (DM)**

Simply check the box next to the tasks you're interested in. The selected tasks will appear as columns in the evaluation table. If you wish to remove all selections, click the **"Uncheck All"** button to reset the task categories.

### Selecting Models to Display

To further refine the models displayed in the leaderboard, you can use the **"Model types"** and **"Precision"** filters on the right-hand side of the interface, and filter models based on their:

* **Type:** Pretrained, fine-tuned, instruction-tuned, or reinforcement-learning (RL)-tuned.  
* **Precision:** float16, bfloat16, or float32.  
* **Model Size:** Ranges from \~1.5 billion to 70+ billion parameters.

### Viewing Results in the Task Table

Once you've selected your tasks, the results will populate in the **task table** (see image). This table provides detailed metrics for each model across the tasks you’ve chosen. The performance of each model is displayed under columns labeled **Average IE**, **Average TA**, **Average QA**, and so on, corresponding to the tasks you selected.

### Submitting a Model for Evaluation

If you have a new model that you’d like to evaluate on the leaderboard, the **submission section** allows you to upload your model file. You’ll need to provide:

* **Model name**  
* **Revision commit**  
* **Model type**  
* **Precision**  
* **Weight type**

After uploading your model, the leaderboard will **automatically start evaluating** it across the selected tasks, providing real-time feedback on its performance.    

## Current Best Models and Surprising Results

Throughout the evaluation process on the Open FinLLM Leaderboard, several models have demonstrated exceptional capabilities across various financial tasks. 

As of the latest evaluation:
- **Best model**: GPT-4 and Llama3.1 have consistently outperformed other models in many tasks, showing high accuracy and robustness in interpreting financial sentiment.  
- **Surprising Results**: The **Forecasting(FO)** task, focused on stock movement predictions, showed that smaller models, such as **Llama3.1-7b, internlm-7b**,often outperformed larger models, for example Llama3.1-70b, in terms of accuracy and MCC. This suggests that model size does not necessarily correlate with better performance in financial forecasting, especially in tasks where real-time market data and nuanced sentiment analysis are critical. These results highlight the importance of evaluating models based on task-specific performance rather than relying solely on size or general-purpose benchmarks.


## **Acknowledgments**

We would like to thank our sponsors, including The Linux Foundation, for their generous support in making the Open FinLLM Leaderboard possible. Their contributions have helped us build a platform that serves the financial AI community and advances the evaluation of financial language models.

We also invite the community to participate in this ongoing project by submitting models, datasets, or evaluation tasks. Your involvement is essential in ensuring that the leaderboard remains a comprehensive and evolving tool for benchmarking financial LLMs. Together, we can drive innovation and help develop models better suited for real-world financial applications.