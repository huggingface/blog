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

 <details><summary><b>Click here for a short explanation of each task</b></summary> 
   
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
</details>

 <details><summary><b>Click here for a detailed explanation of each task</b></summary> 

This section will document each task within the categories in more detail, explaining the specific datasets, evaluation metrics, and financial relevance.

1. **FPB (Financial PhraseBank Sentiment Classification)**

   * **Task Description.** In this task, we evaluate a language model's ability to perform sentiment analysis on financial texts. We employ the Financial PhraseBank dataset1, which consists of annotated phrases extracted from financial news articles and reports. Each phrase is labeled with one of three sentiment categories: positive, negative, or neutral. The dataset provides a nuanced understanding of sentiments expressed in financial contexts, making it essential for applications such as market sentiment analysis and automated trading strategies. The primary objective is to classify each financial phrase accurately according to its sentiment. Example inputs, outputs, and the prompt templates used in this task are detailed in Table 5 and Table 8 in the Appendix.  
   * **Metric.** Accuracy, F1-score. 

2. **FiQA-SA (Sentiment Analysis on FiQA Financial Domain)**

   * **Task Description.** The FiQA-SA task evaluates a language model's capability to perform sentiment analysis within the financial domain, particularly focusing on data derived from the FiQA dataset. This dataset includes a diverse collection of financial texts sourced from various media, including news articles, financial reports, and social media posts. The primary objective of the task is to classify sentiments expressed in these texts into distinct categories, such as positive, negative, and neutral. This classification is essential for understanding market sentiment, as it can directly influence investment decisions and strategies. The FiQA-SA task is particularly relevant in today's fast-paced financial environment, where the interpretation of sentiment can lead to timely and informed decision-making.   
   * **Metrics.** F1 Score.

3. **TSA (Sentiment Analysis on Social Media)**

   * **Task Description.** The TSA task focuses on evaluating a model's ability to perform sentiment analysis on tweets related to financial markets. Utilizing a dataset comprised of social media posts, this task seeks to classify sentiments as positive, negative, or neutral. The dynamic nature of social media makes it a rich source of real-time sentiment data, reflecting public opinion on market trends, company news, and economic events. The TSA dataset includes a wide variety of tweets, featuring diverse expressions of sentiment related to financial topics, ranging from stock performance to macroeconomic indicators. Given the brevity and informal nature of tweets, this task presents unique challenges in accurately interpreting sentiment, as context and subtleties can significantly impact meaning. Therefore, effective models must demonstrate robust understanding and analysis of informal language, slang, and sentiment indicators commonly used on social media platforms.  
   * **Metrics.** F1 Score, RMSE. RMSE provides a measure of the average deviation between predicted and actual sentiment scores, offering a quantitative insight into the accuracy of the model's predictions. 

4. **Headlines (News Headline Classification)**

   * **Task Description.** The Headlines task involves classifying financial news headlines into various categories, reflecting distinct financial events or sentiment classes. This dataset consists of a rich collection of headlines sourced from reputable financial news outlets, capturing a wide array of topics ranging from corporate earnings reports to market forecasts. The primary objective of this task is to evaluate a model's ability to accurately interpret and categorize brief, context-rich text segments that often drive market movements. Given the succinct nature of headlines, the classification task requires models to quickly grasp the underlying sentiment and relevance of each headline, which can significantly influence investor behavior and market sentiment.   
   * **Metrics.** Average F1 Score (AvgF1). This metric provides a balanced measure of precision and recall, allowing for a nuanced understanding of the model’s performance in classifying headlines. A high AvgF1 indicates that the model is effectively identifying and categorizing the sentiment and events reflected in the headlines, making it a critical metric for assessing its applicability in real-world financial contexts.

5. **FOMC (Hawkish-Dovish Classification)**

   * **Task Description.** The FOMC task evaluates a model's ability to classify statements derived from transcripts of Federal Open Market Committee (FOMC) meetings as either hawkish or dovish. Hawkish statements typically indicate a preference for higher interest rates to curb inflation, while dovish statements suggest a focus on lower rates to stimulate economic growth. This classification is crucial for understanding monetary policy signals that can impact financial markets and investment strategies. The dataset includes a range of statements from FOMC meetings, providing insights into the Federal Reserve's stance on economic conditions, inflation, and employment. Accurately categorizing these statements allows analysts and investors to anticipate market reactions and adjust their strategies accordingly, making this task highly relevant in the context of financial decision-making.  
   * **Metrics.** F1 Score and Accuracy. 

6. **FinArg-ACC (Financial Argument Unit Classification)**

   * **Task Description.** The FinArg-ACC task focuses on classifying argument units within financial documents, aiming to identify key components such as main claims, supporting evidence, and counterarguments. This dataset comprises a diverse collection of financial texts, including research reports, investment analyses, and regulatory filings. The primary objective is to assess a model's ability to dissect complex financial narratives into distinct argument units, which is crucial for automated financial document analysis. This task is particularly relevant in the context of increasing regulatory scrutiny and the need for transparency in financial communications, where understanding the structure of arguments can aid in compliance and risk management.  
   * **Metrics.** F1 Score, Accuracy. 

7. **FinArg-ARC (Financial Argument Relation Classification)**

   * **Task Description.** The FinArg-ARC task focuses on classifying relationships between different argument units within financial texts. This involves identifying how various claims, evidence, and counterarguments relate to each other, such as support, opposition, or neutrality. The dataset comprises annotated financial documents that highlight argument structures, enabling models to learn the nuances of financial discourse. Understanding these relationships is crucial for constructing coherent narratives and analyses from fragmented data, which can aid financial analysts, investors, and researchers in drawing meaningful insights from complex information. Given the intricate nature of financial arguments, effective models must demonstrate proficiency in discerning subtle distinctions in meaning and context, which are essential for accurate classification.  
   * **Metrics.** F1 Score, Accuracy

8. **MultiFin (Multi-Class Financial Sentiment Analysis)**

   * **Task Description.** The MultiFin task focuses on the classification of sentiments expressed in a diverse array of financial texts into multiple categories, such as bullish, bearish, or neutral. This dataset includes various financial documents, ranging from reports and articles to social media posts, providing a comprehensive view of sentiment across different sources and contexts. The primary objective of this task is to assess a model's ability to accurately discern and categorize sentiments that influence market behavior and investor decisions. Models must demonstrate a robust understanding of contextual clues and varying tones inherent in financial discussions. The MultiFin task is particularly valuable for applications in sentiment-driven trading strategies and market analysis, where precise sentiment classification can lead to more informed investment choices.  
   * **Metrics.** F1 Score, Accuracy. 

9. **MA (Deal Completeness Classification)**

   * **Task Description:**  
     The MA task focuses on classifying mergers and acquisitions (M\&A) reports to determine whether a deal has been completed. This dataset comprises a variety of M\&A announcements sourced from financial news articles, press releases, and corporate filings. The primary objective is to accurately identify the status of each deal—categorized as completed, pending, or terminated—based on the information presented in the reports. This classification is crucial for investment analysts and financial institutions, as understanding the completion status of M\&A deals can significantly influence investment strategies and market reactions. Models must demonstrate a robust understanding of the M\&A landscape and the ability to accurately classify deal statuses based on often complex and evolving narratives.  
   * **Metrics:**  
     F1 Score, Accuracy.

10. **MLESG (ESG Issue Identification)**

    * **Task Description:**  
      The MLESG task focuses on identifying Environmental, Social, and Governance (ESG) issues within financial texts. This dataset is specifically designed to capture a variety of texts, including corporate reports, news articles, and regulatory filings, that discuss ESG topics. The primary objective of the task is to evaluate a model's ability to accurately classify and categorize ESG-related content, which is becoming increasingly important in today's investment landscape. Models are tasked with detecting specific ESG issues, such as climate change impacts, social justice initiatives, or corporate governance practices. Models must demonstrate a deep understanding of the language used in these contexts, as well as the ability to discern subtle variations in meaning and intent.  
    * **Metrics:**  
      F1 Score, Accuracy.

11. **NER (Named Entity Recognition in Financial Texts)**

    * **Task Description:**  
      The NER task focuses on identifying and classifying named entities within financial documents, such as companies, financial instruments, and individuals. This task utilizes a dataset that includes a diverse range of financial texts, encompassing regulatory filings, earnings reports, and news articles. The primary objective is to accurately recognize entities relevant to the financial domain and categorize them appropriately, which is crucial for information extraction and analysis. Effective named entity recognition enhances the automation of financial analysis processes, allowing stakeholders to quickly gather insights from large volumes of unstructured text.   
    * **Metrics:**  
      Entity F1 Score (EntityF1). 

12. **FINER-ORD (Ordinal Classification in Financial NER)**

    * **Task Description:**  
      The FINER-ORD task focuses on extending standard Named Entity Recognition (NER) by requiring models to classify entities not only by type but also by their ordinal relevance within financial texts. This dataset comprises a range of financial documents that include reports, articles, and regulatory filings, where entities such as companies, financial instruments, and events are annotated with an additional layer of classification reflecting their importance or priority. The primary objective is to evaluate a model’s ability to discern and categorize entities based on their significance in the context of the surrounding text. For instance, a model might identify a primary entity (e.g., a major corporation) as having a higher relevance compared to secondary entities (e.g., a minor competitor) mentioned in the same document. This capability is essential for prioritizing information and enhancing the efficiency of automated financial analyses, where distinguishing between varying levels of importance can significantly impact decision-making processes.  
    * **Metrics:**  
      Entity F1 Score (EntityF1). 

13. **FinRED (Financial Relation Extraction from Text)**

    * **Task Description:**  
      The FinRED task focuses on extracting relationships between financial entities mentioned in textual data. This task utilizes a dataset that includes diverse financial documents, such as news articles, reports, and regulatory filings. The primary objective is to identify and classify relationships such as ownership, acquisition, and partnership among various entities, such as companies, financial instruments, and stakeholders. Accurately extracting these relationships is crucial for building comprehensive knowledge graphs and facilitating in-depth financial analysis. The challenge lies in accurately interpreting context, as the relationships often involve nuanced language and implicit connections that require a sophisticated understanding of financial terminology.  
    * **Metrics:**  
      F1 Score, Entity F1 Score (EntityF1). 

14. **SC (Causal Classification Task in the Financial Domain)**

    * **Task Description:**  
      The SC task focuses on evaluating a language model's ability to classify causal relationships within financial texts. This involves identifying whether one event causes another, which is crucial for understanding dynamics in financial markets. The dataset used for this task encompasses a variety of financial documents, including reports, articles, and regulatory filings, where causal language is often embedded. By examining phrases that express causality—such as "due to," "resulting in," or "leads to"—models must accurately determine the causal links between financial events, trends, or phenomena. This task is particularly relevant for risk assessment, investment strategy formulation, and decision-making processes, as understanding causal relationships can significantly influence evaluations of market conditions and forecasts.  
    * **Metrics:**  
      F1 Score, Entity F1 Score (EntityF1). 

15. **CD (Causal Detection)**

    * **Task Description:**  
      The CD task focuses on detecting causal relationships within a diverse range of financial texts, including reports, news articles, and social media posts. This task evaluates a model's ability to identify instances where one event influences or causes another, which is crucial for understanding dynamics in financial markets. The dataset comprises annotated examples that explicitly highlight causal links, allowing models to learn from various contexts and expressions of causality. Detecting causality is essential for risk assessment, as it helps analysts understand potential impacts of events on market behavior, investment strategies, and decision-making processes. Models must navigate nuances and subtleties in text to accurately discern causal connections.  
    * **Metrics:**  
      F1 Score, Entity F1 Score (EntityF1). 

16. **FinQA (Numerical Question Answering in Finance)**

    * **Task Description:**  
      The FinQA task evaluates a model's ability to answer numerical questions based on financial documents, such as balance sheets, income statements, and financial reports. This dataset includes a diverse set of questions that require not only comprehension of the text but also the ability to extract and manipulate numerical data accurately. The primary goal is to assess how well a model can interpret complex financial information and perform necessary calculations to derive answers. The FinQA task is particularly relevant for applications in financial analysis, investment decision-making, and automated reporting, where precise numerical responses are essential for stakeholders.  
    * **Metrics:**  
      Exact Match Accuracy (EmAcc)

17. **TATQA (Table-Based Question Answering in Financial Documents)**

    * **Task Description:**  
      The TATQA task focuses on evaluating a model's ability to answer questions that require interpreting and extracting information from tables in financial documents. This dataset is specifically designed to include a variety of financial tables, such as balance sheets, income statements, and cash flow statements, each containing structured data critical for financial analysis. The primary objective of this task is to assess how well models can navigate these tables to provide accurate and relevant answers to questions that often demand numerical reasoning or domain-specific knowledge. Models must demonstrate proficiency in not only locating the correct data but also understanding the relationships between different data points within the context of financial analysis.   
    * **Metrics:**  
      F1 Score, Exact Match Accuracy (EmAcc).

18. **ConvFinQA (Multi-Turn Question Answering in Finance)**

    * **Task Description:**  
      The ConvFinQA task focuses on evaluating a model's ability to handle multi-turn question answering in the financial domain. This task simulates real-world scenarios where financial analysts engage in dialogues, asking a series of related questions that build upon previous answers. The dataset includes conversations that reflect common inquiries regarding financial data, market trends, and economic indicators, requiring the model to maintain context and coherence throughout the dialogue. The primary objective is to assess the model's capability to interpret and respond accurately to multi-turn queries, ensuring that it can provide relevant and precise information as the conversation progresses. This task is particularly relevant in financial advisory settings, where analysts must extract insights from complex datasets while engaging with clients or stakeholders.  
    * **Metrics:**  
      Exact Match Accuracy (EmAcc). 

19. **FNXL (Numeric Labeling in Financial Texts)**

    * **Task Description:**  
      The FNXL task focuses on the identification and categorization of numeric values within financial documents. This involves labeling numbers based on their type (e.g., revenue, profit, expense) and their relevance in the context of the text. The dataset used for this task includes a diverse range of financial reports, statements, and analyses, presenting various numeric expressions that are crucial for understanding financial performance. Accurate numeric labeling is essential for automating financial analysis and ensuring that critical data points are readily accessible for decision-making. Models must demonstrate a robust ability to parse context and semantics to accurately classify numeric information, thereby enhancing the efficiency of financial data processing.  
    * **Metrics:**  
      F1 Score, Exact Match Accuracy (EmAcc).

20. **FSRL (Financial Statement Relation Linking)**

    * **Task Description:**  
      The FSRL task focuses on linking related information across different financial statements, such as matching revenue figures from income statements with corresponding cash flow data. This task is crucial for comprehensive financial analysis, enabling models to synthesize data from multiple sources to provide a coherent understanding of a company's financial health. The dataset used for this task includes a variety of financial statements from publicly traded companies, featuring intricate relationships between different financial metrics. Accurate linking of this information is essential for financial analysts and investors who rely on holistic views of financial performance. The task requires models to navigate the complexities of financial terminology and understand the relationships between various financial elements, ensuring they can effectively connect relevant data points.  
    * **Metrics:**  
      F1 Score, Exact Match Accuracy (EmAcc). 

21. **EDTSUM (Extractive Document Summarization in Finance)**

    * **Task Description:**  
      The EDTSUM task focuses on summarizing lengthy financial documents by extracting the most relevant sentences to create concise and coherent summaries. This task is essential in the financial sector, where professionals often deal with extensive reports, research papers, and regulatory filings. The ability to distill critical information from large volumes of text is crucial for efficient decision-making and information dissemination. The EDTSUM dataset consists of various financial documents, each paired with expert-generated summaries that highlight key insights and data points. Models are evaluated on their capability to identify and select sentences that accurately reflect the main themes and arguments presented in the original documents.   
    * **Metrics:**  
      ROUGE, BERTScore, and BARTScore.

22. **ECTSUM (Extractive Content Summarization)**

    * **Task Description:**  
      The ECTSUM task focuses on extractive content summarization within the financial domain, where the objective is to generate concise summaries from extensive financial documents. This task leverages a dataset that includes a variety of financial texts, such as reports, articles, and regulatory filings, each containing critical information relevant to stakeholders. The goal is to evaluate a model’s ability to identify and extract the most salient sentences or phrases that encapsulate the key points of the original text. The ECTSUM task challenges models to demonstrate their understanding of context, relevance, and coherence, ensuring that the extracted summaries accurately represent the main ideas while maintaining readability and clarity.  
    * **Metrics:**  
      ROUGE, BERTScore, and BARTScore.

23. **BigData22 (Stock Movement Prediction)**

    * **Task Description:**  
      The BigData22 task focuses on predicting stock price movements based on financial news and reports. This dataset is designed to capture the intricate relationship between market sentiment and stock performance, utilizing a comprehensive collection of news articles, social media posts, and market data. The primary goal of this task is to evaluate a model's ability to accurately forecast whether the price of a specific stock will increase or decrease within a defined time frame. Models must effectively analyze textual data and discern patterns that correlate with market movements.   
    * **Metrics:**  
      Accuracy, Matthews Correlation Coefficient (MCC).

24. **ACL18 (Financial News-Based Stock Prediction)**

    * **Task Description:**  
      The ACL18 task focuses on predicting stock movements based on financial news articles and headlines. Utilizing a dataset that includes a variety of news pieces, this task aims to evaluate a model's ability to analyze textual content and forecast whether stock prices will rise or fall in the near term. The dataset encompasses a range of financial news topics, from company announcements to economic indicators, reflecting the complex relationship between news sentiment and market reactions. Models must effectively interpret nuances in language and sentiment that can influence stock performance, ensuring that predictions align with actual market movements.  
    * **Metrics:**  
      Accuracy, Matthews Correlation Coefficient (MCC).

25. **CIKM18 (Financial Market Prediction Using News)**

    * **Task Description:**  
      The CIKM18 task focuses on predicting broader market movements, such as stock indices, based on financial news articles. Utilizing a dataset that encompasses a variety of news stories related to market events, this task evaluates a model's ability to synthesize information from multiple sources and make informed predictions about future market trends. The dataset includes articles covering significant financial events, economic indicators, and company news, reflecting the complex interplay between news sentiment and market behavior. The objective of this task is to assess how well a model can analyze the content of financial news and utilize that analysis to forecast market movements.   
    * **Metrics:**  
      Accuracy, Matthews Correlation Coefficient (MCC).

26. **German (Credit Scoring in the German Market)**

    * **Task Description:**  
      The German task focuses on evaluating a model's ability to predict creditworthiness among loan applicants within the German market. Utilizing a dataset that encompasses various financial indicators, demographic information, and historical credit data, this task aims to classify applicants as either creditworthy or non-creditworthy. The dataset reflects the unique economic and regulatory conditions of Germany, providing a comprehensive view of the factors influencing credit decisions in this specific market. Given the importance of accurate credit scoring for financial institutions, this task is crucial for minimizing risk and ensuring responsible lending practices. Models must effectively analyze multiple variables to make informed predictions, thereby facilitating better decision-making in loan approvals and risk management.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

27. **Australian (Credit Scoring in the Australian Market)**

    * **Task Description:**  
      The Australian task focuses on predicting creditworthiness among loan applicants within the Australian financial context. This dataset includes a comprehensive array of features derived from various sources, such as financial histories, income levels, and demographic information. The primary objective of this task is to classify applicants as either creditworthy or non-creditworthy, enabling financial institutions to make informed lending decisions. Given the unique economic conditions and regulatory environment in Australia, this task is particularly relevant for understanding the specific factors that influence credit scoring in this market.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

28. **LendingClub (Peer-to-Peer Lending Risk Prediction)**

    * **Task Description:**  
      The LendingClub task focuses on predicting the risk of default for loans issued through the LendingClub platform, a major peer-to-peer lending service. This task utilizes a dataset that includes detailed information about loan applicants, such as credit scores, income levels, employment history, and other financial indicators. The primary objective is to assess the likelihood of loan default, enabling lenders to make informed decisions regarding loan approvals and risk management. The models evaluated in this task must effectively analyze a variety of features, capturing complex relationships within the data to provide reliable risk assessments.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

29. **ccf (Credit Card Fraud Detection)**

    * **Task Description:**  
      The ccf task focuses on identifying fraudulent transactions within a large dataset of credit card operations. This dataset encompasses various transaction features, including transaction amount, time, location, and merchant information, providing a comprehensive view of spending behaviors. The primary objective of the task is to classify transactions as either legitimate or fraudulent, thereby enabling financial institutions to detect and prevent fraudulent activities effectively. Models must navigate the challenges posed by class imbalance, as fraudulent transactions typically represent a small fraction of the overall dataset.   
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

30. **ccfraud (Credit Card Transaction Fraud Detection)**

    * **Task Description:**  
      The ccfraud task focuses on identifying fraudulent transactions within a dataset of credit card operations. This dataset comprises a large number of transaction records, each labeled as either legitimate or fraudulent. The primary objective is to evaluate a model's capability to accurately distinguish between normal transactions and those that exhibit suspicious behavior indicative of fraud. The ccfraud task presents unique challenges, including the need to handle imbalanced data, as fraudulent transactions typically represent a small fraction of the total dataset. Models must demonstrate proficiency in detecting subtle patterns and anomalies that signify fraudulent activity while minimizing false positives to avoid inconveniencing legitimate customers.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

31. **Polish (Credit Risk Prediction in the Polish Market)**

    * **Task Description:**  
      The Polish task focuses on predicting credit risk for loan applicants within the Polish financial market. Utilizing a comprehensive dataset that includes demographic and financial information about applicants, the task aims to assess the likelihood of default on loans. This prediction is crucial for financial institutions in making informed lending decisions and managing risk effectively. Models must be tailored to account for local factors influencing creditworthiness, such as income levels, employment status, and credit history.   
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

32. **Taiwan (Credit Risk Prediction in the Taiwanese Market)**

    * **Task Description:**  
      The Taiwan task focuses on predicting credit risk for loan applicants in the Taiwanese market. Utilizing a dataset that encompasses detailed financial and personal information about borrowers, this task aims to assess the likelihood of default based on various factors, including credit history, income, and demographic details. The model's ability to analyze complex patterns within the data and provide reliable predictions is essential in a rapidly evolving financial landscape. Given the unique economic conditions and regulatory environment in Taiwan, this task also emphasizes the importance of local context in risk assessment, requiring models to effectively adapt to specific market characteristics and trends.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

33. **Portoseguro (Claim Analysis in the Brazilian Market)**

    * **Task Description:**  
      The Portoseguro task focuses on analyzing insurance claims within the Brazilian market, specifically for auto insurance. This task leverages a dataset that includes detailed information about various claims, such as the nature of the incident, policyholder details, and claim outcomes. The primary goal is to evaluate a model’s ability to predict the likelihood of a claim being approved or denied based on these factors. By accurately classifying claims, models can help insurance companies streamline their decision-making processes, enhance risk management strategies, and reduce fraudulent activities. Models must consider regional nuances and the specific criteria used in evaluating claims, ensuring that predictions align with local regulations and market practices.  
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

34. **Travel Insurance (Travel Insurance Claim Prediction)**

    * **Task Description:**  
      The Travel Insurance task focuses on predicting the likelihood of a travel insurance claim being made based on various factors and data points. This dataset includes historical data related to travel insurance policies, claims made, and associated variables such as the type of travel, duration, destination, and demographic information of the insured individuals. The primary objective of this task is to evaluate a model's ability to accurately assess the risk of a claim being filed, which is crucial for insurance companies in determining policy pricing and risk management strategies. By analyzing patterns and trends in the data, models can provide insights into which factors contribute to a higher likelihood of claims, enabling insurers to make informed decisions about underwriting and premium setting.   
    * **Metrics:**  
      F1 Score, Matthews Correlation Coefficient (MCC).

35. **MultiFin-ES (Multi-Class Financial Sentiment Analysis in Spanish)**

    * **Task Description:**  
      The MultiFin-ES task focuses on analyzing  
    *  sentiment in Spanish-language financial texts, categorizing sentiments into multiple classes such as bullish, bearish, and neutral. This dataset includes a diverse array of financial documents, including news articles, reports, and social media posts, reflecting various aspects of the financial landscape. The primary objective is to evaluate a model's ability to accurately classify sentiments based on contextual cues, linguistic nuances, and cultural references prevalent in Spanish financial discourse. Models must demonstrate proficiency in processing the subtleties of the Spanish language, including idiomatic expressions and regional variations, to achieve accurate classifications.  
    * **Metrics:**  
      F1 Score.

36. **EFP (Financial Phrase Classification in Spanish)**

    * **Task Description:**  
      The EFP task focuses on the classification of financial phrases in Spanish, utilizing a dataset specifically designed for this purpose. This dataset consists of a collection of annotated phrases extracted from Spanish-language financial texts, including news articles, reports, and social media posts. The primary objective is to classify these phrases based on sentiment or intent, categorizing them into relevant classifications such as positive, negative, or neutral. Given the growing importance of the Spanish-speaking market in global finance, accurately interpreting and analyzing sentiment in Spanish financial communications is essential for investors and analysts.   
    * **Metrics:**  
      F1 Score.

37. **EFPA (Financial Argument Classification in Spanish)**

    * **Task Description:**  
      The EFPA task focuses on classifying arguments within Spanish financial documents, aiming to identify key components such as claims, evidence, and counterarguments. This dataset encompasses a range of financial texts, including reports, analyses, and regulatory documents, providing a rich resource for understanding argumentative structures in the financial domain. The primary objective is to evaluate a model's ability to accurately categorize different argument units, which is essential for automating the analysis of complex financial narratives. By effectively classifying arguments, stakeholders can gain insights into the reasoning behind financial decisions and the interplay of various factors influencing the market. This task presents unique challenges that require models to demonstrate a deep understanding of both linguistic and domain-specific contexts.  
    * **Metrics:**  
      F1 Score.

38. **FinanceES (Financial Sentiment Classification in Spanish)**

    * **Task Description:**  
      The FinanceES task focuses on classifying sentiment within a diverse range of financial documents written in Spanish. This dataset includes news articles, reports, and social media posts, reflecting various financial topics and events. The primary objective is to evaluate a model's ability to accurately identify sentiments as positive, negative, or neutral, thus providing insights into market perceptions in Spanish-speaking regions. Given the cultural and linguistic nuances inherent in the Spanish language, effective sentiment classification requires models to adeptly navigate idiomatic expressions, slang, and context-specific terminology. This task is particularly relevant as financial sentiment analysis expands globally, necessitating robust models that can perform effectively across different languages and cultural contexts.  
    * **Metrics:**  
      F1 Score.

39. **TSA-Spanish (Sentiment Analysis in Spanish)**

    * **Task Description:**  
      The TSA-Spanish task focuses on evaluating a model's ability to perform sentiment analysis on tweets and short texts in Spanish related to financial markets. Utilizing a dataset comprised of diverse social media posts, this task aims to classify sentiments as positive, negative, or neutral. The dynamic nature of social media provides a rich source of real-time sentiment data, reflecting public opinion on various financial topics, including stock performance, company announcements, and economic developments. This task presents unique challenges in accurately interpreting sentiment, as context, slang, and regional expressions can significantly influence meaning. Models must demonstrate a robust understanding of the subtleties of the Spanish language, including colloquialisms and varying sentiment indicators commonly used across different Spanish-speaking communities.  
    * **Metrics:**  
      F1 Score.

40. **FinTrade (Stock Trading Dataset)**

    * **Task Description:**  
      The FinTrade task evaluates models on their ability to perform stock trading simulations using a specially developed dataset that incorporates historical stock prices, financial news, and sentiment data over a period of one year. This dataset is designed to reflect real-world trading scenarios, providing a comprehensive view of how various factors influence stock performance. The primary objective of this task is to assess the model's capability to make informed trading decisions based on a combination of quantitative and qualitative data, such as market trends and sentiment analysis. By simulating trading activities, models are tasked with generating actionable insights and strategies that maximize profitability while managing risk. The diverse nature of the data, which includes price movements, news events, and sentiment fluctuations, requires models to effectively integrate and analyze multiple data streams to optimize trading outcomes.  
    * **Metrics:**  
      Sharpe Ratio (SR).
</details>

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
