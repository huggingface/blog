---
title: "Introducing the Open Leaderboard for Japanese LLMs!"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_japanese.png
authors:
- user: akimfromparis
  guest: true
  org: llm-jp
- user: miyao-yusuke
  guest: true
  org: llm-jp
- user: namgiH
  guest: true
  org: llm-jp
- user: t0-0
  guest: true
  org: llm-jp
- user: sh1gechan
  guest: true
  org: llm-jp
- user: hysts
- user: clefourrier
---

# Introduction to the Open Leaderboard for Japanese LLMs

Due to the rapid evolution and the remarkable capabilities of LLMs, it became necessary to create a critical tool to evaluate Japanese LLMs. Today, we are excited to announce the **Open Japanese LLM Leaderboard**, composed of more than 20 datasets from classical to modern NLP tasks to understand underlying mechanisms of Japanese LLMs. The Open Japanese LLM Leaderboard was built by the **[LLM-jp](https://llm-jp.nii.ac.jp/en/)**, a cross-organizational project for the research and development of Japanese large language models (LLMs) in partnership with **Hugging Face**. 

The Japanese language presents its own specific challenges. Morphologically rich and in constant evolution due to historical and cultural interactions with the rest of the world, its writing system is based on a mixture of three separate sets of characters: simplified Chinese ideographic symbols kanjis (漢字), a phonetic lettering system, Hiraganas (平仮名 / ひらがな), and Katakanas (片仮名 / カタカナ) often used for foreigners words. Modern Japanese is arguably one of the hardest language to process, as it mixes up a blend of Sino-Japanese, native Japanese, Latin script (romaji /ローマ字), loanwords from the Dutch, Portuguese, French, English, German, plus Arabic and traditional Chinese numerals. In addition, the Japanese digital world brought us an evolution of emoticons written in Unicode : ), Kaomoji using Cyrillic alphabet. ∵･(ﾟДﾟ) and Greek alphabets ＿φ(°-°=). Without forgetting, the classic emojis that originated from Japan with the rise in popularity of mobile phones in the 1990s.  (Hugging Face Emoji). 🤗

The intricate writing system of Japanese hides an extra layer of complexity, the lack of space between words. Similar to Chinese or Thai language, Japanese language doesn’t have white space between linguistic units, making the detection of word boundaries extremely difficult during tokenization. Over the years, the vibrant Japanese ecosystem (from prestigious university laboratories and AI startups to the R&D centers of industry giants) have incorporated the specificities of Japanese NLP to develop modern robust Japanese LLMs, but the field has been lacking a centralized and open system to compare these models, while addressing the complexity of the Japanese language. 

We therefore introduce the Open Japanese LLM Leaderboard, a collaboration between Hugging Face and LLM-jp, to foster transparency in research, and encourage an open-source model development philosophy. We strongly believe this initiative will serve as a platform for Japanese and international researchers to collaborate, evaluate, and enhance Japanese LLMs.

## Leaderboard Metrics and Tasks

The Open Japanese LLM Leaderboard evaluates Japanese LLMs using a specialized evaluation suite,  **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)**, using a range of tasks from classical ones (such as *Natural Language Inference, Machine Translation, Summarization, Question Answering*) to more modern ones (such as *Code Generation, Mathematical Reasoning* or *Human Examination*).

For the Open Japanese LLM Leaderboard, the evaluation team of LLM-jp has developed one of the most complete benchmark for Japanese LLMs by leveraging more than 16 tasks. Those tasks assess the Japanese LLMs using a 4-shot prompt format by default, testing their ability to adjust and respond accurately, even when provided with minimal context. Please found below 4 examples in Japanese and translated in English between brackets.

## Jamp

**Jamp** (*Controlled Japanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models*) is the Japanese temporal inference benchmark for NLI. The dataset explore English and Japanese sentence pairs of various temporal inference patterns annotated with the golden labels such as entailment, neutral, or contradiction.

Example:
```
Premise:
2000年3月1日15時以来、ビクターは休みに出ている。現在、2000年3月1日19時である。  (Victor has been on vacation since March 1, 2000 at 15:00. It is now 19:00 on March 1, 2000.)   

Hypothesis:
ビクターは2000年3月1日18時には休みに出ていた。 (Victor was out on vacation at 18:00 on March 1, 2000.)

Gold_label: entailment 
Time_format: 年月日時 (year, month, day, hour)
```

## JEMHopQA

**JEMHopQA** (*Japanese Explainable Multi-hop Question Answering*) is a Japanese multi-hop QA dataset that can evaluate internal reasoning. It is a task that takes a question as input and generates an answer and derivations. 

Example:
```
"question": "『仮面ライダー電王』と『あまちゃん』、放送回数が多いのはどちらでしょう？(Which has been broadcast more often, “Kamen Rider Den-O” or “Amachan”?)", "answer": "あまちゃん(Amachan)",
"derivations": [["仮面ライダー電王(Kamen Rider Den-O)","放送回数(number of broadcast)",["49"]], ["あまちゃん(Amachan)","放送回数(number of broadcast)",["156"]]()
```

## JAQKET

**Jaqket** is an open-domain QA dataset in Japanese, using Wikipedia article names as answers. The purpose of this dataset is to promote research on question answering and machine reading in Japanese. 

Example:
```
question  問題文  "格闘家ボブ・サップの出身国はどこでしょう? (From which country is the fighter, Bob Sapp?)"
answer_entity  正解Wikipedia記事名(Correct Wikipedia article)  "アメリカ合衆国 (United States of America)" 
answer_candidates  解答候補Wikipedia記事名のリスト(List of potential candidate from Wikipedia article)  
    ["アメリカ合衆国","カナダ", ...(United States of America, Canada)] 
original_question  正規化される前の問題文 (Problem statement before normalization) 
    "格闘家ボブ・サップの出身国はどこでしょう？(From which country is the fighter, Bob Sapp?)" 
original_answer  正規化される前の正解 (Correct answers before normalization)（訓練データのみ) (training data only)  "アメリカ (America)"
```

## chABSA

**chABSA** was developed as an *Aspect-Based Sentiment Analysis* dataset. chABSA based on a carefully annotated financial reports of Japanese listed-companies in 2016 fiscal year focus on the pair of entity, the attribute, and the sentiment. 230 companies are annotated on 2,260 companies listed in Japan (roughly 10% of all companies) according to the taxonomy of the Japanese financial regulator, *Financial Service Agency (FSA)*.

Example
```
"root":{2 items
"header":{8 items
"document_id":string"E02367"
"document_name":string"任天堂株式会社 (Nintendo Co., Ltd.)"
"doc_text":string"有価証券報告書 (Annual securities report)"
"edi_id":string"E02367"
"security_code":string"79740" (Ticker)
"category33":string"その他製品 (Other Products)"
"category17":string"情報通信・サービスその他 (Information and Communications)"
"scale":string"1"
}
"sentences":[13 items
0:{3 items
    "sentence_id":int0
    "sentence":string"当連結会計年度の状況は、プレイスタイルを多様化させる新しい家庭用据置型テレビゲーム機「Nintendo Switch」を全世界で3月3日に発売し、好調な滑り出しを見せています (During the consolidated fiscal year, Nintendo Switch, a new home console system that diversifies the ways you can play games, was launched on March 3 worldwide and is off to a good start.)"
    "opinions":[1 item
0:{5 items
    "target":string"Nintendo Switch"
    "category":string"product#general"
    "polarity":string"positive"
    "from":int43
    "to":int58
    }
```

## mbpp-ja

The **mbpp-ja** dataset is a Japanese version of *Mostly Basic Python Problems dataset* (MBPP) translated from English into Japanese by **[LLM-jp](https://llm-jp.nii.ac.jp/en/)** by leveraging the translation tool, **[DeepL](https://www.deepl.com)**.

Example:
```
text: Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
code:R = 3 C = 3 def min_cost(cost, m, n): 	tc = [[0 for x in range(C)] for x in range(R)] 	tc[0][0] = cost[0][0] 	for i in range(1, m+1): 		tc[i][0] = tc[i-1][0] + cost[i][0] 	for j in range(1, n+1): 		tc[0][j] = tc[0][j-1] + cost[0][j] 	for i in range(1, m+1): 		for j in range(1, n+1): 			tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 	return tc[m][n]

test_list: [ "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8", "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12", "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16" ]
test_list_jpn: 与えられたコスト行列cost[][]とcost[][]内の位置(m, n)に対して、(0, 0)から(m, n)に到達するための最小コスト経路を求める関数を書きなさい。
```

## Technical Setup

The leaderboard is inspired by the **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)**. Models that are submitted are deployed automatically using HuggingFace’s **[Inference endpoints](https://huggingface.co/docs/inference-endpoints/index)**, evaluated through the **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** library on the version 1.14.1, with memory-efficient inference and serving engine, **[vLLM](https://github.com/vllm-project/vllm)** and computed in the backend by the premium computer platform for research in Japan,  **[MDX](https://mdx.jp/)**. 

## Future directions

The Open Japanese LLM Leaderboard will follow the development of the evaluation tool **[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)** to reflect the constant evolution of Japanese LLMs. The following are just examples of future directions in llm-jp-eval that we would like to support.

- **More Japanese evaluation datasets**
The evaluation tool, llm-jp-eval is already working for this part, for example **[JHumanEval](https://huggingface.co/datasets/kogi-jwu/jhumaneval)** (*Japanese version of HumanEval*) and the open source dataset **[MMLU](https://github.com/hendrycks/test)** (*Measuring Massive Multitask Language Understanding*).

- **Chain-of-Thought evaluation**
We can compare the performance of LLMs between when employing Chain-of-Thought prompts against basic prompts to have a better understanding of the model behaviors

- **Out-of-Choice rates as the metric**
For some evaluation tasks that already have a clear list of labels used in the specific task, such as Natural Language Inference. And that information is included in our basic prompt, we can evaluate how well each LLM follows a given instruction to calculate the OoC rate.

## Acknowledgements

Built by the research consortium **LLM-jp**, the Open Japanese LLM Leaderboard is proudly sponsored by the **[National Institute of Informatics](https://www.nii.ac.jp/en/)** in Tokyo, Japan in collaboration with the  high-performance computing platform, **[MDX](https://mdx.jp/)** program.

We would like to extend our gratitude to **Prof. Yusuke Miyao** and **Namgi Han** from the *University of Tokyo* for their scientific consultation and guidance, as well as, **Clémentine Fourrier** and **Toshihiro Hayashi** of Hugging Face that has assisted with the integration and customization of their new evaluation framework and leaderboard template. 