---
title: "Fixing Open LLM Leaderboard with Math-Verify"
thumbnail: /blog/assets/math_verify_leaderboard/thumbnail.png
authors:
- user: hynky
- user: alozowski
- user: SaylorTwift
- user: clefourrier
---


# Fixing Open LLM Leaderboard with Math-Verify

3 weeks ago, we showed how hard it is to correctly evaluate LLM performance on math problems, and introduced [Math-Verify](https://github.com/huggingface/Math-Verify), a better solution to validate models on math (read more in the [announcement](https://x.com/HKydlicek/status/1881734376696041659))!


Today, we’re thrilled to share that we’ve used Math-Verify to thoroughly re-evaluate all 3,751 models ever submitted to the Open LLM Leaderboard, for even fairer and more robust model comparisons!

## Why math evaluation on the Open LLM Leaderboard was broken
The [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) is the most used leaderboard on the Hugging Face Hub: it compares open Large Language Models (LLM) performance across various tasks. One of these tasks, called MATH-Hard, is specifically about math problems: it evaluates how well LLMs solve high-school and university-level math problems. It is using 1,324 highest difficulty problems (Level 5) from the [Hendrycks MATH](https://github.com/hendrycks/math) dataset spread across 7 topics (precalculus, prealgebra, algebra, intermediate algebra, counting/probability and number theory), using a 5-shot approach (the model is provided with 5 examples in the prompt to showcase how it should answer).

A typical question looks like this:
```
For all real numbers $r$ and $s$, define the mathematical operation $\#$ such that the following conditions apply: $r\ \#\ 0 = r, r\ \#\ s = s\ \#\ r$, and $(r + 1)\ \#\ s = (r\ \#\ s) + s + 1$. What is the value of $11\ \#\ 5$?
```

To which the answer would be:
```
71
```

In the leaderboard, models would have to end their answers with a very specific string (following the [Minerva-Math paper](https://arxiv.org/abs/2206.14858)):
```
“Final answer is [ANSWER]. I hope it is correct.”.
```

The leaderboard would then try to parse `[ANSWER]` with [SymPy](https://docs.sympy.org/latest/index.html) to convert it to a symbolic representation (and simplify the values if needed), before finally comparing it to the gold target.

However, users reported a number of issues with the above. 

To start, a recurring issue was the inability of some models to follow the expected answer format from the examples: they outputted other sentences instead to introduce their answers. Since the format was not followed, answers were marked as wrong even if they were actually correct! (Which is an issue if what you’re interested in is “how good the model is at math” specifically).

| 📄 Example |  ❗️Issue | ✅ Math-Verify | 🛑 Old-Leaderboard |
| --- | --- | --- | --- |
| Therefore, the perimeter of one of these triangles is $14 + 7\sqrt{2}$ inches, expressed in simplest radical form. | Failed extraction | `7*sqrt(2) + 14` | None |
| Therefore, the sum of the infinite geometric series is \(\frac{7}{9}\). | Failed extraction | `7/9` | None |
| \( p(n) \) and \( p(n+1) \) share a common factor greater than 1 is \(\boxed{41}\). | Failed extraction | `4` | None |
| So it’s \frac{1}{9} | Failed extraction | `1/9` | None |
| Concluding he has \boxed{5} cars | Failed extraction | `5` | None |



The next step, converting `[ANSWER]` to the symbolic representation also presented some issues, this time linked to the SymPy parsing:

| 📄 Example |  ❗️Issue | ✅ Math-Verify | 🛑 Old-Leaderboard |
| --- | --- | --- | --- |
| The final answer is $2x + 4y + z - 19 = 0$. I hope it is correct. | Partial parse of parametric eq | Eq(2*x + 4*y + z - 19, 0) | 0 |
| \(23\) | Failed extraction due to latex borders | `23` | None |
| \((- \infty, -14) \cup (-3, \infty)\). | Failed extraction due to interval | Union(Interval.open(-oo, -14), Interval.open(-3, oo)) | None |
| 100\% | Failed extraction due to invalid symbol | `1` | None |
| \begin{pmatrix}\frac{1}{50}&\frac{7}{50}\\frac{7}{50}&\frac{49}{50}\end{pmatrix} | Failed extraction due to Matrix | Matrix([[1/50, 7/50], [7/50, 49/50]]) | None |


On the final step, when comparing the extracted answer with the target expression, a number of issues also occurred:

| 📄 Example |  ❗️Issue | ✅ Math-Verify | 🛑 Old-Leaderboard |
| --- | --- | --- | --- |
| 1/3 == 0.333333 | No rounding support | True | False |
| sqrt(1/2)*7 == sqrt(0.5)*7 | No numerical evaluation support | True | False |
| k = 1 == 1 | No variable assignment support | True | False |
| Matrix.ones == Matrix.ones | No support for matrix equivalence | True | False |
| {1} \union {1,4} == {1,4} | No support for set comparison | True | False |

**All of these issues are now completely fixed with the new Math-Verify parser!**
## Which model is the best at math? A complete reshuffling of cards thanks to fairer evaluations

As all these issues tend to accumulate, some models deeply suffered from this, and their performance was strongly underestimated… so we removed the previous evaluator and added Math-Verify, which was as simple as changing only 3 lines of code! (You can try it too on your math evals!) 
 
This therefore meant re-evaluating all submitted models since June… and it completely overhauled the top 20 models on the MATH subset of the leaderboard.

### Impact of the change
On average, models solved **61 more problems** across the board, equating to a **4.66-point** increase across the board!


![score_change](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/math_verify_leaderboard/score-change.png)

The two subsets that showed the most significant improvement were both algebra-related (Algebra and Prealgebra) with gains of **8.27** and **6.93**, respectively. In extreme cases, some models demonstrated improvements of nearly **90** points on these subsets.
We believe these subsets saw the greatest improvement because they frequently involve answers presented as sets (due to questions with multiple solutions) and matrices. The Math-Verify has enhanced its handling of both answer types, contributing to these notable gains.

![subset_change](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/math_verify_leaderboard/subset-change.png)


### Model Family Changes
We initially discovered the math evaluation issues when inspecting Qwen models, which had unusually low scores compared to the self-reported performance. After the Math-Verify introduction, the scores more than doubled for these models, showcasing previous severe underestimation of performance.

But Qwen models aren’t alone. Another major family affected is **DeepSeek**. After switching to Math-Verify, DeepSeek models almost tripled their scores! This is because their answers are typically wrapped in boxed `(\boxed{})` notations which the old evaluator couldn’t extract.
![model_family_change](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/math_verify_leaderboard/model-family-change.png)

### Changes in the MATH-Hard Leaderboard
As mentioned at the beginning, the Top 20 rankings have undergone a significant shift, with **Nvidia’s AceMath** models now dominating the MATH-Hard leaderboard.
Another major beneficiary of this change are the **Qwen** derivatives, which are now almost exclusively the only models ranking right below AceMath.
Following is the complete table comparing the old and new Top 20 leaderboard rankings:

![math_hard_leaderboard_change](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/math_verify_leaderboard/math-hard-change.png)

### Changes in the Leaderboard
Finally, we examined how the overall Leaderboard results have evolved. While the top four positions remain unchanged, the rest have undergone significant shifts. Due to the rise of multiple Qwen derivatives in the MATH subset, the presence of Qwen models among the top 20 has grown-derived models grown even further at the Overall results. 
![leaderboard_change](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/math_verify_leaderboard/overal-change.png)

Many other models also completely jumped in the rankings, gaining 200 places or more! You can check out the results in more detail at the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).

## Wrapping Up
The introduction of Math-Verify has significantly improved the accuracy and fairness of our evaluations on the Open LLM Leaderboard. This has led to a reshuffling of the leaderboard, with many models showing substantial improvements in their scores.

We encourage all developers and researchers to adopt Math-Verify for their own math evaluations. By doing so, you can ensure that your models are evaluated with more reliable results. Additionally, we invite you to explore the updated rankings and see how your favorite models have changed in performance.


