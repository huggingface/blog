# **New Trackio Integration: From LLM Observability to Rapid Experimentation and Real-Time Control**

## **TL;DR**

Success in developing large language model (LLM) applications depends on fast, systematic experimentation across retrieval, training, evaluation, and other choices. Yet many teams still rely on workflows where experiments are run one after another and evaluation tools only surface results after the fact.

Today, we are excited to announce a new integration between Trackio and RapidFire AI that changes this dynamic. Trackio provides a lightweight, real-time visualization layer for your experiments, while RapidFire AI enables AI developers to define, launch, and compare multiple agentic RAG, fine-tuning, or post-training configurations in one go, while easily analyzing tradeoffs across evals, latency, and cost.

This integration showcases that Trackio works great not only for fine-tuning but also for RAG pipeline optimization: track retrieval metrics such as Precision, Recall, NDCG, and MRR in real-time, as well as LLM-as-judge or other code-based eval metrics such as Relevance, Faithfulness, and Coherence. Whether you are tuning LoRA adapters or comparing chunking strategies for RAG, this Trackio-RapidFire AI integration has you covered.

## **Trackio for Real-Time Experiment Visualization**

Trackio is a free, open-source dashboard for visualizing your experiments as they run. Instead of waiting for experiments to complete, engineers can monitor progress in real-time.

**Why Trackio?**

* **Free and open source**: No usage limits, no pricing tiers; experiment tracking accessible to everyone.
* **Local-first design**: Dashboard runs locally with no server setup required; data persists in a lightweight SQLite database.
* **Familiar API with privacy**: The API resembles W&B but all data is stored locally, offering full privacy.
* **Lightweight**: The core codebase is less than 5,000 lines of Python, making it easy to understand and extend.
* **Works in Google Colab**: Trackio runs seamlessly in Colab notebooks, perfect for quick experimentation without local installation.

From Trackio's dashboard, users can:

* View multiple RAG, fine-tuning, or post-training configurations simultaneously.
* Compare all metrics side-by-side across all as many parallel runs as they want.
* Track loss, evaluation metrics, hyperparameters, etc. in real-time.

## **RapidFire AI: Executing Experiments at Scale**

[RapidFire AI](https://github.com/RapidFireAI/rapidfireai) enables rapid experimentation for easier, faster, and more impactful AI customization. Built for agentic RAG, context engineering, fine-tuning, and post-training of LLMs and other deep learning models, it delivers 16-24x higher throughput without extra resources.

Key capabilities include:

* Hyperparallel execution of RAG and fine-tuning experiments.
* Automated handling of experiment artifacts, parameters, and checkpoints.
* Structured result collection aligned with evaluation metrics.

Together, Trackio handles visualization and comparisons, while RapidFire AI handles speed, scale, and experimental rigor.

## **How RapidFire AI Integrates with Trackio**

When running experiments with RapidFire AI, you typically have many configurations executing simultaneously. This is where Trackio becomes invaluable. RapidFire AI has integrated native Trackio support into its metric logging system:

* Automatic Metric Capture: Training metrics (loss, learning rate, gradient norms, etc.) are automatically logged to Trackio during training.
* Evaluation Metrics: Evaluation metrics (accuracy, precision, ROUGE, BLEU, etc.) are captured at each evaluation step during training and during RAG configuration.
* Hyperparameter Tracking: Each run's configuration is also logged, making it easy to understand what hyperparameters produced which results.
* Real-Time Dashboard: View all your parallel runs in Trackio's dashboard as they execute.

## **Practical Workflows Enabled by the Integration**

### RAG Experimentation

This integration marks the first use of Trackio for RAG pipeline optimization. AI developers can explore multiple RAG configurations in parallel, including variations in:

* Chunk size and overlap
* Embedding models and dimensionality
* Retrieval and reranking approaches 
* Prompt structures
* Base model for generation

Results can be compared side by side across retrieval metrics, accuracy, latency, and inference cost--making tradeoffs immediately visible.

![Trackio dashboard showing RapidFire AI RAG metrics](/blog/assets/trackio-rapidfire-integration/rapidfire-trackio-rag-screenshot.png)

*Trackio dashboard comparing 4 RAG pipeline configurations on retrieval metrics (F1 Score, NDCG@5, MRR, Recall) across different chunk sizes and reranking parameters.*

### Fine-Tuning and Post-Training

For fine-tuning and post-training workflows, RapidFire AI wraps around Hugging Face TRL. The integration supports parallel sweeps across:

* LoRA rank and adapter settings
* Learning rates, lr schedules, other optimizer hyperparameters
* Dataset variants and preprocessing choices

Each resulting model is evaluated using consistent metrics, enabling clear and reproducible comparisons.

![Trackio dashboard showing RapidFire AI fine-tuning metrics](/blog/assets/trackio-rapidfire-integration/trackio-screenshot-sft.png)

*Trackio dashboard comparing 4 fine-tuning runs with different hyperparameters. The plots show training loss, validation loss, learning rate schedules, and ROUGE-L scores—making it easy to identify which configuration (Run 4, in orange) achieves the lowest loss and best generation quality.*

## **Why This Matters for LLM Teams**

By combining Trackio's visualization with RapidFire AI's hyperparallel execution, teams can monitor and reason about tradeoffs directly, enabling informed engineering decisions based on real data rather than intuition or trial-and-error.

For individual engineers, this integration shortens feedback loops and reduces reliance on brittle, ad-hoc experimentation scripts. For teams, it creates shared, reproducible artifacts that make collaboration and review easier. For organizations, it reduces experimentation cost while increasing confidence in production deployments.

By aligning experimentation tools with real-world workflows, Trackio + RapidFire AI aim to make rigorous LLM application development more accessible to the AI community.

## **Get Started**

Ready to try the integration? Here's how to get started:

**1. Install RapidFire AI:**

```bash
pip install rapidfireai
```

Trackio is included as a dependency of RapidFire AI, so no separate installation is needed.

**2. Try the tutorial notebooks:**

We've created hands-on tutorials that walk through complete experiments with Trackio tracking. Each tutorial demonstrates configuring Trackio, running parallel experiments, and viewing results in the dashboard.

**Fine-Tuning (SFT) Tutorials:**

* [SFT with Trackio \- Colab Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-colab-tutorial-sft-trackio.ipynb) \- Run directly in Google Colab, no local installation required
* [SFT with Trackio \- Local Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-tutorial-sft-trackio.ipynb) \- For local development with full RapidFire AI features

**RAG Optimization Tutorials:**

* [RAG FiQA with Trackio \- Colab Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-colab-tutorial-rag-fiqa-trackio.ipynb) \- Run directly in Google Colab, no local installation required
* [RAG FiQA with Trackio \- Local Version](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-tutorial-rag-fiqa-trackio.ipynb) \- For local development with full RapidFire AI features

**3. Learn more:**

* [RapidFire AI Integration Guide](https://github.com/gradio-app/trackio/blob/main/docs/source/rapidfireai_integration.md) \- Official integration documentation with setup instructions and examples
* [Trackio GitHub Repository](https://github.com/gradio-app/trackio) \- Full documentation and examples  
* [Trackio Documentation](https://huggingface.co/docs/trackio/index) \- API reference and guides  
* [RapidFire AI Documentation](https://oss-docs.rapidfire.ai/) \- Getting started with RapidFire AI  
* [RapidFire AI GitHub](https://github.com/RapidFireAI/rapidfireai) \- Source code and more tutorials

## **Conclusion**

By integrating Trackio and RapidFire AI, we have combined the power of hyperparallel experiment execution with free, open-source experiment tracking. This integration not only brings Trackio to RapidFire AI users but also demonstrates Trackio's versatility by applying it to RAG experimentation--a first for the library. AI engineers can now run many RAG, fine-tuning, or post-training configurations simultaneously, while maintaining full visibility into every run's progress.

We believe experiment tracking should be accessible to everyone, not locked behind pricing tiers or requiring complex infrastructure. Trackio embodies this philosophy, and we are excited to bring it to the RapidFire AI community.

We invite you to try the integration, share feedback, and help shape the future of LLM experimentation tooling.
