---
title: "4M Models Scanned: Protect AI + Hugging Face 6 Months In"
thumbnail: /blog/assets/pai-6-month/thumbnail.png
authors:
- user: sean-pai
  guest: true
  org: protectai
---

# 4M Models Scanned: Protect AI \+ Hugging Face 6 Months In

![pai-hf-header](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pai-6-month/pai-hf-header.png)

Hugging Face and Protect AI partnered in [October 2024](https://protectai.com/blog/protect-ai-hugging-face-ml-supply-chain) to enhance machine learning (ML) model security through [Guardian’s](https://protectai.com/guardian) scanning technology for the community of developers who explore and use models from the Hugging Face Hub. The partnership has been a natural fit from the start—Hugging Face is on a mission to democratize the use of open source AI, with a commitment to safety and security; and Protect AI is building the guardrails to make open source models safe for all.

## 4 new threat detection modules launched

Since October, Protect AI has significantly expanded Guardian's detection capabilities, improving existing threat detection capabilities and launching four new detection modules:

1. [PAIT-ARV-100](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-ARV-100): Archive slip can write to file system at load time  
2. [PAIT-JOBLIB-101](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-JOBLIB-101): Joblib model suspicious code execution detected at model load time  
3. [PAIT-TF-200](https://protectai.com/insights/knowledge-base/backdoor-threats/PAIT-TF-200): TensorFlow SavedModel contains architectural backdoor  
4. [PAIT-LMAFL-300](https://protectai.com/insights/knowledge-base/runtime-threats/PAIT-LMAFL-300): Llamafile can execute malicious code during inference

With these updates, Guardian covers more model file formats and detects additional sophisticated obfuscation techniques, including the high severity [CVE-2025-1550 vulnerability](https://protectai.com/insights/knowledge-base/runtime-threats/PAIT-KERAS-301) in Keras. Through enhanced detection tooling, Hugging Face users receive critical security information via inline alerts on the platform and gain access to comprehensive vulnerability reports on [Insights DB](https://protectai.com/insights). Clearly labeled findings are available on each model page, empowering users to make more informed decisions about which models to integrate into their projects.

|![scan-screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pai-6-month/scan-screenshot.png)|
|:--:|
|***Figure 1:** Protect AI’s inline alerts on Hugging Face*|

## By the numbers

**As of April 1, 2025, Protect AI has successfully scanned 4.47 million unique model versions in 1.41 million repositories on the Hugging Face Hub.**

To date, Protect AI has identified a total of **352,000 unsafe/suspicious issues across 51,700 models**. In just the last 30 days, Protect AI has served **226 million requests** from Hugging Face at a **7.94 ms response time**.

# **Maintaining a Zero Trust Approach to Model Security**

Protect AI’s Guardian applies a zero trust approach to AI/ML security. This especially comes into play when treating arbitrary code execution as inherently unsafe, regardless of intent. Rather than just classifying overtly malicious threats, Guardian flags execution risks as suspicious on InsightsDB, recognizing that even harmful code can look innocuous through obfuscation techniques (see more on payload obfuscation below). Attackers can disguise payloads within seemingly benign scripts or extensibility components of a framework, making payload inspection alone insufficient for ensuring security. By maintaining this cautious approach, Guardian helps mitigate risks posed by hidden threats in machine learning models.

# **Evolving Guardian’s Model Vulnerability Detection Capabilities** 

AI/ML security threats are evolving every single day. That's why Protect AI leverages both in-house [threat research teams](https://protectai.com/threat-research) and [huntr](https://huntr.com)—the world's first and largest AI/ML bug bounty program powered by our community of over 17,000 security researchers.

Coinciding with our partnership launch in October, Protect AI launched a new program on huntr to crowdsource research on new [Model File Vulnerabilities](https://blog.huntr.com/hunting-vulnerabilities-in-machine-learning-model-file-formats). Since the launch of the program, **they have  received over 200 reports** that Protect AI teams have worked through and incorporated into Guardian—all of which are automatically applied to the model scans here on Hugging Face. 

|![huntr-screenshot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pai-6-month/huntr-screenshot.png)|
|:--:|
|***Figure 2:** huntr’s bug bounty program*|

## Common attack themes

As more huntr reports come in and more independent threat research is conducted, certain trends have emerged.

**Library-dependent attack chains:** These attacks focus on a bad actor’s ability to invoke functions from libraries present in the ML workstations environment. These are reminiscent of the “drive-by download” style of attacks that afflicted browsers and systems when common utilities like Java and Flash were present. Typically, the scale of impact of these attacks are proportional to the pervasiveness of a given library, with common ML libraries like Pytorch having a far wider potential impact than lesser used libraries. 

**Payload obfuscation:** Several reports have highlighted ways to insert, obfuscate, or “hide” a payload in a model that bypasses common scanning techniques. These vulnerabilities use techniques like compression, encoding, and serialization to obfuscate the payload and are not easily detectable. Compression is an issue since libraries like Joblib allow compressed payloads to be loaded directly. Container formats like Keras and NeMo embed additional model files, each potentially vulnerable to their own specific attack vectors. Compression exposes users to TarSlip or ZipSlip vulnerabilities. While the impacts of these will often be limited to Denial of Service, in certain circumstances these vulnerabilities can lead to Arbitrary Code Execution by leveraging path traversal techniques, allowing malicious attackers to overwrite files that are often automatically executed.

**Framework-extensibility vulnerabilities**: ML frameworks provide numerous extensibility mechanisms that inadvertently create dangerous attack vectors: custom layers, external code dependencies, and configuration-based code loading. For example, CVE-2025-1550 in Keras, reported to us by the huntr community, demonstrates how custom layers can be exploited to execute arbitrary code despite security features. Configuration files with serialization vulnerabilities similarly allow dynamic code loading. These deserialization vulnerabilities make models exploitable through crafted payloads embedded in formats that users load without suspicion. Despite security improvements from vendors, older vulnerable versions and insecure dependency handling continue to present significant risk in ML ecosystems.

**Attack vector chaining**: Recent reports demonstrate how multiple vulnerabilities can be combined to create sophisticated attack chains that can bypass detection. By sequentially exploiting vulnerabilities like obfuscated payloads and extension mechanisms, researchers have shown complex pathways for compromise that appear benign when examined individually. This approach significantly complicates detection and mitigation efforts, as security tools focused on single-vector threats often miss these compound attacks. Effective defense requires identifying and addressing all links in the attack chain rather than treating each vulnerability in isolation.

# **Delivering Comprehensive Threat Detection for Hugging Face Users**

The industry-leading Protect AI threat research team, with help from the huntr community, is continuously gathering data and insights in order to develop new and more robust model scans as well as automatic threat blocking (available to Guardian customers). In the last few months, Guardian has:

**Enhanced detection of library-dependent attacks**: Significant expansion of Guardian’s scanning capabilities for detecting library-dependent attack vectors. The scanners for [PyTorch](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-PYTCH-101) and [Pickle](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-PKL-101) now perform deep structure analysis of serialized code, examining execution paths and identifying potentially malicious code patterns that could be triggered through library dependencies. For example, the PyTorch torchvision.io functions can overwrite any file on the victim’s system to either include a payload or delete all of its content. Guardian can now detect many more of these dangerous functions in popular libraries such as PyTorch, Numpy, and Pandas.

**Uncovered obfuscated attacks:** Guardian performs multi-layered analyses across various archive formats, decompressing nested archives and examining compressed payloads for malicious models. This approach detects attempts to hide malicious code through compression, encoding, or serialization techniques. Joblib, for example, supports saving models using different compression formats which can obfuscate Pickle deserialization vulnerabilities, and the same can be done in other formats like Keras which can include Numpy weights files that have deserialization payloads in them.

**Detected exploits in framework extensibility components:** Guardian’s constantly improving detection modules alerted users on Hugging Face to models that were impacted by CVE-2025-1550 (a critical security finding) before the vulnerability was publicly disclosed. These detection modules comprehensively analyze ML framework extension mechanisms, allowing only standard or verified components and blocking potentially dangerous implementations, regardless of their apparent intent. 

**Identified additional architectural backdoors**: Guardian’s architectural backdoor detection capabilities were expanded beyond ONNX formats to include additional model formats like [TensorFlow](https://protectai.com/insights/knowledge-base/backdoor-threats/PAIT-TF-200). 

**Expanded model format coverage:** Guardian’s true strength comes from the depth of its coverage, which has driven substantial expansion of detection modules to include additional formats like [Joblib](https://protectai.com/insights/knowledge-base/deserialization-threats/PAIT-JOBLIB-100) and an increasingly popular [llamafile](https://protectai.com/insights/knowledge-base/runtime-threats/PAIT-LMAFL-300) format, with support for additional ML frameworks coming soon. 

**Provided deeper model analysis:** Actively research on additional ways to augment current detection capabilities for better analysis and detection of unsafe models. Expect to see significant enhancements in reducing both false positives and false negatives in the near future. 

# **It Only Gets Better from Here**

Through the partnership with Protect AI and Hugging Face, we’ve made third-party ML models safer and more accessible. We believe that having more eyes on model security can only be a good thing. We’re increasingly seeing the security world pay attention and lean in, making threats more discoverable and AI usage safer for all.

