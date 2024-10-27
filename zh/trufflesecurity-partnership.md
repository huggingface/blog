---
title: "Hugging Face与TruffleHog成为合作伙伴，实现风险信息预警"
thumbnail: /blog/assets/trufflesecurity-partnership/thumbnail.png
authors:
- user: mcpotato
translators:
- user: smartisan


---

# Hugging Face与TruffleHog合作，实现风险预警

我们非常高兴地宣布与Truffle Security建立合作伙伴关系并在我们的平台集成TruffleHog强大的风险信息扫描功能。这些特性是[我们持续致力于提升安全性](https://huggingface.co/blog/2024-security-features)的重要举措之一。

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/trufflesecurity-partnership/truffle_security_landing_page.png"/>

TruffleHog是一款开源工具，用于检测和验证代码中的机密信息泄露。它拥有广泛的检测器，覆盖多种流行SaaS和云服务提供商，可扫描文件和代码仓库中的敏感信息，如凭证、令牌和加密密钥。

错误地将敏感信息提交到代码仓库可能会造成严重问题。TruffleHog通过扫描代码仓库中的机密信息，帮助开发者在问题发生前捕获并移除这些敏感信息，保护数据并防止昂贵的安全事件。

为了对抗公共和私有代码仓库中的机密信息泄露风险，我们与TruffleHog团队合作开展了两项举措：利用TruffleHog增强我们的自动扫描流程，以及在TruffleHog中创建原生的Hugging Face扫描器。

## 使用 TruffleHog 增强我们的自动化扫描流程

在 Hugging Face，我们致力于保护用户的敏感信息。因此，我们扩展了包括 TruffleHog在内的自动化扫描流程

每次推送到代码库时，我们都会对每个新文件或修改文件运行 `trufflehog filesystem` 命令，扫描潜在的风险。如果检测到已验证的风险，我们会通过电子邮件通知用户，使他们能够采取纠正措施

已验证的风险是指那些已确认可以用于对其相应提供者进行身份验证的风险。请注意，未验证的风险不一定是无害或无效的：验证可能由于技术原因而失败，例如提供者的停机时间。

即使我们为你运行 trufflehog或者你自己在代码库上运行 trufflehog 也始终是有价值的。例如，你可能已经更换了泄露的密匙，并希望确保它们显示为“未验证”，或者你希望手动检查未验证的风险是否仍然构成威胁。

We will eventually migrate to the `trufflehog huggingface` command, the native Hugging Face scanner, once support for LFS lands.

当我们支持 LFS后，我们最终会迁移到原生的 Hugging Face 扫描器，即 `trufflehog huggingface` 命令。

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/token-leak-email-example.png"/>

## TruffleHog 原生 Hugging Face 扫描器

创建原生 Hugging Face 扫描器的目标是积极的帮助我们的用户（以及保护他们的安全团队）扫描他们自己的账户数据，以发现泄露的风险。

TruffleHog 的新的开源 Hugging Face 集成可以扫描模型、数据集和 Spaces，以及任何相关的 PRs 或 Discussions。

唯一的限制是 TruffleHog 目前不会扫描任何存储在 LFS 格式中的文件。他们的团队正在努力解决这个问题，以便尽快支持所有的 `git` 源。

要使用 TruffleHog 扫描你或你组织的 Hugging Face 模型、数据集和 Spaces 中的秘密，请运行以下命令：

```sh
# For your user
trufflehog huggingface --user <username>

# For your organization
trufflehog huggingface --org <orgname>

# Or both
trufflehog huggingface --user <username> --org <orgname>
```

你可以使用 (`--include-discussions`) 和 PRs (`--include-prs`) 的可选命令来扫描 Hugging Face 讨论和 PR 评论。

如果你想要仅扫描一个模型、数据集或 Space，TruffleHog 有针对每一个的特定命令。

```sh
# Scan one model
trufflehog huggingface --model <model_id>

# Scan one dataset
trufflehog huggingface --dataset <dataset_id>

# Scan one Space
trufflehog huggingface --space <space_id>
```

如果你需要传入认证令牌，你可以使用 --token 命令，或者设置 HUGGINGFACE_TOKEN 环境变量。

这里是 TruffleHog 在 [mcpotato/42-eicar-street](https://huggingface.co/mcpotato/42-eicar-street) 上运行时的输出示例：

```
trufflehog huggingface --model mcpotato/42-eicar-street
🐷🔑🐷  TruffleHog. Unearth your secrets. 🐷🔑🐷

2024-09-02T16:39:30+02:00	info-0	trufflehog	running source	{"source_manager_worker_id": "3KRwu", "with_units": false, "target_count": 0, "source_manager_units_configurable": true}
2024-09-02T16:39:30+02:00	info-0	trufflehog	Completed enumeration	{"num_models": 1, "num_spaces": 0, "num_datasets": 0}
2024-09-02T16:39:32+02:00	info-0	trufflehog	scanning repo	{"source_manager_worker_id": "3KRwu", "model": "https://huggingface.co/mcpotato/42-eicar-street.git", "repo": "https://huggingface.co/mcpotato/42-eicar-street.git"}
Found unverified result 🐷🔑❓
Detector Type: HuggingFace
Decoder Type: PLAIN
Raw result: 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
Commit: 9cb322a7c2b4ec7c9f18045f0fa05015b831f256
Email: Luc Georges <luc.sydney.georges@gmail.com>
File: token_leak.yml
Line: 1
Link: https://huggingface.co/mcpotato/42-eicar-street/blob/9cb322a7c2b4ec7c9f18045f0fa05015b831f256/token_leak.yml#L1
Repository: https://huggingface.co/mcpotato/42-eicar-street.git
Resource_type: model
Timestamp: 2024-06-17 13:11:50 +0000
2024-09-02T16:39:32+02:00	info-0	trufflehog	finished scanning	{"chunks": 19, "bytes": 2933, "verified_secrets": 0, "unverified_secrets": 1, "scan_duration": "2.176551292s", "trufflehog_version": "3.81.10"}
```



致敬 TruffleHog 团队，感谢他们提供了这样一个优秀的工具，使我们的社区更安全！随着我们继续合作，敬请期待更多功能，通过 Hugging Face Hub平台为所有人提供更加安全的服务。
