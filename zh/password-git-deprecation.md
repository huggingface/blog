---
title: Hub 上的 Git 操作不再支持使用密码验证
thumbnail: /blog/assets/password-git-deprecation/thumbnail.png
authors:
- user: Sylvestre
- user: pierric
- user: sbrandeis
translators:
- user: chenglu
---

# Hugging Face Hub: Git 操作认证的重要变更


在 Hugging Face，我们一直致力于提升服务安全性，因此，我们将对通过 Git 与 Hugging Face Hub 交互时的认证方式进行更改。从 **2023 年 10 月 1 日** 开始，我们将不再接受密码作为命令行 Git 操作的认证方式。我们推荐使用更安全的认证方法，例如用个人访问令牌替换密码或使用 SSH 密钥。

## 背景

近几个月来，我们已经实施了各种安全增强功能，包括登录提醒和 Git 中对 SSH 密钥的支持，不过，用户仍然可以使用用户名和密码进行 Git 操作的认证。

为了进一步提高安全性，我们现在转向基于令牌或 SSH 密钥的认证。与传统的密码认证相比，基于令牌和 SSH 密钥的认证有多个优点，包括唯一性、可撤销和随机特性，这些都增强了安全性和控制。

## 立即需采取的行动

如果你当前使用 HF 账户密码进行 Git 认证，请在 **2023 年 10 月 1 日** 之前切换到使用个人访问令牌或 SSH 密钥。

### 切换到个人访问令牌

你需要为你的账户生成一个访问令牌；你可以按照 [这个文档](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) 中提到的方法来生成一个访问令牌。

生成访问令牌后，你可以使用以下命令更新你的 Git 仓库：

```
$: git remote set-url origin https://<user_name>:<token>@huggingface.co/<user_name>/<repo_name>
$: git pull origin
```

或者，如果你克隆了一个新的仓库，当你的 Git 凭证管理器要求你提供认证凭证时，你可以直接输入令牌来替代密码。

### 切换到 SSH 密钥

按照我们的 [指南文档](https://huggingface.co/docs/hub/security-git-ssh) 生成 SSH 密钥并将其添加到你的账户。

然后，你可以使用以下命令更新你的 Git 仓库：

```
$: git remote set-url origin git@hf.co:<user_name>/<repo_name>
```

## 时间表

在接下来的时间里，这个变动将以下面的时间表来执行：

- 现在开始起：依赖密码进行 Git 认证的用户可能会收到电子邮件，敦促他们更新认证方法。
- 10 月 1 日：个人访问令牌或 SSH 密钥将成为所有 Git 操作的强制要求。

如需更多详情，可以通过 [website@huggingface.co](mailto:website@huggingface.co) 联系支持团队，以解决你的疑问或顾虑。
