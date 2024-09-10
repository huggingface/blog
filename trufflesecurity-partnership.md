---
title: Hugging Face partners with TruffleHog to Scan for Secrets
thumbnail: /blog/assets/trufflesecurity-partnership/thumbnail.png
authors:
- user: mcpotato
---

# Hugging Face partners with TruffleHog to Scan for Secrets

We're excited to announce our partnership and integration with Truffle Security, bringing TruffleHog's powerful secret scanning features to our platform as part of [our ongoing commitment to security](https://huggingface.co/blog/2024-security-features).

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/trufflesecurity-partnership/truffle_security_landing_page.png"/>

TruffleHog is an open-source tool that detects and verifies secret leaks in code. With a wide range of detectors for popular SaaS and cloud providers, it scans files and repositories for sensitive information like credentials, tokens, and encryption keys.

Accidentally committing secrets to code repositories can have serious consequences. By scanning repositories for secrets, TruffleHog helps developers catch and remove this sensitive information before it becomes a problem, protecting data and preventing costly security incidents.

To combat secret leakage in public and private repositories, we worked with the TruffleHog team on two different initiatives:
Enhancing our automated scanning pipeline with TruffleHog
Creating a native Hugging Face scanner in TruffleHog

## Enhancing our automated scanning pipeline with TruffleHog

At Hugging Face, we are committed to protecting our users' sensitive information. This is why we've implemented an automated security scanning pipeline that scans all repos and commits. We have extended our automated scanning pipeline to include TruffleHog, which means there are now three types of scans:

- malware scanning: scans for known malware signatures with [ClamAV](https://www.clamav.net/)
- pickle scanning: scans pickle files for malicious executable code with [picklescan](https://github.com/mmaitre314/picklescan)
- secret scanning: scans for passwords, tokens and API keys with TruffleHog

We run the `trufflehog filesystem` command on every new or modified file on each push to a repository, scanning for potential secrets. If and when a verified secret is detected, we notify the user via email, empowering them to take corrective action.

Verified secrets are the ones that have been confirmed to work for authentication against their respective providers. Note, however, that unverified secrets are not necessarily harmless or invalid: verification can fail due to technical reasons, such as in the case of down time from the provider.

It will always be valuable to run trufflehog on your repositories yourself, even when we do it for you. For instance, you could have rotated the secrets that were leaked and want to make sure they come up as “unverified”, or you’d like to manually check if unverified secrets still pose a threat.

We will eventually migrate to the `trufflehog huggingface` command, the native Hugging Face scanner, once support for LFS lands.

<img class="block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/token-leak-email-example.png"/>

## TruffleHog Native Hugging Face Scanner

The goal for creating a native Hugging Face scanner in TruffleHog is to empower our users (and the security teams protecting them) to proactively scan their own account data for leaked secrets.

TruffleHog’s new open-source Hugging Face integration can scan models, datasets and Spaces, as well as any relevant PRs or Discussions. The only limitation is TruffleHog will not currently scan files stored in LFS. Their team is looking to address this for all of their `git` sources soon. 

To scan all of your, or your organization’s Hugging Face models, datasets, and Spaces for secrets using TruffleHog, run the following command(s):

```sh
# For your user
trufflehog huggingface --user <username>

# For your organization
trufflehog huggingface --org <orgname>

# Or both
trufflehog huggingface --user <username> --org <orgname>
```

You can optionally include the (`--include-discussions`) and PRs (`--include-prs`) flags to scan Hugging Face discussion and PR comments.

If you’d like to scan just one model, dataset or Space, TruffleHog has specific flags for each of those.

```sh
# Scan one model
trufflehog huggingface --model <model_id>

# Scan one dataset
trufflehog huggingface --dataset <dataset_id>

# Scan one Space
trufflehog huggingface --space <space_id>
```

If you need to pass in an authentication token, you can do so using the --token flag or by setting a HUGGINGFACE_TOKEN environment variable.

Here is an example of TruffleHog’s output when run on [mcpotato/42-eicar-street](https://huggingface.co/mcpotato/42-eicar-street):

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
Kudos to the TruffleHog team for offering such a great tool to make our community safe! Stay tuned for more features as we continue to collaborate to make the Hub more secure for everyone.

