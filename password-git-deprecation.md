---
title: Password Git Authentication Deprecation
thumbnail: TODO
authors:
- user: Sylvestre
- user: TODO

---

# Hugging Face Hub: Important Git Authentication Changes

<!-- {blog_metadata} -->
<!-- {authors} -->

As part of our commitment to improving the security of our services, we are making changes to the way you authenticate with the Hugging Face Hub Git.
Starting from **October 1st, 2023**, we will no longer accept account passwords for authenticated Git operations. Instead, we recommend using most secure authentication, such as personal access tokens or SSH keys.

## Background

In recent months, we have implemented various security enhancements, including sign-in alerts and support for SSH keys. However, users have still been able to authenticate Git operations using their username and password. To further improve security, we are now transitioning to token-based or SSH key authentication.

## Advantages of Token-Based Authentication

Token-based authentication offers several benefits over traditional password authentication:

- **Unique**: Tokens can be generated per use or per device, providing an additional layer of security.
- **Revocable**: Tokens can be individually revoked without affecting other credentials.
- **Limited**: Tokens can be finely scoped for specific access needs.
- **Random**: Tokens are immune to dictionary or brute force attacks.

## Action Required Today

If you currently use a password to authenticate against the Git repositories of our models, datasets, or Spaces, please switch to using a personal access token or SSH keys before **October 1st, 2023**.

### Switching to personal access token
You will need to generate an access token for your account, you can follow https://huggingface.co/docs/hub/security-tokens#user-access-tokens to generate one.

After generating your access token, you can update your git repository using the following commands:

```bash
$: git remote set-url origin https://<token>@huggingface.co/<user_name>/<repo_name>
$: git pull https://<token>@huggingface.co/<user_name>/<repo_name>.git
```


### Switching to SSH keys

Follow our guide to generate a SSH key and add it to your account, https://huggingface.co/docs/hub/security-git-ssh

Then I'll be able to update your git repository using:

```bash
$: git remote set-url origin git@hf.co:<user_name>/<repo_name>
```


## Timeline

Here's what you can expect in the coming weeks:

- Today: Users relying on passwords for Git authentication may receive emails urging them to update their authentication method.
- October 1st: Personal access tokens or SSH keys will be mandatory for all Git operations.

For more details, reach out to HF Support to address any questions or concerns.