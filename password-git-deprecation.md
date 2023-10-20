---
title: Deprecation of Git Authentication using password
thumbnail: /blog/assets/password-git-deprecation/thumbnail.png
authors:
- user: Sylvestre
- user: pierric
- user: sbrandeis

---

# Hugging Face Hub: Important Git Authentication Changes


Because we are committed to improving the security of our services, we are making changes to the way you authenticate when interacting with the Hugging Face Hub through Git.
Starting from **October 1st, 2023**, we will no longer accept passwords as a way to authenticate your command-line Git operations. Instead, we recommend using more secure authentication methods, such as replacing the password with a personal access token or using an SSH key.

## Background

In recent months, we have implemented various security enhancements, including sign-in alerts and support for SSH keys in Git. However, users have still been able to authenticate Git operations using their username and password. To further improve security, we are now transitioning to token-based or SSH key authentication.
Token-based and SSH key authentication offer several advantages over traditional password authentication, including unique, revocable, and random features that enhance security and control.
## Action Required Today

If you currently use your HF account password to authenticate with Git, please switch to using a personal access token or SSH keys before **October 1st, 2023**.

### Switching to personal access token
You will need to generate an access token for your account; you can follow https://huggingface.co/docs/hub/security-tokens#user-access-tokens to generate one.

After generating your access token, you can update your Git repository using the following commands:

```bash
$: git remote set-url origin https://<user_name>:<token>@huggingface.co/<repo_path>
$: git pull origin
```
where `<repo_path>` is in the form of:
- `<user_name>/<repo_name>` for models
- `datasets/<user_name>/<repo_name>` for datasets
- `spaces/<user_name>/<repo_name>` for Spaces

If you clone a new repo, you can just input a token in place of your password when your Git credential manager asks you for your authentication credentials.

### Switching to SSH keys

Follow our guide to generate an SSH key and add it to your account: https://huggingface.co/docs/hub/security-git-ssh

Then you'll be able to update your Git repository using:

```bash
$: git remote set-url origin git@hf.co:<repo_path> # see above for the format of the repo path
```

## Timeline

Here's what you can expect in the coming weeks:

- Today: Users relying on passwords for Git authentication may receive emails urging them to update their authentication method.
- October 1st: Personal access tokens or SSH keys will be mandatory for all Git operations.

For more details, reach out to HF Support to address any questions or concerns at website@huggingface.co
