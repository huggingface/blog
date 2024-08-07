---
title: 2024 Security Feature Highlights
thumbnail: /blog/assets/2024-security-feature.png
authors:
- user: jack-kumar
---

# 2024 Security Feature Highlights

Security is a top priority at Hugging Face, and we're committed to continually enhancing our defenses to safeguard our users. In our ongoing security efforts, we have developed a range of security features designed to empower users to protect themselves and their  assets. In this blog post, we'll take a look at our current security landscape as of August 6th, 2024, and break down key security features available on the Hugging Face Hub. We'll explore two categories: the essential security features available to all users, and the advanced controls available to Enterprise users.

**Table Of Contents**
- [Hub Security Features](#hub-security-features)
  - [Fine Grained Tokens](#fine-grained-token)
  - [Two Factor Authentication](#two-factor-authentication-2fa)
  - [Commit Signing](#commit-signing)
  - [Organizational Access Controls](#organizational-access-controls)
  - [Automated Security Scanning](#automated-security-scanning)
- [Enterprise Security Features](#enterprise-security-features)
  - [Single Sign-On (SSO)](#single-sign-on-sso)
  - [Resource Groups](#resource-groups)
  - [Organizational Token Management](#organizational-token-management)
  - [Data Residency](#data-residency)
  - [Audit Logs](#audit-logs)
  - [Compliance](#compliance)
  - [Custom Security Features](#custom-security-features)
- [Conclusion](#conclusion)

## Hub Security Features
The following security features are available to all users of the Hugging Face Hub. We highly recommend that you use all of these controls where possible as it will help increase your resiliency against a variety of common attacks, such as phishing, token leaks, credential stuffing, session hijacking, etc.

### Fine Grained Token
User Access Tokens are required to access Hugging Face via APIs. In addition to the standard "read" and "write" tokens, Hugging Face supports "fine-grained" tokens which allow you enforce least privilege by defining permissions on a per resource basis, ensuring that no other resources can be impacted in the event the token is leaked. Fine-grained tokens offer a plethora of ways to tune your token, see the images below for the options available. You can learn more about tokens here: https://huggingface.co/docs/hub/en/security-tokens 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/fine-grained-tokens-1.png)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/fine-grained-tokens-2.png)

### Two Factor Authentication (2FA)
Two factor authentication adds an extra layer of protection to your online accounts by requiring two forms of verification before granting access. 2FA combines something you know (like a password) with something you have (such as a smartphone) to ensure that only authorized users can access sensitive information. By enabling 2FA, you can greatly reduce the risk of unauthorized access from compromised passwords, credential stuffing and phishing. You can learn more about 2FA here: https://huggingface.co/docs/hub/en/security-2fa 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/2fa.png)

### Commit Signing
Although Git has an authentication layer to control who can push commits to a repo, it does not authenticate the actual commit author. This means it's possible for bad actors to impersonate authors by using `git config --global user.email you@company.com` and `git config --global user.name Your Name`. This config does not automatically give them access to push to your repositories that they otherwise wouldn't have - but it does allow them to impersonate you anywhere they can push to. This could be a public repository or a private repository using compromised credentials or stolen SSH key.  

Commit signing adds an additional layer of security by using GPG to mitigate this issue; you can learn more at [Git Tools: Signing Your Work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work). Hugging Face gives authors the ability to add their GPG keys to their profile. When a signed commit is pushed, the signature is authenticated using the GPG key in the authors profile. If it's a valid signature, the commit will be marked with a “Verified” badge. You can learn more about commit signing here: https://huggingface.co/docs/hub/en/security-gpg 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/commit-signing.png)

### Organizational Access Controls
Organizations on Hugging Face have access to Organizational Access Controls. This allows teams and businesses to define least privilege access to their organization by assigning "read", "write", "contributor" or "admin" roles to each of their users. This helps ensure that the compromise of one user account (such as via phishing) cannot affect the entire organization. You can learn more about Organizational Access Controls here: https://huggingface.co/docs/hub/en/organizations-security 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/organizational-access-controls.png)

### Automated Security Scanning
Hugging Face implements an automated security scanning pipeline that scans all repos and commits. Currently, there are three major components of the pipeline:
- malware scanning: scans for known malware signatures
- pickle scanning: scans pickle files for malicious executable code
- secret scanning: scans for passwords, tokens and API keys

In the event a malicious file is detected, the scans will place a notice on the repo allowing users to see that they may potentially be interacting with a malicious repository. You can see an example of a (fake) malicious repository here: https://huggingface.co/mcpotato/42-eicar-street/tree/main. 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/security-scanning.png)

For any secrets detected, the pipeline will send an email notifying the owner so that they can invalidate and refresh the secret. You can learn more about automated scanning here: 
- https://huggingface.co/docs/hub/en/security-malware 
- https://huggingface.co/docs/hub/en/security-pickle 
- https://huggingface.co/docs/hub/en/security-secrets 

## Enterprise Security Features
In addition to the security features available to all users, Hugging Face offers advanced security controls for Enterprise users. These additional controls allow enterprises to build a security configuration that is most effective for them.

### Single Sign-On (SSO)
Single sign-on (SSO) allows a user to access multiple applications with one set of credentials. Enterprises have widely moved to SSO as it allows their employees to access a variety of corporate software using identities that are managed centrally by their IT team. Hugging Face Enterprise supports SSO with both the SAML 2.0 and OpenID Connect (OIDC) protocols, and supports any compliant provider such as Okta, OneLign, Azure AD, etc. Additionally, SSO users can be configured to be dynamically assigned access control roles based on data provided by your identity provider. You can learn more about SSO here: https://huggingface.co/docs/hub/en/security-sso 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/sso.png)

### Resource Groups
In addition to the base organizational access controls, Enterprises can define and manage groups of repositories as Resource Groups. This allows you to segment your resources by team or purpose, such as "Research", "Engineering", "Production" so that the compromise of one segment can not affect others. You can learn more about Resource Groups here: https://huggingface.co/docs/hub/en/security-resource-groups 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/resource-groups.png)

### Organizational Token Management
✨New✨ Enterprise users can now manage which tokens can access their organization and resources. Organization owners can enforce the usage of fine-grained tokens and require administrator approval for each token. Administrators can review and revoke each token that has access to their repositories at any time.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/organizational-token-management-1.png)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/organizational-token-management-2.png)

### Data Residency
Enterprise users have access to data residency controls, which allow them to define where repositories (models, datasets, spaces) are stored. This allows for regulatory and legal compliance, while also improving download and upload performance by bringing the data closer to your users. We currently support US and EU regions, with Asia-Pacific coming soon. You can learn more about Data Residency here: https://huggingface.co/docs/hub/en/storage-regions 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/data-residency.png)

### Audit Logs
Enterprise users have access to audit logs that allow organization admins to review changes to repositories, settings and billing. The audit logs contain the username, location, IP, and action taken and can be downloaded as a JSON file which can be used in your own security tooling. You can learn more about Audit Logs here: https://huggingface.co/docs/hub/en/audit-logs 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/2024-security-features/audit-log.png)

### Compliance
Hugging Face is SOC2 Type 2 certified and GDPR compliant. We offer Business Associate Addendums for GDPR data processing agreements to Enterprise Plan users. You can learn more about our Compliance efforts here: https://huggingface.co/docs/hub/en/security

### Custom Security Features
Hugging Face offers custom agreements and development of features and tools for Enterprise accounts which are established via Statement of Work (SoW) and Service Level Agreements (SLA). You can reach out directly to sales to discuss your options at https://huggingface.co/contact/sales.

## Conclusion
At Hugging Face, we're committed to providing a secure and trustworthy platform for the AI community. With our robust security features, users can focus on building and deploying AI models with confidence. Whether you're an individual researcher or a large enterprise, our security features are designed to empower you to protect yourself and your assets. By continually enhancing our defenses and expanding our security capabilities, we aim to stay ahead of emerging threats and maintain the trust of our users. If you have any questions or feedback about our security features, we'd love to hear from you. Reach out at security@huggingface.co!
