---
title: "Subscribe to Enterprise Hub with your AWS Account"
thumbnail: /blog/assets/158_aws_marketplace/thumbnail.jpg
authors:
- user: Violette
- user: sbrandeis
- user: jeffboudier
---

# Subscribe to Enterprise Hub with your AWS Account

You can now upgrade your Hugging Face Organization to Enterprise using your AWS account - get started [on the AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n6vsyhdjkfng2).

## What is Enterprise Hub? 

[Enterprise Hub](https://huggingface.co/enterprise) is a premium subscription to upgrade a free Hugging Face organization with advanced security features, access controls, collaboration tools and compute options. With Enterprise Hub, companies can build AI privately and securely within our GDPR compliant and SOC2 Type 2 certified platform. Exclusive features include: 
- Single Sign-On: ensure all members of your organization are employees of your company.
- Resource Groups: manage teams and projects with granular access controls for repositories.
- Storage Regions: store company repositories in Europe for GDPR compliance.
- Audit Logs: access detailed logs of changes to your organization and repositories.
- Advanced Compute Options: give your team higher quota and access to more powerful GPUs.
- Private Datasets Viewer: enable the Dataset Viewer on your private datasets for easier collaboration.
- Train on DGX Cloud: train LLMs without code on NVIDIA H100 GPUs managed by NVIDIA DGX Cloud.
- Premium Support: get the most out of Enterprise Hub and control your costs with dedicated support.

If you're admin of your organization, you can upgrade it easily with a credit card. But how do you upgrade your organization to Enterprise Hub using your AWS account? We'll walk you through it step by step below.

### 1. Getting Started

Before you can connect your AWS Account with your Hugging Face account, you need to fulfill the following prerequisites: 

- Have access to an active AWS account with access to subscribe to products on the AWS Marketplace.
- Create a [Hugging Face organization account](https://huggingface.co/organizations/new) with a registered and confirmed email. (You cannot connect user accounts)
- Be a member of the Hugging Face organization you want to connect with the [“admin” role](https://huggingface.co/docs/hub/organizations-security).
- Logged into the Hugging Face Platform.

Once you meet these requirements, you can proceed with connecting your AWS and Hugging Face accounts.

### 2. Connect your Hugging Face Account with your AWS Account

The first step is to go to the [AWS Marketplace offering](https://aws.amazon.com/marketplace/pp/prodview-n6vsyhdjkfng2) and subscribe to the Hugging Face Platform. There you open the [offer](https://aws.amazon.com/marketplace/pp/prodview-n6vsyhdjkfng2) and then click on “View purchase options” at the top right screen. 

![Marketplace Offer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/01_bis_offering.jpg "Marketplace Offer")

You are now on the “subscribe” page, where you can see the summary of pricing and where you can subscribe. To subscribe to the offer, click “Subscribe”. 

![Marketplace Subscribe](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/02_bis_subscribe.jpg "Marketplace Subscribe")

After you successfully subscribe, you should see a green banner at the top with a button “Set up your account”. You need to click on “Set up your account” to connect your Hugging Face Account with your AWS account.  

![Marketplace Redirect](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/03_bis_redirect.jpg "Marketplace Redirect")

After clicking the button, you will be redirected to the Hugging Face Platform, where you can select the Hugging Face organization account you want to link to your AWS account. After selecting your account, click “Submit” 

![Connect Account](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/04_connect.jpg "Connect Account")

After clicking "Submit", you will be redirected to the Billings settings of the Hugging Face organization, where you can see the current state of your subscription, which should be `subscribe-pending`.

![Subscription Pending](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/05_pending.jpg "Subscription Pending")

After a few minutes, you should receive 2 emails: 1 from AWS confirming your subscription and 1 from Hugging Face, which should look like the image below:

![Email confirmation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/07_email.jpg "Email confirmation")

If you have received this, your AWS Account and Hugging Face organization account are now successfully connected! 
To confirm it, you can open the Billing settings for [your organization account](https://huggingface.co/settings/organizations), where you should now see a `subscribe-success` status.

![Subscription Confirmed](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/06_success.jpg "Subscription Confirmed")

### 3. Activate the Enterprise Hub for your team and unlock new features

If you want to enable the Enterprise Hub and use your organization as a private and safe collaborative platform for your team to build AI with open source, please follow the steps below.

Open the Billing settings for your organization, click on the ‘Enterprise Hub’ Tab, and click on “Subscribe Now”

![Subscribe Now](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/08_subscribe.jpg "Subscribe Now")

Now select the number of Enterprise Hub seats you are willing to buy for your organization, the billing frequency and click on Checkout. 

![Select Seats](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/09_select.jpg "Select Seats")


### Congratulations! 🥳

Your organization is now upgraded to Enterprise Hub, and its billing is directly managed by your AWS account. All members of your organization can now benefit from the advanced features of Enterprise Hub to build AI privately and securely.

The pricing for Hugging Face Hub through the AWS marketplace offer is identical to the [public Hugging Face pricing](https://huggingface.co/pricing), but you will be billed through your AWS Account. You can monitor your organization's usage and billing anytime within the Billing section of your [organization settings](https://huggingface.co/settings/organizations).

---
Thanks for reading! If you have any questions, please contact us at [api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co).
