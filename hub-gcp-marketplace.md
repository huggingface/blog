---
title: Pay for Hugging Face Hub with your Google Cloud Account
thumbnail: /blog/assets/hub-gcp-marketplace/thumbnail.jpg
authors:
  - user: zaid-alhadad
  - user: pierric
  - user: pagezyhf
  - user: michellehbn
---

# Pay for Hugging Face Hub with your Google Cloud Account

You can now link your Hugging Face organization to your Google Cloud account, so that your Hugging Face usage shows up on your Google Cloud bill instead of being charged to a credit card. Get started [on the Google Cloud Marketplace](https://console.cloud.google.com/marketplace/product/huggingface-public/hugging-face-hub).

## Why pay through Google Cloud?

Hugging Face is where teams find, build and run open models: from hosting private models and datasets, to running GPU workloads with [Jobs](https://huggingface.co/docs/hub/jobs-overview), deploying models with [Inference Endpoints](https://huggingface.co/docs/inference-endpoints), calling thousands of models through [Inference Providers](https://huggingface.co/docs/inference-providers), or hosting demos on GPU [Spaces](https://huggingface.co/docs/hub/spaces-overview).

If your company already runs on Google Cloud, adding a new vendor and a corporate credit card just to use these services can slow things down. By linking your organization through the Google Cloud Marketplace:

- **One bill**: your Hugging Face usage appears on your existing Google Cloud invoice, next to the rest of your cloud spend.
- **No new vendor to onboard**: purchases go through your existing Google Cloud billing account and procurement process.
- **Everything is covered**: upgrading your org to  [Team or Enterprise](https://huggingface.co/enterprise) subscription, pay-as-you-go compute (Jobs, Inference Endpoints, Inference Providers, GPU Spaces, ZeroGPU quota extension) and additional storage at  the [public Hugging Face pricing](https://huggingface.co/pricing).

If you're an admin of your Hugging Face organization, here's how to set it up, step by step.

### 1. Subscribe to Hugging Face on the Google Cloud Marketplace

Go to the [Hugging Face Hub listing](https://console.cloud.google.com/marketplace/product/huggingface-public/hugging-face-hub) on the Google Cloud Marketplace, select the Google Cloud project you want to use, and click on "Subscribe".

![Hugging Face Hub listing on the Google Cloud Marketplace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-product-listing.png "Hugging Face Hub listing on the Google Cloud Marketplace")

You are now on the subscription page. Select the billing account to be charged, review and accept the terms, then click "Subscribe".

![Subscription page with billing account selection and terms](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-subscribe-page.png "Subscription page with billing account selection and terms")

Once the order is placed, Google Cloud asks you to finish setting up your account with Hugging Face. Click on "Sign up with Hugging Face" to be redirected to Hugging Face.

![Order request sent to Hugging Face, with a "Sign up with Hugging Face" button](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-order-sent-signup.png "Order request sent to Hugging Face")

### 2. Link your Hugging Face organization

You are now on Hugging Face, where you can select the organization you want to link to your Google Cloud account. Only organizations where you are an admin are listed. Select it and confirm.

![Selecting the Hugging Face organization to link to the Google Cloud account](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/select-organization-to-link.png "Selecting the Hugging Face organization to link")

You are then redirected to the billing settings of your organization. The payment method section now shows the linked Google Cloud account, with a pending status.

![Organization billing settings showing the linked Google Cloud account as pending](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/billing-payment-method-pending.png "Linked Google Cloud account pending validation")

Every cloud marketplace subscription is reviewed by the Hugging Face team before paid usage starts being billed through it. While the subscription is pending, usage keeps being billed to the organization's existing payment method, if any.

### 3. Subscription approved

Once your subscription is approved, you will receive an email confirming that your Google Cloud account and Hugging Face organization are linked. Your usage will be billed to your GCP account starting the first day of the next month.

![Email confirming the Marketplace account was approved as a payment method](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/approval-confirmation-email.png "Approval confirmation email")

To confirm it, open the billing settings of [your organization](https://huggingface.co/settings/organizations): the Google Cloud account is now shown as the organization's active payment method.

![Organization billing settings showing the Google Cloud account as the active payment method](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/billing-payment-method-active.png "Google Cloud account active as payment method")

### Congratulations!

Your organization is now billed through your Google Cloud account. All members of your organization can use Hugging Face compute services, and the usage will show up on your Google Cloud bill.

A few things worth knowing:

- **Google Cloud replaces the credit card** as the organization's payment method. Usage is metered and reported to Google Cloud, so adding credits is not available while the organization is billed through the marketplace.
- **Included credits still apply**: if your organization has a Team or Enterprise subscription, the monthly included credits are applied before any pay-as-you-go usage is reported to Google Cloud.
- **Monitor your usage** at any time from the billing section of your [organization settings](https://huggingface.co/settings/organizations), and find the charges in the [Billing reports](https://console.cloud.google.com/billing) of the Google Cloud console.
- **Linking and unlinking are audited**: both actions are recorded in your organization's [audit logs](https://huggingface.co/docs/hub/audit-logs).

---

Thanks for reading! You can find more details in the [billing documentation](https://huggingface.co/docs/hub/billing#cloud-providers-partnerships). If you have any questions, please contact us at [billing@huggingface.co](mailto:billing@huggingface.co).