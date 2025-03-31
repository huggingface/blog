---
title: "How Hugging Face Scaled Secrets Management for AI Infrastructure" 
thumbnail: /blog/assets/infisical/thumbnail.png
authors:
- user: segudev
  guest: true
  org: Infisical
---

# How Hugging Face Scaled Secrets Management for AI Infrastructure

Hugging Face has become synonymous with advancing AI at scale. With over 4 million builders deploying models on the Hub, the rapid growth of the platform necessitated a rethinking of how sensitive configuration data —secrets— are managed.

Last year, the engineering teams set out to improve the handling of their secrets and credentials. After evaluating tools like HashiCorp Vault, they ultimately chose [Infisical](https://infisical.com/).

This case study details their migration to Infisical, explains how they integrated its powerful features, and highlights how it enabled engineers to work more efficiently and securely.

## Background

As Hugging Face's infrastructure evolved from an AWS-only setup to a multi-cloud environment that includes Azure and GCP, the engineering team needed a more agile, secure, and centralized way to manage secrets. Instead of reworking legacy systems or adopting heavyweight solutions like HashiCorp Vault, they turned to Infisical due to its developer-friendly workflows, multi-cloud abstraction, and robust security capabilities.

The key challenges they faced were:

- An increased risk of “[secret sprawl](https://infisical.com/blog/what-is-secret-sprawl)” due to inconsistent management across environments.
- Complex permission management as the team scaled, requiring tight, role-based access controls (RBAC) integrated with the organization’s SSO (Okta).
- Difficulties with local development where traditional [.env files](https://infisical.com/blog/stop-using-env-files) compromised both security and developer productivity.
- The burden of manual secret rotation, which became painfully evident after a security incident that involved exposed credentials.

In addition, the team needed a solution that adhered to infrastructure-as-code practices, supported project-by-project secret management, and provided a smooth balance between automation and manual control during deployments.

## Implementation

Infisical’s flexible architecture was an ideal solution. The engineering team seized the opportunity to re-examine their internal project structure, splitting projects into distinct infrastructure and application domains. This allowed them to implement a clearer separation of concerns and standardize secret rotation practices—a priority in the wake of a recent security incident.

By leveraging Terraform, which was previously used to create Kubernetes secrets from AWS configurations, they found the transition to the Infisical Kubernetes Operator exceptionally smooth. This integration enabled security improvements while standardizing secrets management across all environments.

### Kubernetes Integration

Kubernetes is at the heart of Hugging Face’s production environment, and Infisical's [Kubernetes Operator](https://infisical.com/docs/integrations/platforms/kubernetes) has been instrumental in automating secret updates. The Operator continuously monitors for changes to any secret in Infisical and ensures that these updates are propagated to the corresponding Kubernetes objects. Whenever a change is detected, it can automatically reload dependent Deployments, ensuring that containers always run with the most recent secrets.

**Example:**

A new secret is required by an application running in Kubernetes. The secret can be created via the Infisical's CLI or the web UI, then the developer creates an `InfisicalSecret` resource in Kubernetes that specifies which secret from Infisical should be synced: 

```yaml
apiVersion: infisical.com/v1alpha1
kind: InfisicalSecret
metadata:
  name: my-app-secret
  namespace: production
spec:
  infisicalSecretId: "123e4567-e89b-12d3-a456-426614174000"
  targetSecretName: "my-app-k8s-secret"
```
Once the CRD is applied, the Infisical Operator continuously watches for updates. When changes are detected in Infisical, the Operator automatically updates the Kubernetes secret (`my-app-k8s-secret`).

<figure class="image text-center">
    <img class="mx-auto" src="/blog/assets/infisical/infisical-operator.png" alt=" Infisical secrets management flow in Kubernetes">
    <figcaption>Secrets management flow with Infisical: When a secret is updated in the Infisical Platform, the Infisical Operator automatically syncs the changes to the referenced Kubernetes secret, which is then made available to the application.</figcaption>
</figure>

Better yet, since the application's Deployment references `my-app-k8s-secret` as an environment variable source or mounted volume, the Operator can automatically trigger a container reload when the secret changes.

In practice, Hugging Face engineers favor waiting for manual redeployments despite the Operator’s ability to automatically trigger container restarts. This decision was driven by the need for precise control over deployments, particularly when high traffic (over 10 million requests per minute) and numerous replicas are involved.

### Local Development

For local development, [Infisical’s CLI](https://infisical.com/docs/cli/usage) streamlines workflows by injecting secrets directly into development environments. This removes the need for insecure local .env files, aligning local configurations with production standards and reducing onboarding friction.

## Security and Access Management

Security improvements form the backbone of this migration. By integrating Infisical with existing identity providers such as Okta, Hugging Face established a fine-grained RBAC system. Permissions are automatically mapped from Okta groups, ensuring that developers retain administrative rights over their projects, while frontend and backend teams receive appropriately restricted read or write access.

Additionally, the [secret sharing](https://infisical.com/docs/documentation/platform/secret-sharing) functionality allows secure credentials sharing among ML/AI researchers at Hugging Face. The centralized Infisical platform also simplifies auditing and managing secret rotations—a necessity highlighted by previous security incidents.

## CI/CD and Infrastructure Integration

Seamless integration with CI/CD pipelines further enhanced the overall security posture. Infisical was embedded into the deployment pipeline via GitHub Actions using [OIDC authentication](https://infisical.com/docs/documentation/platform/identities/oidc-auth/github) and Terraform integration. By operating self-hosted runners within a secure environment, every deployment adhered to production-grade security standards. This integrated approach minimized risks and ensured a uniform experience from local development to cloud deployment.

## Technical Outcomes & Insights

Centralizing secrets management with Infisical brought tangible improvements:

- Engineers no longer need to spend valuable time manually configuring environment secrets. Self-serve workflows accelerated onboarding and daily development cycles. 
- Automated audits and fine-grained access controls enabled rapid incident response and promoted a “shift left” approach to security.  
- Consistent integration across cloud providers, Kubernetes clusters, and CI/CD pipelines eliminated discrepancies in secret management, thus reinforcing the infrastructure's security and reliability.

As noted by Adrien Carreira, Head of Infrastructure at Hugging Face,

>"Infisical provided all the functionality and security settings we needed to boost our security posture and save engineering time. Whether you're working locally, running kubernetes clusters in production, or operating secrets within CI/CD pipelines, Infisical has a seamless prebuilt workflow."

## Conclusion

Hugging Face's migration to Infisical demonstrates how a technically driven, engineering-centric approach to managing secrets across multiple cloud platforms delivers significant benefits. For tackling similar challenges, using Infisical is a practical way to work more efficiently while keeping security strong.

When the secure path is made the easiest path, teams can focus on building innovative products instead of worrying about managing secrets.

## Resources

For teams interested in adopting a similar approach:
- [Secure GitOps Workflows: A Practical Guide to Secrets Management](https://infisical.com/blog/gitops-secrets-management)
- [Kubernetes Secrets Management in 2025 - A Complete Guide](https://infisical.com/blog/kubernetes-secrets-management-2025)
- [Platform Documentation](https://infisical.com/docs/documentation/platform/organization)
- [CLI Reference](https://infisical.com/docs/cli/overview)

---

*This technical case study was adapted from the original case study published at [infisical.com/customers/hugging-face](https://infisical.com/customers/hugging-face)*
