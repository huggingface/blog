---
title: "How Hugging Face Scaled Secrets Management for AI Infrastructure" 
thumbnail: /blog/assets/infisical/thumbnail.png
authors:
- user: guest
---

# How Hugging Face Scaled Secrets Management for AI Infrastructure
Managing secrets at scale becomes increasingly complex as infrastructure grows. For Hugging Face, this challenge intensified as their platform scaled to support over 4 million AI builders deploying models on the Hub. TThis case study explores how they approached secrets management to support their growing infrastructure needs.

## Technical Challenge
As Hugging Face's infrastructure scaled to support millions of model deployments, their infrastructure and engineering teams identified security and operationnal challenges.

### Security Risk Management
Being at the forefront of AI development, Hugging Face needed to ensure their security infrastructure exceeded industry standards. This included:
- Maintaining tight access controls across their infrastructure
- Implementing a "Security Shift Left" approach
- Establishing comprehensive audit capabilities

### Secret Sprawl
With increasing infrastructure complexity and new engineering projects, [secret sprawl](https://infisical.com/blog/what-is-secret-sprawl) became a significant concern. The team needed to:
- Automate secrets management processes
- Streamline secret deployment workflows
- Establish a single source of truth for credentials

### Developer Experience
Supporting a large engineering team required maintaining developer productivity through:
- Self-serve secret management workflows
- Efficient developer onboarding processes
- Streamlined local development setup

## Solution
To solve the above, Hugging Face partnered with [Infisical](https://infisical.com/) to centralize its secrets management workflows and establish a single source of truth for infrastructure credentials, with several key technical components involved:

### Kubernetes Integration
With Kubernetes being central to Hugging Face's infrastructure, they implemented Infisical's [Kubernetes Operator](https://infisical.com/docs/integrations/platforms/kubernetes) to:
- Automatically propagate secrets to containers
- Handle application redeployments based on secret updates
- Maintain consistent secret management across clusters

### Local Development Workflow
For local development environments, the team utilized the [Infisical CLI](https://infisical.com/docs/cli/usage) to:

- [Inject secrets](https://infisical.com/docs/cli/commands/run) into local application environments
- Eliminate the need for local [.env files](https://infisical.com/blog/stop-using-env-files)
- Reduce security risks from secrets on local machines

### Centralized Management
The team established a central secrets management system using:

- A [web dashboard](https://infisical.com/docs/documentation/platform/project) enabling self-serve secrets management
- [Role-based access controls](https://infisical.com/docs/documentation/platform/access-controls/role-based-access-controls#role-based-access-controls) for different teams
- [Secret referencing and importing](https://infisical.com/docs/documentation/platform/secret-reference) capabilities for maintaining a single source of truth across infrastructure.
- [Secret Sharing](https://infisical.com/docs/documentation/platform/secret-sharing) to generate encrypted links to share secrets with each other or with stakeholders outside of the organization.

## Results and Impact

With the help of Infisical, Hugging Face was able increase both operational efficiency and security posture through centralized secrets management.

### Developer Workflow Efficiency
The new system improved development workflows through:

- Self-serve secrets management based on permissions. This saves developers time and speeds up development iterations.
- Faster developer onboarding: new engineers are now able to immediately get up and running with access to the necessary environments.
- Synchronized secrets across team environments: engineers easily check out the right environment and start their applications locally.
- Automated application redeployments: using Infisical, Hugging Face is able to automatically redeploy their applications based on secret changes in various environments.

### Security Improvements

Security is often a matter of making the secure path the easiest path. Beyond all the points mentioned above, the following measures helped strengthen Hugging Face's security posture regarding secrets management:

- Implemented tight and granular access controls
- Established comprehensive audit logging
- Integrated secure authentication methods
- Enhanced security through centralized management

### Security Culture Enhancement

Finally, the implementation helped foster better security practices by:

- Enabling secure secret sharing via encrypted channels
- Promoting responsible coding practices
- Implementing permission-based access controls

## Technical Insights
As noted by **Adrien Carreira**, Head of Infrastructure at Hugging Face:

> "Infisical provided all the functionality and security settings we needed to boost our security posture and save engineering time. Whether you're working locally, running kubernetes clusters in production, or operating secrets within CI/CD pipelines, Infisical has a seamless prebuilt workflow."

The implementation demonstrated that proper secrets management can simultaneously enhance security and developer productivity - a rare combination in infrastructure tooling.

## Resources
For teams looking to implement similar solutions:

- [Platform Documentation](https://infisical.com/docs/documentation/platform/organization)
- [CLI Reference](https://infisical.com/docs/cli/overview)
- [Kubernetes Integration Guide](https://infisical.com/docs/integrations/platforms/kubernetes/overview)
- [Secret Reference Documentation](https://infisical.com/docs/documentation/platform/secret-reference)

---

*This technical case study was adapted from the original customer story published at [infisical.com/customers/hugging-face](https://infisical.com/customers/hugging-face)*