---
title: "When an agent goes rogue, the question is not who it is. It is what it did next."
thumbnail: /blog/assets/salt-rogue-agent-actions/thumbnail.png
authors:
- user: relayshieldadmin
  guest: true
---

# When an agent goes rogue, the question is not who it is. It is what it did next.

Agent builders publish the pieces rogue agents are built from: models, Spaces, and MCP servers. Salt Security's 2H 2026 report gives that a number: 49.8% of firms confirmed or suspected an agent took an unintended, unexpected, or unauthorized action in the past year. Those agents rarely break in through a login. They break through a tool call, which is where screening has to sit.

Half of firms have already lived the version of agent risk that no identity check, by itself, can stop.

Salt Security reports that 49.8 percent of respondents confirmed or suspected that an AI agent in their organization took an action in the past year that was not intended, expected, or authorized. Among that group, more than half reported an unauthorized external API call or transaction, and more than 40 percent reported an agent modifying or deleting data.

That is not a story about a stranger breaking in. It is a story about an authorized agent taking an unauthorized step.

## Adoption has outrun tracing

The same report puts the scale in plain terms. 86.4 percent of respondents report at least one AI agent operating in their organization, and 59.5 percent run more than ten. 76.8 percent report agents that chain actions across other agents, tools, and APIs, creating sequences that no single team designed in advance.

And yet only 12.5 percent say they can consistently trace a complete agent journey, from its initial instruction through the tools and APIs it accesses to the final outcome. More than 60 percent would need hours or days of manual investigation to reconstruct what a specific agent did the previous day. Only 17.5 percent can automatically stop a misbehaving agent in production.

Read those three numbers together and the gap is clear. Agents are fast, chained, and consequential. Tracing is slow, manual, and partial. Containment is, for most firms, not automatic at all.

If you cannot reconstruct yesterday, you cannot contain today, and you cannot prove what happened after the fact. That is the operational cost of a rogue action, even when the damage itself is small.

## Identity is necessary. It is not sufficient.

The last week of agent security news makes the same point from three different directions, and each direction leaves a different hole.

Identity answers who the agent is, who owns it, and what authority it was granted. That matters, and it is the right first layer. An agent with no owner, no lifecycle status, and no delegated authority is ungovernable by definition.

Isolation answers what code can reach. A sandboxed agent, with limits on compute, memory, system APIs, and outbound connections, is harder to turn into a general purpose foothold. That matters too.

Neither answers the question Salt's respondents are actually stuck on: what did this agent touch, in what order, and was that step inside the intent it was given?

A verified agent can still call the wrong tool. An isolated agent can still, within its allowed reach, call a known malicious MCP server, a phishing domain, a compromised wallet, or an API it was never meant to transact with. The step that causes the incident is often not the login, and not the code execution. It is the tool call.

## Three checks, at the point where the action happens

That is where the RelayShield story fits, and it is worth stating it as three separate checks, because collapsing them into one is how the gap reappears.

First, who is this agent? TAP and PAP verification establish an agent's identity and its verification outcomes, in a form that can be checked again later, not just at onboarding.

Second, what does it intend to do? Intent binding ties a specific payment and a specific action together before settlement, so the intent is recorded at the moment money moves, not reconstructed from logs afterwards. The console panel for intent bindings shows those settle-check decisions, so a binding, and a decision about it, are visible rather than implied.

Third, what is it actually touching, right now? The MCP Proxy Firewall screens at the tool call, not only at discovery. Discovery screening can flag a suspicious server, tool description, or configuration before it is used. Runtime screening is where a call to a known threat, a misconfigured server, or a shadow-installed server can be blocked at the point of use, instead of being discovered in a manual review the next day.

Put together, the sequence a team needs after an incident is short: the instruction, the verified identity, the bound intent, the tool calls in order, and the screening decision on each call. If any one of those is missing, reconstruction becomes the hours or days of manual work Salt describes. If all of them are present, the same incident becomes a traceable, stoppable event.

## The test to run on your own agents

You do not need Salt's survey to test this. Pick one agent that runs in production, and ask four questions about yesterday:

1. Who owned it, and was its identity verified?
2. What did it intend to do, and was that intent bound to the action it took?
3. What tools and APIs did it actually call, in order?
4. Could you have stopped the next call automatically, or would you have had to shut down the application?

If the answer to the third question takes hours, or the answer to the fourth is manual shutdown, then your agent program has the same gap Salt measured, even if you have not yet had a rogue action you know about. The 49.8 percent figure includes suspected cases for a reason: many firms cannot yet tell the difference between a clean day and an unnoticed one.

That is the discipline agent security has to become. Not fewer agents, and not slower agents, but agents whose actions are verified, bound, screened at the tool call, and reconstructable the next morning without a forensic project.

---
*Sources: Salt Security, State of Agentic AI and API Security 2H 2026, Oct 13, 2026, via IT Brief and Security Today coverage. All survey figures in this post are Salt's, attributed to Salt, and include confirmed or suspected cases as Salt reported them.*

*Originally published at [blog.relayshield.net/salt-rogue-agent-actions](https://blog.relayshield.net/salt-rogue-agent-actions).*
