---
title: "Give Your Coding Agents a Memory You Own"
thumbnail: /blog/assets/funes/thumbnail.png
authors:
- user: dacorvo
---

# Give Your Coding Agents a Memory You Own

I work across several machines, and I switch coding agents depending on the task.
Every one of them meets my projects as a stranger. The reasoning from “last Tuesday”
disappears when the session ends. Each new agent, on each new host, starts from zero.

Earlier this year, [*Software Forgets: Agent Traces Are the
Memory*](https://huggingface.co/blog/huggingface/agent-traces-as-memory) made the case
that coding agents already produce the record we keep losing. As they search a
codebase, try approaches, hit errors, read documentation, and change direction, they
leave behind a dense account of not just *what* changed, but *why*.

While the diagnosis is correct, traces are only potential memory. The session logs of
an agent are still just an archive. You cannot `grep` your way to *“why did we move
off the streaming parser?”* across ten thousand turns. For an agent to use those traces
while it works, they need **indexing**, **retrieval**, **ranking**, and
**exact provenance**.

That is what [`funes`](https://github.com/huggingface/funes) provides. It is a durable
memory layer for your agents (Claude Code, Codex, pi, and Hermes). It is built from the
sessions already on your machine. It works locally, becomes part of your agent's normal
workflow with one command, and can be published as a Hugging Face **private** dataset
when you want it to travel.

## Add memory to the agent you already use

`funes` is a single binary. Its default inference backend has no ML runtime dependency,
and embedding and reranking happen *on your machine*. Install it, then add it to an
agent:

```bash
curl -fsSL https://huggingface.co/buckets/huggingface/funes/resolve/install.sh | sh
funes add claude    # or: codex, pi, hermes
```

For Claude Code, Codex, and Hermes, that one `add` command builds the first index,
gives the agent `recall` and `get` tools, and installs hooks that index each completed
turn. Indexing is incremental, with new runs adding new turns rather than embedding the
whole history again. The older and deeper content can backfill in bounded steps. With
pi, which has no hook system, `add` installs the read-side tools and `funes index`
refreshes the memory when you run it.

From there, you just work. When a task touches a past decision, rationale, or finding,
the agent can reach for `recall` itself. You do not need to remember the old session or
paste its context into the new one.

![A coding agent reaches for funes on its own, recalls an earlier decision, and grounds its answer in the retrieved session](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/funes/recall.gif)

> [!NOTE]
> With funes added, recall happens inside the conversation. The agent reaches for its
> memory on its own and names the session behind its answer.

`recall` returns the original text, not a summary, and shows exactly where it came from
(the agent, timestamp, session, and turn). Each result includes a `get` command that
opens the full turn and its surrounding context.

Underneath, one deterministic pipeline parses every supported trace into the same
turn-and-block shape, chunks it, embeds it with a pinned local model, and writes it to
a local **Lance** dataset. A query combines vector and BM25 search, fuses their
rankings, reranks the candidates with a cross-encoder, reweights them by recency, and
attaches neighboring chunks.

That design gives funes three important properties:

- **One memory across agents:** Claude Code, Codex, pi, and Hermes all write to the
  same shape. `recall` spans their histories, and every hit says which agent produced
  it.
- **Raw evidence stays intact:** Nothing is distilled into a fact at write time. A
  result can always lead back to the turn that produced it.
- **`recall` is local by default:** No account or Hub repository is required. A hosted
  model does not process your sessions for indexing; embedding and reranking run on
  your machine, and your coding agent does the reasoning.

The *agent as a stranger* problem is already solved on one machine. But memory gets
more useful when the next agent is running somewhere else.

## A memory is a dataset, not a service

To make a memory follow your work, bind it when you add funes to an agent:

```bash
# acme/funes-memory is a memory dataset built in the past
funes add codex acme/funes-memory
```

The agent now recalls from that memory (`acme/funes-memory`). For agents with hooks
(Claude Code, Codex, and Hermes), funes continues to index locally each turn and
publishes at session boundaries. Run the same command on another machine and the memory
follows you there.

Underneath, the local memory is a Lance dataset, and the shared memory is a Hugging
Face dataset (private by default) you own. Publishing is also available directly:

```bash
funes push acme/funes-memory
```

Before anything reaches the Hub, credentials have already been redacted during
indexing. The push then scans every chunk again and withholds anything that still
looks like a secret.

When an agent reads a remote memory, funes caches the dataset files locally, so warm
queries return to local speed. The Hub supplies the ownership, access control,
versioning, and distribution it already supplies for other datasets. Your memory does
not become an account in a separate memory service, and you do not rent it back
through an API.

`recall` is shaped for agents. When you want to put a question to a memory yourself,
use `ask`. It reads your local memory by default:

```bash
funes ask claude "what did we decide about the streaming parser"
```

Or point it at a shared memory. We published funes's development history as the public
[`huggingface/funes-memory`](https://huggingface.co/datasets/huggingface/funes-memory)
dataset, so you can ask why funes works the way it does without creating a memory of
your own:

```bash
funes ask claude "why is funes append-only" --memory huggingface/funes-memory
```

![Asking the published funes memory why it is append-only; funes retrieves the relevant sessions and a coding agent answers from them](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/funes/ask.gif)

> [!NOTE]
> `funes ask` is the read-only, one-question sibling of `funes add`. It recalls the
> passages, hands them to a coding agent, and returns a grounded answer that names its
> sources. It does not install an integration or change the agent's persistent setup.

A retrieval miss is not papered over. If the passages do not support an answer, the
agent says so. You can rephrase the question or add funes to the agent so it can search
the memory iteratively during normal work.

## Switching agents without losing the thread

A shared memory is not tied to the agent or model that created it. Start a task in
Claude Code, continue it in Codex next week, and the second agent can recall the first
agent's reasoning. Use pi with a local model or one served through the Hugging Face
router, then return to Claude.

![Claude Code chooses an embedding model, then Codex recalls that decision in a separate session](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/funes/cross-agents.gif)

*Claude makes a decision; a hook indexes it; Codex recalls it in another session. The
older hits in the demo are earlier recordings of the same experiment — an append-only
memory remembered the rehearsals too.*

This matters in a few different scopes:

- **Across your machines:** Bind each agent to one memory and recall the history from
  whichever host you are using.
- **Across a team:** A new teammate's agent can retrieve months of decisions on day
  one, including dead ends and rationale that never made it into a pull request.
- **Alongside an open-source project:** A maintainer can curate reviewed sessions into
  a project memory — something like a searchable `CLAUDE.md`, except it contains the
  history of why the project is the way it is instead of a page someone must keep
  rewriting. Anyone can read a public project memory with `--memory`.

Published memories carry a dataset card and the `funes` tag, making them recognizable
and [discoverable on the Hub](https://huggingface.co/datasets?other=funes). The Hub
already hosts open weights and open datasets. `funes` adds open working memory. It holds
the decisions, failed approaches, and rationale behind a project, queryable by another
agent and traceable to the sessions that produced them.

> *“To think is to forget differences, generalize, make abstractions.”*
> — Jorge Luis Borges, *Funes the Memorious*

Get started at
[`github.com/huggingface/funes`](https://github.com/huggingface/funes). One `funes add
claude` — or `codex`, `pi`, or `hermes` — connects the agent to the memory. Then you
just work.
