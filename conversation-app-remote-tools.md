---
title: "Remote Tools for Reachy Mini Conversation App"
thumbnail: /blog/assets/conversation-app-remote-tools/reachy_mini_remote_spaces_thumbnail.png
authors:
- user: alozowski
---

# Remote Tools for Reachy Mini Conversation App

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/conversation-app-remote-tools/reachy_mini_window.jpg" alt="Reachy Mini looking out the window">
  <figcaption><em>Reachy Mini no longer has to look out the window to tell you the weather</em></figcaption>
</figure>

The Reachy Mini conversation app can now use tools hosted in public Hugging Face Spaces, called over MCP. You can give your robot a new ability, like checking the weather or searching the web, by adding a Space from the Hub instead of editing the app. The tool keeps running in the Space itself, so no code is downloaded onto your machine. And you can publish your own tools for other people to use.

Adding a tool takes one command:

```
reachy-mini-conversation-app tool-spaces add pollen-robotics/reachy-mini-weather-tool
```

Then start the app as usual:
```
reachy-mini-conversation-app
```

Now you can just ask:

```
What's the weather in Paris today?
```

Below, we look at what a tool is, how profiles control what the robot can use, and the current limits of the remote path.

## Built-in tools

When you talk to the robot, what you get back isn't only a voice, it's a system that reacts to the conversation: the robot can move and respond non-verbally, when it's applicable. The part we want to focus on here is the tools that make that possible. A tool is something the model can do during a conversation: play an emotion, move the head, look through the camera. Each tool has a name and a short description. The model reads those, decides when one is useful, calls it, and uses what comes back.

Today every tool is local and ships inside the app, and most of them are about the robot's body:

| Tool | What it does |
| --- | --- |
| `move_head` | Queue a head pose change |
| `dance` / `stop_dance` | Play or clear a dance from the dances library |
| `play_emotion` / `stop_emotion` | Play or clear a recorded emotion clip |
| `head_tracking` | Toggle head-tracking offsets |
| `camera` | Capture a frame and analyze it |
| `idle_do_nothing` | Explicitly stay idle on an idle turn |
 
## How profiles control tools

A tool in the code isn't usable until it's enabled in **a profile**, a folder with two files that matter here: `instructions.txt` (the prompt) and `tools.txt` (the tools that are turned on).

The `default` profile enables the full set:

```
# profiles/default/tools.txt
dance
stop_dance
play_emotion
stop_emotion
camera
idle_do_nothing
head_tracking
move_head
```

If a name isn't in `tools.txt`, the model can't call it.

You can also write your own tool: add a Python file to the profile (or `external_tools/`), give it a name and description, and list that name in `tools.txt`.

Today there are built-in tools and custom local tools, and `tools.txt` decides which are active. This works well for the robot's body and keeps the trusted core small.

## The limits of local tools

The constraint here is that every tool has to be local Python. For `move_head` or `play_emotion` that's right: they talk to the hardware and belong in the app but a lot of useful things have nothing to do with the body, like web search, weather, or lookups. For those, keeping everything local is mostly friction:

- sharing a tool means handing someone your Python files
- updating it means sending those files again
- changing it means editing the app, even though the capability is really separate from it

## Calling tools from Spaces

Remote tools add a third kind, alongside the built-in and custom local tools you already have, for capabilities that are easier to publish, share, and update on their own:

- built-in robot tools stay local and trusted
- shareable remote tools can live in public Hugging Face Spaces
- you can still use custom one-off tools from `external_tools/`
It's a good fit for stateless capabilities like search, weather, and lookups: anything you want to iterate on without touching the app itself. And because anyone can publish a compatible Space, it's easy to share tools and build on each other's work.

We started with two canary tools, small test tools to exercise the new flow:
- [pollen-robotics/reachy-mini-search-tool](https://huggingface.co/spaces/pollen-robotics/reachy-mini-search-tool)
- [pollen-robotics/reachy-mini-weather-tool](https://huggingface.co/spaces/pollen-robotics/reachy-mini-weather-tool)

They're enough to exercise the whole feature: install from the Hub, discover the remote tools, enable them per profile, and let the realtime backend call them exactly like built-in tools.

To use both at once, add each Space and their tools stack in the same profile:

```
reachy-mini-conversation-app tool-spaces add pollen-robotics/reachy-mini-search-tool
reachy-mini-conversation-app tool-spaces add pollen-robotics/reachy-mini-weather-tool
```

Now the robot can search the web and check the weather in the same conversation, which is exactly what the `canary_web_search_weather` profile below does.
 
## Install, list, remove
 
```
# install + enable in active profile
reachy-mini-conversation-app tool-spaces add <owner/space-name>
 
# enable in a specific profile
reachy-mini-conversation-app tool-spaces add <owner/space-name> --profile <NAME>
 
# install without enabling
reachy-mini-conversation-app tool-spaces add <owner/space-name> --install-only
 
# list installed spaces
reachy-mini-conversation-app tool-spaces list
 
# remove an installed space
reachy-mini-conversation-app tool-spaces remove <owner/space-name>
```

`add` validates the Space on the Hub, probes the MCP endpoint, discovers its tools, and by default appends the tool IDs to the active profile's `tools.txt`. The active profile is `default` unless you've set `REACHY_MINI_CUSTOM_PROFILE`. Use `--install-only` to skip that step.

> `tools.txt` is the gatekeeper: a remote tool is only active if its ID appears in the profile's `tools.txt`, alongside whatever built-in tools you want.
 
### Where the manifest lives

Installed sources are persisted in:
- `installed_tool_spaces.json` in managed app mode
- `external_content/installed_tool_spaces.json` in terminal mode

## Tool naming

Each installed Space gets a local alias derived from its slug, with hyphens, dots, and slashes collapsing to underscores:

```
pollen-robotics/reachy-mini-search-tool → pollen_robotics_reachy_mini_search_tool
```

Remote tools are then namespaced with a double underscore:

```
pollen_robotics_reachy_mini_search_tool__search_web
pollen_robotics_reachy_mini_weather_tool__get_day_brief
```

This keeps remote tool names from colliding with built-in ones and lets multiple Spaces coexist in the same profile.

The implementation also strips redundant Space-name prefixes when possible, so a verbose remote tool name becomes a cleaner local ID. If stripping would cause a collision between two tools from the same Space, the code falls back to the fully namespaced name.

There is also a duplicate safety check at registry level: `Tool.name` values must be unique across the entire merged tool set. The app fails fast if two sources claim the same name.

## Example profiles

For this work we created two focused canary profiles to isolate the MCP experiment from the full embodied tool set.

The first keeps a few expressive tools (emotions, head movement) and adds web search on top:

```
# profiles/canary_web_search/tools.txt
play_emotion
stop_emotion
idle_do_nothing
move_head
pollen_robotics_reachy_mini_search_tool__search_web
```

The second is the same, plus the weather tool alongside search:

```
# profiles/canary_web_search_weather/tools.txt
play_emotion
stop_emotion
idle_do_nothing
move_head
pollen_robotics_reachy_mini_search_tool__search_web
pollen_robotics_reachy_mini_weather_tool__get_day_brief
```

The small physical tool set means Reachy Mini can still react expressively while answering current questions from the web.

## Why the prompts matter

The remote-tool plumbing gets the tools into the model. The prompts decide how the model uses them.

That was especially visible in the search-plus-weather canary. A combined question like:

```
Should I bring a jacket in Bordeaux today, and is there anything major happening downtown tonight?
```

can be handled in at least three ways: weather first then search, search first then weather, or both in the same turn. If the prompt is vague, the model serialises the calls and creates unnecessary latency. So the canary prompts became part of the feature, not just incidental configuration.

#### `canary_web_search/instructions.txt`

```
[default_prompt]

## CANARY WEB SEARCH RULES
You have one remote tool for current web information.
Use it when the user asks for up-to-date facts, news, live availability, or anything else that may have changed recently.

When the search result already answers the question, answer directly in plain language.
Lead with the answer, not with tool chatter.
For remote lookups that may take a moment, you may give one very short English acknowledgment such as "Let me check that and I'll be right back," then continue.
Answer in English unless the user explicitly asks for another language.
Mention uncertainty briefly if the result snippet is incomplete or ambiguous.
Only mention links when they add value or the user asks for sources.

Keep responses short and spoken-style, as if read aloud by a voice assistant. One or two sentences is usually enough. Skip preamble, lists, headers, and filler. Give just the fact or direct answer the user needs.
```

#### `canary_web_search_weather/instructions.txt`

```
[default_prompt]

## CANARY SEARCH AND WEATHER RULES
You have two remote tools:
- a weather brief tool for compact day weather at a location
- a web search tool for broader current web information

Use the weather tool for today's conditions, temperature, rain chance, sunrise, sunset, or simple advice like whether to bring a jacket.
Use web search for news, events, business hours, travel information, severe alerts, or broader current context.

When the user's question mixes a weather part and a current-info part (for example, "should I bring a jacket in Bordeaux today, and is there anything major happening downtown tonight?"), call both tools in parallel in the same turn. Do not wait for one result before starting the other unless the weather result is needed to narrow the search.

Then merge the results into a single short answer. Cover the weather part first, then the events or news part, in plain connected sentences. Do not label the sections or mention which tool gave which piece.

When the user asks about events, news, or what is happening, give them the actual answer from the search results: name specific events, venues, or headlines. Do not tell the user to check websites, visit listing sites, or look something up themselves. If the search returns nothing concrete, say plainly that you didn't find any notable events, rather than redirecting them elsewhere.

For remote lookups that may take a moment, you may give one very short English acknowledgment such as "Let me check that and I'll be right back," then continue.
Answer in English unless the user explicitly asks for another language.
Do not talk about tool usage unless the user asks.

Keep responses short and spoken-style, as if read aloud by a voice assistant. One or two sentences is usually enough. Skip preamble, lists, headers, and filler. Give just the fact or direct answer the user needs.
```

## What works today, and what doesn't

| Capability | Supported |
| --- | --- |
| Install by slug for public, MCP-compatible Gradio Spaces (standard `/gradio_api/mcp/` endpoint) | ✅ |
| Multiple Spaces at once | ✅ |
| Per-profile enablement via `tools.txt` | ✅ |
| Namespaced remote tool IDs | ✅ |
| Backend-agnostic registration (OpenAI, Gemini, Hugging Face) | ✅ |
| No arbitrary code downloaded into the local app | ✅ |
| Private or authenticated Spaces | ❌ |
| Non-Gradio Spaces | ❌ |
| Arbitrary raw MCP URLs or non-Hugging Face MCP servers | ❌ |
| Guaranteed parallel tool orchestration | ❌ |

Two things are worth calling out. First, the Space has to actually behave like an MCP server; if tool discovery fails, the install fails. Second, prompt instructions can encourage parallel calls but cannot guarantee them. If deterministic orchestration matters for a use case, that logic should move from the prompt into code.

## Tips for publishing a tool Space

If you want others to use your tool, publish it as a public Gradio Space that exposes the standard MCP endpoint, and keep the tools stateless so they work well over the network. Whether a Space installs depends on this runtime behavior, not on tags.

Tags aren't required for installation, but they help people find compatible Spaces:

- `reachy-mini-tool`
- `mcp`

## Conclusion

The app now has three kinds of tools sharing one registry: built-in, local custom, and remote MCP tools, and profiles still decide which of them a given assistant can reach. A small, trusted core stays at the center while the optional capabilities around it can be added, tested, and swapped without touching the app itself.

What we're most curious about now is what people build. If you publish a tool Space, tag it `reachy-mini-tool` and `mcp` so others can find it. We'd love to see what Reachy Mini ends up able to do!

*Acknowledgements: Many thanks to [Fabien Danieau](https://huggingface.co/FabienDanieau) for proofreading this post and helping test the workflow, to [Andres Marafioti](https://huggingface.co/andito) for helping test it, and to [Remi Fabre](https://huggingface.co/RemiFabre) and the Pollen Robotics team for the ideas and feedback that shaped the remote tools workflow.*
