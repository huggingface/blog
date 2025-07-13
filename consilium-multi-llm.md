---
title: "Consilium: When Multiple LLMs Collaborate" 
thumbnail: /blog/assets/consilium-multi-llm/thumbnail.png
authors:
- user: azettl
---

# Consilium: When Multiple LLMs Collaborate

Picture this: four AI experts sitting around a poker table, debating your toughest decisions in real-time. That's exactly what Consilium, the multi-LLM platform I built during the Gradio Agents & MCP Hackathon, does. It lets AI models discuss complex questions and reach consensus through structured debate.

The platform works both as a visual Gradio interface and as an MCP (Model Context Protocol) server that integrates directly with applications like Cline (Claude Desktop had issues as the timeout could not be adjusted). The core idea was always about LLMs reaching consensus through discussion; that's where the name Consilium came from. Later I added other decision modes like majority voting and ranked choice to make the collaboration more sophisticated.

## From Concept to Architecture

This wasn't my original hackathon idea. I initially wanted to build a simple MCP server to talk to my projects in RevenueCat. But I reconsidered when I realized a multi-LLM platform where these models discuss questions and return well-reasoned answers would be far more compelling.

The timing turned out to be perfect. Shortly after the hackathon, Microsoft published their AI Diagnostic Orchestrator (MAI-DxO), which is essentially an AI doctor panel with different roles like "Dr. Challenger Agent" that iteratively diagnose patients. In their setup with OpenAI o3, they correctly solved 85.5% of medical diagnosis benchmark cases, while practicing physicians achieved only 20% accuracy. This validates exactly what Consilium demonstrates: multiple AI perspectives collaborating can dramatically outperform individual analysis.

After settling on the concept, I needed something that worked as both an MCP server and an engaging Hugging Face space demo. Initially I considered using the standard Gradio Chat component, but I wanted my submission to stand out. The idea was to seat LLMs around a table in a boardroom with speech bubbles, which should capture the collaborative discussion while also making it visually engaging. As I did not manage to style a standard table nicely so it was actually recognized as a table, I went for a poker-style roundtable. This approach also meant I could submit to two hackathon tracks by building a custom Gradio component and MCP server.

## Building the Visual Foundation

The custom Gradio component became the heart of the submission; the poker-style roundtable where participants sit and display speech bubbles showing their responses, thinking status, and research activities immediately caught the eye of anyone visiting the space. The component development was remarkably smooth thanks to Gradio's excellent developer experience, though I did encounter one documentation gap around PyPI publishing that led to my first contribution to the Gradio project.

```python
# The visual component integration
roundtable = consilium_roundtable(
    label="AI Expert Roundtable",
    value=json.dumps({
        "participants": [],
        "messages": [],
        "currentSpeaker": None,
        "thinking": [],
        "showBubbles": [],
        "avatarImages": avatar_images
    })
)
```

The visual design proved robust throughout the hackathon; after the initial implementation, I only needed to add features like user-defined avatars and center table text, while the core interaction model remained unchanged.

## Making LLMs Actually Discuss

While implementing, I realized there was no real discussion happening between the LLMs because they lacked clear roles. They received the full context of ongoing discussions but didn't know how to engage meaningfully. I introduced distinct roles to create productive debate dynamics, which, after a few tweaks, ended up being like this:

```python
self.roles = {
    'expert_advocate': "You are a PASSIONATE EXPERT advocating for your specialized position. Present compelling evidence with conviction.",
    'critical_analyst': "You are a RIGOROUS CRITIC. Identify flaws, risks, and weaknesses in arguments with analytical precision.",
    'strategic_advisor': "You are a STRATEGIC ADVISOR. Focus on practical implementation, real-world constraints, and actionable insights.",
    'research_specialist': "You are a RESEARCH EXPERT with deep domain knowledge. Provide authoritative analysis and evidence-based insights."
}
```

This solved the discussion problem but raised a new question: how to determine consensus or identify the strongest argument? I implemented a lead analyst system where users select one LLM to synthesize the final result and evaluate whether consensus was reached.

I also wanted users to control communication structure. Beyond the default full-context sharing, I added two alternative modes:

* **Ring**: Each LLM only receives the previous participant's response  
* **Star**: All messages flow through the lead analyst as a central coordinator

Finally, discussions need endpoints. I implemented configurable rounds (1-5), with testing showing that more rounds increase the likelihood of reaching consensus (though at higher computational cost).

## LLM Selection and Research Integration

The current model selection includes Mistral Large, DeepSeek-R1, Meta-Llama-3.3-70B, and QwQ-32B. While notable models like Claude Sonnet and OpenAI's o3 are absent, this reflected hackathon credit availability and sponsor award considerations rather than technical limitations.

```python
self.models = {
    'mistral': {
        'name': 'Mistral Large',
        'api_key': mistral_key,
        'available': bool(mistral_key)
    },
    'sambanova_deepseek': {
        'name': 'DeepSeek-R1',
        'api_key': sambanova_key,
        'available': bool(sambanova_key)
    }
}
```

For models supporting function calling, I integrated a dedicated research agent that appears as another roundtable participant. Rather than giving models direct web access, this agent approach provides visual clarity about external resource availability and ensures consistent access across all function-calling models.

```python
def handle_function_calls(self, completion, original_prompt: str, calling_model: str) -> str:
    """UNIFIED function call handler with enhanced research capabilities"""
    
    message = completion.choices[0].message
    
    # If no function calls, return regular response
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        return message.content
    
    # Process each function call
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute research and show progress
        result = self._execute_research_function(function_name, arguments, calling_model_name)
```

The research agent accesses five sources: Web Search, Wikipedia, arXiv, GitHub, and SEC EDGAR. I built these tools on an extensible base class architecture for future expansion while focusing on freely embeddable resources.

```python
class BaseTool(ABC):
    """Base class for all research tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.rate_limit_delay = 1.0  # Be respectful to APIs
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> str:
        """Main search method - implemented by subclasses"""
        pass
    
    def score_research_quality(self, research_result: str, source: str = "web") -> Dict[str, float]:
        """Score research based on recency, authority, specificity, relevance"""
        quality_score = {
            "recency": self._check_recency(research_result),
            "authority": self._check_authority(research_result, source),
            "specificity": self._check_specificity(research_result),
            "relevance": self._check_relevance(research_result)
        }
        return quality_score
```

Since research operations can be time-intensive, the speech bubbles display progress indicators and time estimates to maintain user engagement during longer research tasks.

## Discovering the Open Floor Protocol

After the hackathon, Deborah Dahl introduced me to the Open Floor Protocol, which aligns perfectly with the roundtable approach. This protocol provides standardized JSON message formatting for cross-platform agent communication. Its key differentiator from other agent-to-agent protocols is that all agents maintain constant conversation awareness exactly like sitting at the same table.

The protocol's interaction patterns map directly to Consilium's architecture:

* **Delegation**: Transferring control between agents  
* **Channeling**: Passing messages without modification  
* **Mediation**: Coordinating behind the scenes  
* **Orchestration**: Multiple agents collaborating

I'm currently integrating Open Floor Protocol support to allow users to add any OFP-compliant agents to their roundtable discussions. You can follow this development at: https://huggingface.co/spaces/azettl/consilium_ofp

## Lessons Learned and Future Implications

The hackathon introduced me to multi-agent debate research I hadn't previously encountered, including foundational studies like https://arxiv.org/abs/2305.19118. The community experience was remarkable; all participants actively supported each other through Discord feedback and collaboration. Seeing my roundtable component integrated into another project (Agents-MCP-Hackathon/multi-agent-chat) was one of my highlights working on Consilium.

I will continue to work on Consilium and with expanded model selection, Open Floor Protocol integration, and configurable agent roles, the platform could support virtually any multi-agent debate scenario imaginable.

Building Consilium reinforced my conviction that AI's future lies not just in more powerful individual models, but in systems enabling effective AI collaboration. As specialized smaller language models become more efficient and resource-friendly, I believe roundtables of task-specific SLMs with dedicated research agents may provide compelling alternatives to general-purpose large language models for many use cases.