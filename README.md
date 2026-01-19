# LangGraph Research Assistant

A simple research assistant built with LangGraph that demonstrates core concepts for building stateful AI workflows.

**NEW:** Now with MCP (Model Context Protocol) integration for enhanced tool usage!

## What This Project Demonstrates

This project showcases the following concepts:

| Concept                | Implementation                                                      |
| ---------------------- | ------------------------------------------------------------------- |
| **State Management**   | `ResearchState` TypedDict defining shared state                     |
| **Node Functions**     | `planner_node`, `researcher_node`, `summarizer_node`                |
| **Graph Construction** | `StateGraph` with nodes and edges                                   |
| **Conditional Edges**  | `should_continue` function for dynamic routing                      |
| **Graph Compilation**  | `workflow.compile()` for execution                                  |
| **Visualization**      | `get_graph().draw_mermaid()` for diagram generation                 |
| **MCP Integration**    | Official MCP SDK (`mcp` package) for standardized tool access (NEW) |

---

## What is MCP (Model Context Protocol)?

MCP is a **standardized protocol** developed by Anthropic that enables LLMs to interact with external tools in a consistent, reusable way.

### Why MCP?

Think of MCP as a **"USB standard for AI tools"**:

| Problem                                   | MCP Solution                             |
| ----------------------------------------- | ---------------------------------------- |
| Every LLM app reimplements the same tools | Write once, use everywhere               |
| Tools are tightly coupled to applications | Tools are independent servers            |
| No standard for tool interfaces           | JSON-RPC based standard protocol         |
| Hard to share tools across projects       | Any MCP client works with any MCP server |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Research Agent                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LangGraph Workflow                    │    │
│  │   START → planner → researcher → summarizer → END       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              │ (when --mcp enabled)              │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     MCP Client                           │    │
│  │   • Discovers available tools                            │    │
│  │   • Calls tools via JSON-RPC                            │    │
│  │   • Returns results to workflow                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────│─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    stdio transport     │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                        MCP Server                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │ web_search  │  │ get_facts   │  │  validate_claim     │     │
│  │             │  │             │  │                     │     │
│  │ Search web  │  │ Get curated │  │ Check if claim is   │     │
│  │ for info    │  │ topic facts │  │ reasonable          │     │
│  └─────────────┘  └─────────────┘  └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
# Get your token at: https://huggingface.co/settings/tokens
```

---

## Running the Assistant

### Without MCP (Original Mode)

```bash
python research_agent.py
```

### With MCP (Enhanced Mode)

```bash
python research_agent.py --mcp
```

When running with `--mcp`, the agent will:

1. Spawn the MCP server as a subprocess
2. Connect and discover available tools
3. Use MCP tools during the research phase
4. Display which tools are being called

---

## MCP Tools Available

| Tool             | Description                     | Input        | Output                                        |
| ---------------- | ------------------------------- | ------------ | --------------------------------------------- |
| `web_search`     | Search the web for information  | `query: str` | Search results with titles, snippets, sources |
| `get_facts`      | Get curated facts about a topic | `topic: str` | List of verified facts                        |
| `validate_claim` | Check if a claim is reasonable  | `claim: str` | Assessment with confidence score              |

---

## Example Output

### Without MCP

```
============================================================
🔬 RESEARCH ASSISTANT
============================================================
Topic: artificial intelligence
🔌 MCP Mode: DISABLED (use --mcp flag to enable)
============================================================

📋 PLANNER: Breaking down topic into key questions...
   Generated 3 research questions:
   • 1. What is artificial intelligence?
   • 2. How does machine learning work?
   • 3. What are the applications of AI?

🔍 RESEARCHER: Investigating each question...
   Researching question 1/3...
   ✓ Question 1 answered
   ...
```

### With MCP

```
============================================================
🔬 RESEARCH ASSISTANT
============================================================
Topic: artificial intelligence
🔌 MCP Mode: ENABLED
🔌 Using MCP tools: web_search, get_facts, validate_claim
============================================================

📋 PLANNER: Breaking down topic into key questions...
   Generated 3 research questions:
   • 1. What is artificial intelligence?
   • 2. How does machine learning work?
   • 3. What are the applications of AI?

🔍 RESEARCHER: Investigating each question...
   Researching question 1/3...
      📡 MCP: Calling web_search...
      📡 MCP: Calling get_facts...
      📡 MCP: Calling validate_claim...
   🔍 MCP Validation: LIKELY_VALID (confidence: 0.75)
   ✓ Question 1 answered
   ...
```

---

## Project Structure

```
LangGraph-Research-Assistant/
├── research_agent.py     # Main agent with MCP integration
├── mcp_server.py         # MCP server with research tools
├── mcp_config.json       # MCP configuration
├── run_mcp_server.py     # Standalone server runner (optional)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

---

## Code Deep Dive

### Before MCP (Original researcher_node)

```python
def researcher_node(state: ResearchState) -> dict:
    questions = state["questions"]
    llm = get_llm()

    answers = []
    for question in questions:
        # Just LLM call
        response = llm.invoke([...prompt with question...])
        answers.append(response.content)

    return {"answers": answers}
```

### After MCP (Enhanced researcher_node)

```python
def researcher_node(state: ResearchState) -> dict:
    questions = state["questions"]
    use_mcp = state.get("use_mcp", False)
    llm = get_llm()

    answers = []
    for question in questions:
        # If MCP enabled, gather additional context first
        if use_mcp and mcp_client.connected:
            # Call web_search for relevant information
            search_results = await mcp_client.call_tool("web_search", {...})
            # Call get_facts for curated facts
            facts = await mcp_client.call_tool("get_facts", {...})
            # Include in prompt...

        # LLM call with enriched context
        response = llm.invoke([...prompt with MCP context...])

        # Validate the answer
        if use_mcp:
            validation = await mcp_client.call_tool("validate_claim", {...})

        answers.append(response.content)

    return {"answers": answers}
```

---

## Key MCP Concepts Demonstrated

### 1. Tool Discovery

```python
# Client asks server: "What tools do you have?"
request = {"method": "tools/list"}
response = await mcp_client.send(request)
# Returns: [{"name": "web_search", ...}, {"name": "get_facts", ...}]
```

### 2. Tool Invocation

```python
# Client calls a tool
request = {
    "method": "tools/call",
    "params": {
        "name": "web_search",
        "arguments": {"query": "artificial intelligence"}
    }
}
result = await mcp_client.send(request)
```

### 3. Server Registration (using FastMCP)

```python
from mcp.server.fastmcp import FastMCP

# Create server instance
mcp = FastMCP("research-tools")

# Register tools using decorators - FastMCP handles schema generation
@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information."""
    return json.dumps({"results": [...]})

@mcp.tool()
def get_facts(topic: str) -> str:
    """Get curated facts about a topic."""
    return json.dumps({"facts": [...]})

# Run the server
mcp.run(transport="stdio")
```

---

## Configuration

Edit `mcp_config.json` to customize MCP behavior:

```json
{
  "mcpServers": {
    "research-tools": {
      "command": "python",
      "args": ["mcp_server.py"],
      "tools": [
        { "name": "web_search", "enabled": true },
        { "name": "get_facts", "enabled": true },
        { "name": "validate_claim", "enabled": true }
      ]
    }
  }
}
```

---

## Key Takeaways

1. **MCP separates tools from applications** - Write tools once, use them with any MCP-compatible LLM app

2. **Standardized protocol** - JSON-RPC based communication means consistent interfaces

3. **Easy extensibility** - Add new tools to the MCP server without changing the main agent

4. **Tool reusability** - The same MCP server could power Claude, GPT, or any other LLM

5. **Clean architecture** - The agent doesn't need to know how tools are implemented, just how to call them

---

## Extending This Project

Ideas for enhancement:

- Add more MCP tools (database queries, API calls, file operations)
- Implement HTTP/SSE transport for remote MCP servers
- Add authentication to MCP server
- Create parallel research with multiple MCP servers
- Add caching layer for MCP responses

---
