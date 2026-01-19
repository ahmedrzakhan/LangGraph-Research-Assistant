"""
MCP Server for Research Assistant
=================================

This module implements an MCP (Model Context Protocol) server using the official
Anthropic MCP SDK. It exposes research-related tools that can be consumed by
any MCP-compatible client.

What is MCP?
------------
MCP (Model Context Protocol) is a standardized protocol developed by Anthropic
that enables LLMs to interact with external tools in a consistent way. Think of
it as a "USB standard" for AI tools - any MCP-compatible tool works with any
MCP-compatible LLM application.

This server uses FastMCP, the official high-level API for building MCP servers.
FastMCP makes it easy to:
- Define tools using simple decorators (@mcp.tool())
- Handle type validation automatically via Pydantic
- Support multiple transport mechanisms (stdio, HTTP, SSE)

Benefits of MCP:
- Tool Reusability: Write once, use with any LLM (Claude, GPT, etc.)
- Standardized Interface: Consistent way to define and call tools
- Separation of Concerns: Tools are independent from the LLM application
- Easy Extensibility: Add new tools without modifying the main application

This server exposes three research tools:
1. web_search: Simulates searching the web for information
2. get_facts: Returns curated facts about a topic
3. validate_claim: Checks if a claim seems reasonable
"""

import json
import random
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# =============================================================================
# MCP SERVER SETUP
# =============================================================================
# FastMCP is the high-level API for building MCP servers.
# It handles all the protocol details automatically.

# Create the FastMCP server instance
# FastMCP is the high-level API that handles all MCP protocol details
mcp = FastMCP("research-tools")


# =============================================================================
# SIMULATED DATA SOURCES
# =============================================================================
# In a real application, these would connect to actual APIs or databases.
# For educational purposes, we use simulated data to demonstrate MCP concepts.

SIMULATED_SEARCH_RESULTS = [
    {"title": "Overview and Introduction", "snippet": "A comprehensive look at the topic covering key aspects and recent developments.", "source": "Wikipedia"},
    {"title": "Expert Analysis", "snippet": "Industry experts weigh in on the latest trends and best practices.", "source": "TechReview"},
    {"title": "Research Study", "snippet": "Academic research provides evidence-based insights into the subject.", "source": "Journal of Science"},
]

TOPIC_FACTS = {
    "artificial intelligence": [
        "AI systems can now pass professional exams like the bar exam and medical licensing tests.",
        "The global AI market is projected to reach $1.8 trillion by 2030.",
        "Machine learning, a subset of AI, requires large datasets for training.",
        "Neural networks are inspired by the structure of the human brain.",
    ],
    "climate change": [
        "Global average temperature has risen by about 1.1C since pre-industrial times.",
        "The Paris Agreement aims to limit warming to 1.5C above pre-industrial levels.",
        "Renewable energy sources now account for over 30% of global electricity generation.",
        "Arctic sea ice has declined by about 13% per decade since 1979.",
    ],
    "default": [
        "This topic has been extensively studied by researchers worldwide.",
        "Recent developments have led to significant advancements in this field.",
        "Experts recommend staying updated with peer-reviewed sources.",
        "The subject continues to evolve with new discoveries being made regularly.",
    ]
}

VALIDATION_KEYWORDS = {
    "reasonable": ["research shows", "studies indicate", "experts say", "data suggests", "according to"],
    "questionable": ["always", "never", "100%", "guaranteed", "miracle", "instantly"],
}


# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================
# Tools are defined using the @mcp.tool() decorator.
# FastMCP automatically:
# - Generates JSON Schema from type hints
# - Validates input parameters
# - Handles serialization/deserialization


@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the web for information about a query.

    Returns relevant search results with titles, snippets, and sources.
    In a real implementation, this would connect to a search API like
    Google, Bing, or DuckDuckGo.

    Args:
        query: The search query to look up

    Returns:
        JSON string containing search results
    """
    # Simulate search results customized to the query
    results = []
    for result in SIMULATED_SEARCH_RESULTS:
        results.append({
            "title": f"{result['title']} - {query[:30]}",
            "snippet": result["snippet"],
            "source": result["source"],
            "relevance_score": round(random.uniform(0.7, 0.99), 2)
        })

    response = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "result_count": len(results),
        "results": results
    }

    return json.dumps(response, indent=2)


@mcp.tool()
def get_facts(topic: str) -> str:
    """
    Get curated facts about a specific topic.

    Returns verified factual information useful for research.
    In a real implementation, this would query a knowledge base
    or facts API with citations.

    Args:
        topic: The topic to get facts about

    Returns:
        JSON string containing curated facts
    """
    # Look up facts for the topic (case-insensitive)
    topic_lower = topic.lower().strip()
    facts = TOPIC_FACTS.get(topic_lower, TOPIC_FACTS["default"])

    # Select a random subset to simulate variety
    selected_facts = random.sample(facts, min(3, len(facts)))

    response = {
        "topic": topic,
        "fact_count": len(selected_facts),
        "facts": selected_facts,
        "source": "Curated Knowledge Base",
        "last_updated": datetime.now().isoformat()
    }

    return json.dumps(response, indent=2)


@mcp.tool()
def validate_claim(claim: str) -> str:
    """
    Validate whether a claim seems reasonable based on common patterns.

    Returns a validation assessment with confidence level.
    This is a simplified validation that checks for:
    - Presence of source citations (good indicator)
    - Absolute language (potential red flag)
    - Claim structure and length

    In a real implementation, this would cross-reference with
    fact-checking databases and use ML models.

    Args:
        claim: The claim text to validate

    Returns:
        JSON string containing validation results
    """
    claim_lower = claim.lower()

    # Count indicators
    reasonable_count = sum(1 for kw in VALIDATION_KEYWORDS["reasonable"] if kw in claim_lower)
    questionable_count = sum(1 for kw in VALIDATION_KEYWORDS["questionable"] if kw in claim_lower)

    # Calculate confidence score
    base_score = 0.6  # Start with neutral-positive
    base_score += reasonable_count * 0.1
    base_score -= questionable_count * 0.15
    confidence = max(0.1, min(0.95, base_score))

    # Determine assessment
    if confidence >= 0.7:
        assessment = "LIKELY_VALID"
        explanation = "The claim appears to be well-structured and uses language consistent with factual statements."
    elif confidence >= 0.4:
        assessment = "UNCERTAIN"
        explanation = "The claim requires additional verification. Consider checking primary sources."
    else:
        assessment = "QUESTIONABLE"
        explanation = "The claim contains language patterns often associated with unverified information."

    response = {
        "claim": claim[:100] + ("..." if len(claim) > 100 else ""),
        "assessment": assessment,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "recommendation": "Always verify claims with multiple reliable sources."
    }

    return json.dumps(response, indent=2)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # Print info to stderr (doesn't interfere with MCP protocol on stdout)
    print("Starting MCP Research Tools Server...", file=sys.stderr)
    print("Tools available: web_search, get_facts, validate_claim", file=sys.stderr)

    # Run the server using stdio transport (default for local development)
    # FastMCP handles all the MCP protocol details automatically
    mcp.run(transport="stdio")
