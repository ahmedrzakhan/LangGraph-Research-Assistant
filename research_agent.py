"""
LangGraph Research Assistant
============================
A simple research assistant that demonstrates LangGraph's core concepts:
- State management with TypedDict
- Node functions for processing
- Graph compilation and execution
- Conditional edges for workflow control

NEW: MCP (Model Context Protocol) Integration
---------------------------------------------
This agent now supports MCP, a standardized protocol for LLM-tool integration.
MCP enables tools to be:
- Reusable across different LLM applications
- Standardized with consistent interfaces
- Independently developed and maintained

When running with --mcp flag, the agent uses external MCP tools:
- web_search: Search the web for information
- get_facts: Get curated facts about a topic
- validate_claim: Validate if a claim seems reasonable

This agent takes a research topic and:
1. Plans key questions to research
2. Researches answers to each question (with MCP tools if enabled)
3. Summarizes findings into a final report
"""

import os
import sys
import json
import asyncio
import argparse
from typing import TypedDict, List, Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# LangChain imports for LLM interaction with Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

# MCP SDK imports - Official Anthropic Model Context Protocol client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# MCP CLIENT WRAPPER
# =============================================================================
# This wrapper class uses the official MCP SDK to connect to MCP servers.
# The SDK handles all protocol details (JSON-RPC, transport, etc.) automatically.

class MCPClientWrapper:
    """
    MCP Client using the official Anthropic MCP SDK.

    This client demonstrates the core MCP concepts:
    - Tool Discovery: List available tools from the server via session.list_tools()
    - Tool Invocation: Call tools via session.call_tool()
    - Lifecycle Management: Proper connection and cleanup with context managers

    The MCP SDK provides:
    - Automatic protocol handling (JSON-RPC 2.0)
    - Multiple transport options (stdio, HTTP, SSE)
    - Type-safe tool definitions
    - Built-in error handling
    """

    def __init__(self):
        """Initialize the MCP client wrapper."""
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.connected = False
        self.tools = []

    async def connect(self) -> bool:
        """
        Connect to the MCP server using stdio transport.

        The stdio transport spawns the server as a subprocess and communicates
        via stdin/stdout. This is ideal for local development and testing.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create an exit stack to manage the async context managers
            self.exit_stack = AsyncExitStack()

            # Define server parameters - how to spawn the MCP server
            server_params = StdioServerParameters(
                command=sys.executable,  # Python interpreter
                args=["mcp_server.py"],  # Server script
                cwd=os.path.dirname(os.path.abspath(__file__)),  # Working directory
            )

            # Connect to the server using stdio transport
            # stdio_client returns (read_stream, write_stream) for communication
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # Create a client session - this handles the MCP protocol
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the connection (MCP handshake)
            await self.session.initialize()

            # Discover available tools
            tools_result = await self.session.list_tools()
            self.tools = tools_result.tools

            self.connected = True
            return True

        except Exception as e:
            print(f"   Failed to connect to MCP server: {e}")
            if self.exit_stack:
                await self.exit_stack.aclose()
            return False

    async def call_tool(self, tool_name: str, arguments: dict) -> Optional[str]:
        """
        Call an MCP tool with the given arguments.

        Uses the official SDK's session.call_tool() method which handles:
        - Request formatting
        - Response parsing
        - Error handling

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool

        Returns:
            The tool's response as a string, or None if failed
        """
        if not self.connected or not self.session:
            return None

        try:
            # Call the tool using the SDK
            result = await self.session.call_tool(tool_name, arguments)

            # Extract text content from the response
            if result.content:
                for content in result.content:
                    if content.type == "text":
                        return content.text

            return None

        except Exception as e:
            print(f"   Error calling tool {tool_name}: {e}")
            return None

    async def disconnect(self):
        """Disconnect from the MCP server and cleanup resources."""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
        except Exception:
            # Ignore cleanup errors (can happen with piped input)
            pass
        self.connected = False
        self.session = None

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]


# Global MCP client instance (initialized when needed)
mcp_client: Optional[MCPClientWrapper] = None


# =============================================================================
# STATE DEFINITION
# =============================================================================
# TypedDict defines the structure of data that flows through the graph.
# Each node can read from and write to this shared state.

class ResearchState(TypedDict):
    """
    The state object that gets passed between nodes in our graph.

    Attributes:
        topic: The research topic provided by the user
        questions: List of research questions generated by the planner
        answers: List of answers to each research question
        final_report: The summarized research report
        current_step: Tracks which step we're on for progress display
        use_mcp: Whether to use MCP tools for research (NEW)
        mcp_results: Results from MCP tool calls for debugging/display (NEW)
    """
    topic: str
    questions: List[str]
    answers: List[str]
    final_report: str
    current_step: str
    use_mcp: bool
    mcp_results: List[str]


# =============================================================================
# LLM SETUP
# =============================================================================
# Initialize the Hugging Face model that will power our research assistant

def get_llm():
    """
    Create and return a ChatHuggingFace instance.
    Uses Qwen2.5-72B-Instruct via Hugging Face Inference API.
    You can change the model to any supported model on Hugging Face Hub.
    """
    # Create the Hugging Face endpoint
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",  # High-quality, well-supported model
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,  # Moderate creativity for research tasks
    )
    # Wrap it in ChatHuggingFace for chat-style interactions
    return ChatHuggingFace(llm=llm)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================
# Each node is a function that takes the current state and returns updates to it.
# Nodes represent discrete steps in our workflow.

def planner_node(state: ResearchState) -> dict:
    """
    PLANNER NODE
    ------------
    Takes the research topic and generates 3 key questions to investigate.

    This demonstrates:
    - Reading from state (topic)
    - Using LLM to generate structured output
    - Returning state updates (questions)
    """
    print(f"\n📋 PLANNER: Breaking down topic into key questions...")

    topic = state["topic"]
    llm = get_llm()

    # Create a prompt that asks the LLM to generate research questions
    messages = [
        SystemMessage(content="""You are a research planner. Given a topic,
        generate exactly 3 key questions that would help someone understand
        the topic comprehensively. Return ONLY the questions, one per line,
        numbered 1-3."""),
        HumanMessage(content=f"Topic: {topic}")
    ]

    # Call the LLM
    response = llm.invoke(messages)

    # Parse the response into a list of questions
    questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    # Display the generated questions
    print(f"   Generated {len(questions)} research questions:")
    for q in questions:
        print(f"   • {q}")

    # Return state updates - this gets merged with existing state
    return {
        "questions": questions,
        "current_step": "planning_complete"
    }


def researcher_node(state: ResearchState) -> dict:
    """
    RESEARCHER NODE
    ---------------
    Takes each question and generates a research answer.

    This demonstrates:
    - Iterating over state data (questions)
    - Making multiple LLM calls
    - Aggregating results into state (answers)

    NEW: MCP Integration
    --------------------
    When use_mcp is True, this node will:
    1. Call web_search to get information about each question
    2. Call get_facts to enrich the answer with factual data
    3. Call validate_claim to verify the generated answer
    4. Combine all information for a more comprehensive response
    """
    print(f"\n🔍 RESEARCHER: Investigating each question...")

    questions = state["questions"]
    topic = state["topic"]
    use_mcp = state.get("use_mcp", False)
    llm = get_llm()

    answers = []
    mcp_results = []

    for i, question in enumerate(questions, 1):
        print(f"   Researching question {i}/{len(questions)}...")

        # If MCP is enabled, use MCP tools to gather additional context
        mcp_context = ""
        if use_mcp and mcp_client and mcp_client.connected:
            mcp_context = asyncio.get_event_loop().run_until_complete(
                _gather_mcp_context(question, topic, mcp_results)
            )

        # Create a prompt for researching each question
        # If MCP provided context, include it in the prompt
        if mcp_context:
            system_content = f"""You are a research assistant investigating
            the topic: {topic}. You have access to the following research data
            gathered from various tools:

            {mcp_context}

            Using this information, provide a concise but informative answer
            (2-3 sentences) to the following question. Focus on key facts
            and insights. Integrate the tool-provided data into your response."""
        else:
            system_content = f"""You are a research assistant investigating
            the topic: {topic}. Provide a concise but informative answer
            (2-3 sentences) to the following question. Focus on key facts
            and insights."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question)
        ]

        response = llm.invoke(messages)
        answer = response.content.strip()

        # If MCP is enabled, validate the answer
        if use_mcp and mcp_client and mcp_client.connected:
            validation = asyncio.get_event_loop().run_until_complete(
                _validate_answer(answer, mcp_results)
            )
            if validation:
                print(f"   🔍 MCP Validation: {validation}")

        answers.append(answer)
        print(f"   ✓ Question {i} answered")

    return {
        "answers": answers,
        "mcp_results": mcp_results,
        "current_step": "research_complete"
    }


async def _gather_mcp_context(question: str, topic: str, mcp_results: List[str]) -> str:
    """
    Use MCP tools to gather context for answering a question.

    This function demonstrates how MCP tools can be composed together
    to provide richer context for the LLM.

    Args:
        question: The research question to investigate
        topic: The overall research topic
        mcp_results: List to append results to (for logging)

    Returns:
        A formatted string with context from MCP tools
    """
    context_parts = []

    # 1. Web Search - Get relevant search results
    print(f"      📡 MCP: Calling web_search...")
    search_result = await mcp_client.call_tool("web_search", {"query": f"{topic} {question}"})
    if search_result:
        mcp_results.append(f"web_search: {search_result[:200]}...")
        context_parts.append(f"Web Search Results:\n{search_result}")

    # 2. Get Facts - Get curated facts about the topic
    print(f"      📡 MCP: Calling get_facts...")
    facts_result = await mcp_client.call_tool("get_facts", {"topic": topic})
    if facts_result:
        mcp_results.append(f"get_facts: {facts_result[:200]}...")
        context_parts.append(f"Curated Facts:\n{facts_result}")

    return "\n\n".join(context_parts) if context_parts else ""


async def _validate_answer(answer: str, mcp_results: List[str]) -> Optional[str]:
    """
    Use MCP validate_claim tool to check the answer.

    Args:
        answer: The generated answer to validate
        mcp_results: List to append results to (for logging)

    Returns:
        Validation assessment string or None
    """
    print(f"      📡 MCP: Calling validate_claim...")
    validation_result = await mcp_client.call_tool("validate_claim", {"claim": answer})
    if validation_result:
        mcp_results.append(f"validate_claim: {validation_result[:200]}...")
        try:
            validation_data = json.loads(validation_result)
            return f"{validation_data.get('assessment', 'UNKNOWN')} (confidence: {validation_data.get('confidence', 'N/A')})"
        except json.JSONDecodeError:
            pass
    return None


def summarizer_node(state: ResearchState) -> dict:
    """
    SUMMARIZER NODE
    ---------------
    Combines all research findings into a cohesive final report.

    This demonstrates:
    - Reading multiple state fields (topic, questions, answers)
    - Synthesizing information
    - Producing final output (final_report)
    """
    print(f"\n📝 SUMMARIZER: Creating final research report...")

    topic = state["topic"]
    questions = state["questions"]
    answers = state["answers"]
    llm = get_llm()

    # Combine questions and answers for context
    qa_pairs = "\n\n".join([
        f"Q: {q}\nA: {a}"
        for q, a in zip(questions, answers)
    ])

    messages = [
        SystemMessage(content="""You are a research summarizer. Given a topic
        and Q&A pairs from research, create a well-structured summary report.
        Include:
        1. A brief introduction
        2. Key findings (bullet points)
        3. A conclusion
        Keep it concise but comprehensive."""),
        HumanMessage(content=f"Topic: {topic}\n\nResearch Findings:\n{qa_pairs}")
    ]

    response = llm.invoke(messages)

    print("   ✓ Report generated")

    return {
        "final_report": response.content.strip(),
        "current_step": "complete"
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTION
# =============================================================================
# Conditional edges determine which node to visit next based on state

def should_continue(state: ResearchState) -> str:
    """
    Determines the next step in the workflow based on current state.

    This demonstrates conditional routing in LangGraph.
    Returns the name of the next node to visit.
    """
    current_step = state.get("current_step", "")

    if current_step == "planning_complete":
        return "researcher"
    elif current_step == "research_complete":
        return "summarizer"
    else:
        return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
# Build the state graph that defines our workflow

def create_research_graph():
    """
    Creates and compiles the research assistant graph.

    Graph Structure:
    START -> planner -> researcher -> summarizer -> END

    Returns:
        CompiledGraph: The compiled graph ready for execution
    """
    # Initialize a StateGraph with our state schema
    workflow = StateGraph(ResearchState)

    # Add nodes to the graph
    # Each node is identified by a string name and associated with a function
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("summarizer", summarizer_node)

    # Add edges to define the flow
    # START -> planner: The entry point of our graph
    workflow.add_edge(START, "planner")

    # Add conditional edges from planner
    # This demonstrates dynamic routing based on state
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "researcher": "researcher",
            "summarizer": "summarizer",
            END: END
        }
    )

    # Add conditional edges from researcher
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "summarizer": "summarizer",
            END: END
        }
    )

    # summarizer -> END: Final node leads to completion
    workflow.add_edge("summarizer", END)

    # Compile the graph - this validates the structure and prepares for execution
    return workflow.compile()


# =============================================================================
# GRAPH VISUALIZATION
# =============================================================================

def visualize_graph(graph):
    """
    Generate a Mermaid diagram of the graph structure.
    Useful for documentation and understanding the workflow.
    """
    print("\n📊 Graph Structure (Mermaid Diagram):")
    print("-" * 40)
    print(graph.get_graph().draw_mermaid())
    print("-" * 40)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_research(topic: str, use_mcp: bool = False) -> str:
    """
    Execute the research workflow for a given topic.

    Args:
        topic: The research topic to investigate
        use_mcp: Whether to use MCP tools for enhanced research

    Returns:
        str: The final research report
    """
    # Create the graph
    graph = create_research_graph()

    # Initialize the starting state
    initial_state = {
        "topic": topic,
        "questions": [],
        "answers": [],
        "final_report": "",
        "current_step": "",
        "use_mcp": use_mcp,
        "mcp_results": []
    }

    print(f"\n{'='*60}")
    print(f"🔬 RESEARCH ASSISTANT")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    if use_mcp:
        print(f"🔌 MCP Mode: ENABLED")
        if mcp_client and mcp_client.connected:
            tools = mcp_client.get_tool_names()
            print(f"🔌 Using MCP tools: {', '.join(tools)}")
    else:
        print(f"🔌 MCP Mode: DISABLED (use --mcp flag to enable)")
    print(f"{'='*60}")

    # Execute the graph
    # The graph will process through each node, updating state along the way
    final_state = graph.invoke(initial_state)

    return final_state["final_report"]


async def initialize_mcp() -> bool:
    """
    Initialize the MCP client and connect to the server.

    Uses the official MCP SDK to establish a connection via stdio transport.
    The SDK handles the protocol details automatically.

    Returns:
        bool: True if MCP was successfully initialized
    """
    global mcp_client

    print("\n🔌 Initializing MCP connection...")
    mcp_client = MCPClientWrapper()

    if await mcp_client.connect():
        tools = mcp_client.get_tool_names()
        print(f"✓ Connected to MCP server (using official MCP SDK)")
        print(f"✓ Available tools: {', '.join(tools)}")
        return True
    else:
        print("✗ Failed to connect to MCP server")
        print("  Make sure mcp_server.py exists and MCP SDK is installed")
        return False


async def cleanup_mcp():
    """Cleanup MCP connection on exit."""
    global mcp_client
    if mcp_client:
        await mcp_client.disconnect()
        print("\n🔌 MCP connection closed")


def main():
    """
    CLI interface for the research assistant.

    Usage:
        python research_agent.py          # Run without MCP
        python research_agent.py --mcp    # Run with MCP tools enabled

    The --mcp flag enables MCP (Model Context Protocol) integration,
    which provides additional research tools:
    - web_search: Search for information
    - get_facts: Get curated facts
    - validate_claim: Validate claims
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LangGraph Research Assistant with optional MCP support"
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Enable MCP (Model Context Protocol) tools for enhanced research"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("       🔬 LangGraph Research Assistant 🔬")
    print("="*60)

    # Check for API key
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("\n❌ Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
        print("   Please copy .env.example to .env and add your Hugging Face token.")
        return

    # Initialize MCP if requested
    use_mcp = args.mcp
    if use_mcp:
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if not loop.run_until_complete(initialize_mcp()):
            print("\n⚠️  Continuing without MCP tools...")
            use_mcp = False

    # Get topic from user
    print("\nEnter a research topic (or 'quit' to exit):")

    try:
        while True:
            topic = input("\n📌 Topic: ").strip()

            if topic.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            if not topic:
                print("Please enter a valid topic.")
                continue

            try:
                # Run the research
                report = run_research(topic, use_mcp=use_mcp)

                # Display results
                print(f"\n{'='*60}")
                print("📄 FINAL RESEARCH REPORT")
                print(f"{'='*60}")
                print(report)
                print(f"{'='*60}")

                # Option to visualize graph
                show_graph = input("\nShow graph structure? (y/n): ").strip().lower()
                if show_graph == 'y':
                    graph = create_research_graph()
                    visualize_graph(graph)

            except Exception as e:
                print(f"\n❌ Error during research: {e}")
                print("   Please check your API key and try again.")

            print("\nEnter another topic or 'quit' to exit:")

    finally:
        # Cleanup MCP connection
        if use_mcp and mcp_client:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(cleanup_mcp())


if __name__ == "__main__":
    main()
