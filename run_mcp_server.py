#!/usr/bin/env python3
"""
MCP Server Runner
=================

This script starts the MCP server for the Research Assistant.

How to Use:
-----------
Option 1 (Recommended): Run the research agent with --mcp flag
   $ python research_agent.py --mcp
   (This automatically spawns the MCP server as a subprocess)

Option 2: Run the server standalone for testing
   $ python run_mcp_server.py
   (Then connect a separate MCP client for debugging)

Note: The --mcp flag in research_agent.py automatically manages the server
lifecycle, so you typically don't need to run this script directly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import mcp

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           MCP Research Tools Server                          ║
╠══════════════════════════════════════════════════════════════╣
║  Protocol: Model Context Protocol (MCP)                      ║
║  SDK: Official Anthropic MCP SDK                             ║
║  Transport: stdio                                            ║
║                                                              ║
║  Available Tools:                                            ║
║  - web_search     : Search the web for information           ║
║  - get_facts      : Get curated facts about a topic          ║
║  - validate_claim : Validate if a claim seems reasonable     ║
║                                                              ║
║  Typical Usage:                                              ║
║    python research_agent.py --mcp                            ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""", file=sys.stderr)

    # Run the server using stdio transport
    mcp.run(transport="stdio")
