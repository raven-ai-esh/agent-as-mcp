"""CLI entry point for raven-skills.

Usage:
    # Quick start with defaults
    python -m raven_skills serve
    
    # With custom settings
    python -m raven_skills serve --port 8000 --model gpt-4o
    
    # With custom app (FastAPI-style)
    python -m raven_skills serve my_app:mcp
"""

import argparse
import sys
import os

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def main():
    parser = argparse.ArgumentParser(
        prog="raven-skills",
        description="Skill-based AI agents library",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run as MCP server",
    )
    
    # Positional argument for custom app (optional)
    serve_parser.add_argument(
        "app",
        nargs="?",
        default=None,
        help="Custom app module:variable (e.g., my_app:mcp). If not provided, uses default server.",
    )
    
    # Server settings
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    serve_parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="Transport type (default: http)",
    )
    
    # Storage settings
    serve_parser.add_argument(
        "--storage",
        type=str,
        default="./skills.json",
        help="Path to skills JSON file (default: ./skills.json)",
    )
    
    # LLM settings
    serve_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)",
    )
    
    # API settings
    serve_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (overrides OPENAI_API_KEY env var)",
    )
    serve_parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=None,
        help="Separate base URL for embeddings (if different from LLM)",
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_server(args)
    else:
        parser.print_help()


def run_server(args):
    """Run the MCP server."""
    
    # If custom app provided, load it
    if args.app:
        run_custom_app(args)
        return
    
    # Otherwise use default server
    run_default_server(args)


def run_custom_app(args):
    """Run a custom MCP app (FastAPI-style)."""
    try:
        module_path, var_name = args.app.split(":")
    except ValueError:
        print(f"Error: Invalid app format '{args.app}'")
        print("Expected format: module:variable (e.g., my_app:mcp)")
        sys.exit(1)
    
    # Add current directory to path for imports
    sys.path.insert(0, os.getcwd())
    
    try:
        import importlib
        module = importlib.import_module(module_path)
        mcp = getattr(module, var_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading app: {e}")
        sys.exit(1)
    
    print(f"Starting custom MCP server from {args.app}...")
    _run_mcp(mcp, args)


def run_default_server(args):
    """Run the default MCP server with CLI settings."""
    try:
        from raven_skills.mcp_server import create_mcp_server
    except ImportError as e:
        print(f"Error: {e}")
        print("Install MCP SDK with: pip install mcp")
        sys.exit(1)
    
    print(f"Starting raven-skills MCP server...")
    print(f"  Storage: {args.storage}")
    print(f"  LLM Model: {args.model}")
    print(f"  Embedding Model: {args.embedding_model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")
    print(f"  Transport: {args.transport}")
    
    mcp = create_mcp_server(
        storage_path=args.storage,
        llm_model=args.model,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_base_url=args.embedding_base_url,
    )
    
    _run_mcp(mcp, args)


def _run_mcp(mcp, args):
    """Run the MCP server with the specified transport."""
    if args.transport == "stdio":
        print("  Mode: stdio (for Claude Desktop)")
        mcp.run(transport="stdio")
    else:
        print(f"  URL: http://localhost:{args.port}/mcp")
        import uvicorn
        app = mcp.streamable_http_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
