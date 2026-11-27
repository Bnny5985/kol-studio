"""Entry point for the bundled MCP server (PyInstaller console executable `kolstudio-mcp`).
AI clients start it and talk to it over stdin/stdout."""
from kol_studio.mcp_server import main

if __name__ == "__main__":
    main()
