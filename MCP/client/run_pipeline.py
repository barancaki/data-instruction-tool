import sys
import os
import asyncio

from MCP.mcp_server.mcp_server import MCPServer
from mcp.client.stdio import stdio_client  # MCP client stdio helper

async def main():
    # MCPServer'i başlat
    server = MCPServer()

    # MCP server ile stdio üzerinden iletişim
    async with stdio_client(server) as streams:
        # Burada test kodlarını çalıştırabilirsin
        print("MCP client-server bağlantısı başarılı!")

if __name__ == "__main__":
    asyncio.run(main())
