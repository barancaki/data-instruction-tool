# MCP/mcp_server/mcp_server.py
import asyncio

class MCPServer:
    def __init__(self, script_path=None):
        self.script_path = script_path  # opsiyonel, pipeline testinde kullanılabilir

    async def start(self):
        print("MCP Server çalışıyor...")
        await asyncio.sleep(1)  # dummy server için bekleme

    async def stop(self):
        print("MCP Server durduruldu")
