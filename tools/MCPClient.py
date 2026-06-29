import asyncio
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import log.logconfig  # noqa: F401
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "sheldonMCPServer": {
        "transport": "stdio",
        "command": "python",
        "args": ["E:\PycharmProjects\demo\tools\MCPServer.py"]
    },
    "gaodeMCPServer": {
        "transport": "http",
        "url": "https://mcp.amap.com/mcp?key=a859140b7e5cd85aaff02af8d9643672"
    }
})

async def main():
    tools = await client.get_tools()
    print(f"一共有{len(tools)}个工具，分别是：")
    k = 1
    for tool in tools:
        print(f"{k}、{tool.name}: {tool.description}")
        k += 1
    print(type(tools))
    print(tools)

if __name__ == "__main__":
    asyncio.run(main())