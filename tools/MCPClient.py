from authlib.oauth2.rfc7523 import client
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

client = MultiServerMCPClient({
    "sheldonMCPServer": {
        "transport": "stdio",
        "command": "python",
        "args": ["E:\PycharmProjects\demo\MCPServer.py"]
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