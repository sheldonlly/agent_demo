from fastmcp import FastMCP

mcp = FastMCP(name="SheldonMCPServer")

import numexpr

@mcp.tool
def get_weather(city: str) -> str:
    '''获取对应城市的天气情况'''
    return f"{city}天气晴朗，温度为25摄氏度"

@mcp.tool
def caculate(expression: str) -> str:
    '''计算表达式的值'''
    return str(numexpr.evaluate(expression))

if __name__ == "__main__":
    mcp.run(transport="stdio")