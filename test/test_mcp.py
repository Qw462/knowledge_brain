import asyncio
import httpx2
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

async def main():
    endpoint = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
    api_key = "sk-eda9f484ffc44cfdb926ea4e9c4e967e"
    auth_header = {"Authorization": f"Bearer {api_key}"}

    print(f"endpoint={endpoint}")
    print(f"header={auth_header}")

    # 自定义httpx2客户端，带上鉴权请求头
    async with httpx2.AsyncClient(headers=auth_header, timeout=12.0) as http_client:
        async with streamable_http_client(endpoint, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("✅握手成功！MCP connect完成")
                tools = await session.list_tools()
                print("可用工具列表：", [t.name for t in tools.tools])

if __name__ == "__main__":
    asyncio.run(main())
