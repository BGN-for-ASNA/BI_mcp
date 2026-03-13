import asyncio
import os
import sys

# Add the parent directory of BI_mcp to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from BI_mcp.server import app, list_tools, list_resources


async def check():
    print("--- Tools ---")
    tools = await list_tools()
    for t in tools:
        print(f"Tool: {t.name}")

    print("\n--- Resources ---")
    res = await list_resources()
    for r in res:
        print(f"Resource: {r.uri}")


if __name__ == "__main__":
    asyncio.run(check())
