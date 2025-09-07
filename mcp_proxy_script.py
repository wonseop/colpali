# This script acts as a proxy between a WebSocket server and stdio.
# It is designed to be run inside a Docker container by Claude Desktop.
# The user specifies the remote server address via the REMOTE_URL environment
# variable in their local claude_desktop_config.json file.

import os
import sys
import asyncio
import websockets

async def read_stdin(websocket):
    """Reads from stdin and forwards to the WebSocket."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line: # EOF
                break
            message = line.decode().strip()
            await websocket.send(message)
        except asyncio.CancelledError:
            break

async def read_websocket(websocket):
    """Reads from the WebSocket and forwards to stdout."""
    async for message in websocket:
        print(message, flush=True)

async def main():
    """
    Connects to a WebSocket server and relays messages.
    """
    remote_url = os.environ.get("REMOTE_URL")
    if not remote_url:
        print("Error: REMOTE_URL environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        async with websockets.connect(remote_url) as websocket:
            print(f"Proxy connected to {remote_url}", file=sys.stderr)
            
            stdin_task = asyncio.create_task(read_stdin(websocket))
            websocket_task = asyncio.create_task(read_websocket(websocket))

            done, pending = await asyncio.wait(
                [stdin_task, websocket_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Proxy shutting down.", file=sys.stderr)
