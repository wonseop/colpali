# This script acts as a proxy between a WebSocket server and stdio.
# It is designed to be run inside a Docker container by Claude Desktop.
# The user specifies the remote server address via the REMOTE_URL environment
# variable in their local claude_desktop_config.json file.

import os
import sys
import asyncio
import websockets

async def main():
    """
    Connects to a WebSocket server specified by the REMOTE_URL environment variable
    and relays messages between the WebSocket and stdin/stdout.
    """
    remote_url = os.environ.get("REMOTE_URL")
    if not remote_url:
        print("Error: REMOTE_URL environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        async with websockets.connect(remote_url) as websocket:
            print(f"Proxy connected to {remote_url}", file=sys.stderr)

            # Create tasks for reading from stdin and WebSocket
            stdin_reader = asyncio.create_task(read_stdin(websocket))
            websocket_reader = asyncio.create_task(read_websocket(websocket))

            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [stdin_reader, websocket_reader],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks to ensure clean exit
            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"Error connecting or during proxy operation: {e}", file=sys.stderr)
        sys.exit(1)

async def read_stdin(websocket):
    """Reads from stdin and forwards to the WebSocket."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while not websocket.closed:
        line = await reader.readline()
        if not line:
            break
        await websocket.send(line.decode())

async def read_websocket(websocket):
    """Reads from the WebSocket and forwards to stdout."""
    async for message in websocket:
        print(message, flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Proxy shutting down.", file=sys.stderr)
