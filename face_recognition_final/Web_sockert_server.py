import asyncio
import websockets

# List to keep track of connected clients
connected_clients = set()

# WebSocket handler to send notifications
async def notify_clients(recognized_name, authorized):
    message = {
        'name': recognized_name,
        'authorized': authorized
    }
    if connected_clients:
        await asyncio.wait([client.send(str(message)) for client in connected_clients])

# WebSocket server to handle connections from frontend
async def websocket_server(websocket, path):
    # Register client
    connected_clients.add(websocket)
    try:
        # Wait for client to disconnect
        await websocket.wait_closed()
    except:
        pass
    finally:
        # Unregister client when it disconnects
        connected_clients.remove(websocket)

# Start the WebSocket server
async def start_server():
    server = await websockets.serve(websocket_server, 'localhost', 8765)
    await server.wait_closed()

# Start the WebSocket server in the background
asyncio.get_event_loop().run_until_complete(start_server())
