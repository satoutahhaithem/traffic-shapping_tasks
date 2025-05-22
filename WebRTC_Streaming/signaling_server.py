import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store for connected clients
connected = {}
offers = {}
answers = {}

async def signaling(websocket):
    client_id = None
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data.get("type") == "register":
                client_id = data["id"]
                connected[client_id] = websocket
                logger.info(f"Client {client_id} registered")
                
                # If this is a receiver and there's an offer waiting, send it
                if client_id.startswith("receiver") and client_id.replace("receiver", "sender") in offers:
                    sender_id = client_id.replace("receiver", "sender")
                    await websocket.send(json.dumps({
                        "type": "offer",
                        "sdp": offers[sender_id]
                    }))
                    logger.info(f"Sent stored offer to {client_id}")
                
                # If this is a sender and there's an answer waiting, send it
                if client_id.startswith("sender") and client_id.replace("sender", "receiver") in answers:
                    receiver_id = client_id.replace("sender", "receiver")
                    await websocket.send(json.dumps({
                        "type": "answer",
                        "sdp": answers[receiver_id]
                    }))
                    logger.info(f"Sent stored answer to {client_id}")
            
            elif data.get("type") == "offer":
                # Store the offer
                offers[client_id] = data["sdp"]
                logger.info(f"Received offer from {client_id}")
                
                # If the receiver is connected, send the offer
                receiver_id = client_id.replace("sender", "receiver")
                if receiver_id in connected:
                    await connected[receiver_id].send(json.dumps({
                        "type": "offer",
                        "sdp": data["sdp"]
                    }))
                    logger.info(f"Sent offer to {receiver_id}")
            
            elif data.get("type") == "answer":
                # Store the answer
                answers[client_id] = data["sdp"]
                logger.info(f"Received answer from {client_id}")
                
                # If the sender is connected, send the answer
                sender_id = client_id.replace("receiver", "sender")
                if sender_id in connected:
                    await connected[sender_id].send(json.dumps({
                        "type": "answer",
                        "sdp": data["sdp"]
                    }))
                    logger.info(f"Sent answer to {sender_id}")
            
            elif data.get("type") == "ice":
                # Forward ICE candidates to the other peer
                target_id = None
                if client_id.startswith("sender"):
                    target_id = client_id.replace("sender", "receiver")
                else:
                    target_id = client_id.replace("receiver", "sender")
                
                if target_id in connected:
                    await connected[target_id].send(json.dumps({
                        "type": "ice",
                        "candidate": data["candidate"]
                    }))
                    logger.info(f"Forwarded ICE candidate from {client_id} to {target_id}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    finally:
        if client_id and client_id in connected:
            del connected[client_id]
            logger.info(f"Client {client_id} disconnected")

async def main():
    # Start the signaling server on all interfaces
    server = await websockets.serve(signaling, "0.0.0.0", 8765)
    logger.info("Signaling server started on port 8765 (all interfaces)")
    
    # Keep the server running
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())