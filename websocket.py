import asyncio, websockets

async def main():
    async with websockets.connect("ws://localhost:18080/ws/video") as ws:
        data = open("teste.h264", "rb").read()
        CH = 4096  # o parser do servidor junta e separa os frames sozinho
        for i in range(0, len(data), CH):
            await ws.send(data[i:i+CH])
            await asyncio.sleep(0.002)
        await asyncio.sleep(2)

asyncio.run(main())