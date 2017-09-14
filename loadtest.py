import asyncio
from aiohttp import ClientSession
import time

'''
Simple load test to
'''

async def fetch(url, data, session):
    async with session.post(url, json=data) as response:
        return await response.read()

async def run(r):
    method1 = "http://localhost:8080/method1"
    data1 = {'input1': [[1],[2]], "input2": [[0,1,2],[3,4,5]]}
    method2 = "http://localhost:8080/method2"
    data2 = {'input1': [[1],[2]], "input2": [[0,1,2],[3,4,5]]}
    tasks = []

    async with ClientSession() as session:
        start = time.time()
        for i in range(r):
            task1 = asyncio.ensure_future(fetch(method1, data1, session))
            tasks.append(task1)
            task2 = asyncio.ensure_future(fetch(method2, data2, session))
            tasks.append(task2)

        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print('total time: %f' % elapsed)
        print('%f req / sec' % (len(tasks) / elapsed))

loop = asyncio.get_event_loop()
future = asyncio.ensure_future(run(5000))
loop.run_until_complete(future)
