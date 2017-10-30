import asyncio
from aiohttp import ClientSession
import time
from grpclib.client import Channel

from service_pb2 import ModelQuery, ModelResult
from service_grpc import ModelStub

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

'''
TODO: Test grpc
channel = Channel(loop=loop)
stub = ModelStub(channel)

async def make_request():
    print('sending message through grpc')
    tensors = {
        'x1': tf.make_tensor_proto([[1], [3]], tf.float32, [2, 1]),
        'x2': tf.make_tensor_proto([[1], [2]], tf.float32, [2, 1])
    }
    response = await stub.Infer(ModelQuery(method='add', query=tensors))
    print('response', response.status, {k: tf.make_ndarray(v) for k,v in response.result.items()} )

loop.run_until_complete(make_request())
'''

future = asyncio.ensure_future(run(5000))
loop.run_until_complete(future)
