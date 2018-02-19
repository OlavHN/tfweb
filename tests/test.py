import unittest
import tensorflow as tf
import numpy as np
import asyncio
from aiohttp import ClientSession

from infer.model import Model
from infer.batcher import Batcher

from grpclib.client import Channel
from infer.service_pb2 import PredictRequest
from infer.service_grpc import ModelStub


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_tensorflow(self):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.assertEqual(
                    sess.run(tf.constant(5) + tf.constant(5)), np.array(10))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(path='examples/basic/model', tags=['serve'])

    def tearDown(self):
        for task in asyncio.Task.all_tasks():
            task.cancel()

    def test_model_parse_valid(self):
        parsed = self.model.parse(
                method='add',
                request={'x1': [[1]],
                         'x2': [[1]]},
                validate_batch=True)

        query_params, result_params = (asyncio.get_event_loop()
                                       .run_until_complete(parsed))

        self.assertEqual(
                query_params, {
                        'x2:0': np.array([[1.]], dtype=np.float32),
                        'x1:0': np.array([[1.]], dtype=np.float32)
                })

        self.assertEqual(result_params['result'].name, 'add:0')
        self.assertEqual(result_params['result'].shape.as_list(), [None, 1])
        self.assertEqual(result_params['result'].dtype, tf.float32)

    def test_model_parse_invalid(self):
        parsed = self.model.parse(
                method='add',
                request={'x1': [[1]],
                         'x2': [[1], [2]]},
                validate_batch=True)

        self.assertRaises(ValueError,
                          asyncio.get_event_loop().run_until_complete, parsed)

    def test_model_query_session(self):
        parsed = self.model.parse(
                method='add',
                request={'x1': [[1]],
                         'x2': [[1]]},
                validate_batch=True)

        query_params, result_params = (asyncio.get_event_loop()
                                       .run_until_complete(parsed))

        result = self.model.query(query_params, result_params)

        self.assertEqual(result['result'], np.array([[2.]]))


class TestBatcher(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.model = Model(path='examples/basic/model', tags=['serve'])
        self.batcher = Batcher(self.model, self.loop)

    def tearDown(self):
        for task in asyncio.Task.all_tasks():
            task.cancel()

    def test_find_batched_methods(self):
        batched_methods, direct_methods = self.batcher.find_batched_methods()

        self.assertEqual(direct_methods, [])
        self.assertEqual(len(batched_methods), 2)

    def test_batch_query(self):
        resultTask = self.batcher.batch_query(
                method='add', data={
                        'x1': [[1], [3]],
                        'x2': [[1], [-9]]
                })

        result = asyncio.get_event_loop().run_until_complete(resultTask)

        self.assertTrue((np.array([[2.], [-6.]]) == result['result']).all())


@unittest.skip("Skipping integration tests. Requires spinning up a server")
class TestIntegration(unittest.TestCase):
    def test_10000_json(self):
        num_requests = 10000
        loop = asyncio.get_event_loop()

        async def fetch(url, data, session):
            async with session.post(url, json=data) as response:
                self.assertEqual(response.status, 200)
                return await response.read()

        async def bound_fetch(sem, url, data, session):
            # Getter function with semaphore.
            async with sem:
                return await fetch(url, data, session)

        async def run(r):
            url = "http://localhost:8080/{}"
            tasks = []
            sem = asyncio.Semaphore(1000)

            async with ClientSession() as session:
                for i in range(r):
                    data = {'x1': [[1], [2]], "x2": [[1], [2]]}
                    # pass Semaphore and session to every GET request
                    task = asyncio.ensure_future(
                            bound_fetch(sem, url.format("multiply"), data,
                                        session))
                    tasks.append(task)

                return await asyncio.gather(*tasks)

        future = asyncio.ensure_future(run(num_requests))

        print(loop.run_until_complete(future))
        print(future.result())
        self.assertTrue(True)

    def test_10000_grpc(self):
        num_requests = 10000
        loop = asyncio.get_event_loop()
        channel = Channel(host='127.0.0.1', port=50051, loop=loop)
        stub = ModelStub(channel)

        async def make_request():
            tensors = {
                    'x1': tf.make_tensor_proto([[1], [3]], tf.float32, [2, 1]),
                    'x2': tf.make_tensor_proto([[1], [2]], tf.float32, [2, 1])
            }

            request = PredictRequest(inputs=tensors)
            request.model_spec.signature_name = 'add'
            return await stub.Predict(request)

        async def run(n):
            tasks = []
            sem = asyncio.Semaphore(
                    100)  # grpc lib needs one socet per request
            for i in range(n):

                async def bound():
                    async with sem:
                        return await make_request()

                tasks.append(asyncio.ensure_future(bound()))

            return await asyncio.gather(*tasks)

        future = asyncio.ensure_future(run(num_requests))

        for result in loop.run_until_complete(future):
            print({k: tf.make_ndarray(v) for k, v in result.result.items()})
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
