import numpy as np
import asyncio
from aiohttp import web


class Batcher:
    def __init__(self, model, loop, batch_size=32):
        self.loop = loop
        self.batch_size = batch_size
        self.model = model

        batched_methods, direct_methods = self.find_batched_methods()
        self.direct_methods = set(map(lambda x: x['name'], direct_methods))

        self.batched_queues = {
                signature['name']: asyncio.Queue(
                        maxsize=batch_size, loop=loop)
                for signature in batched_methods
        }

        for queue in self.batched_queues:
            loop.create_task(
                    self.batch(self.batched_queues[queue], batch_size))

    def find_batched_methods(self):
        batched_methods = []
        direct_methods = []
        for signature in self.model.list_signatures():
            for _, tensor in list(signature['inputs'].items()) + list(
                    signature['outputs'].items()):
                if isinstance(tensor['shape'], list) and len(
                        tensor['shape']) and tensor['shape'][0] == -1:
                    continue
                else:
                    direct_methods.append(signature)
                    break
            else:
                batched_methods.append(signature)

        return batched_methods, direct_methods

    async def batch_query(self, method, data):
        query_params, result_params = await self.model.parse(
                method, data, True)

        task = asyncio.Queue(loop=self.loop)

        # Split in into examples and pass them through!
        keys, values = zip(*sorted(query_params.items()))
        num_items = 0
        for example in zip(*values):
            num_items += 1
            await self.batched_queues[method].put((keys, example,
                                                   result_params, task))

        # Get results back
        results = []
        for _ in range(num_items):
            result = await task.get()
            if result is None:
                # Run them through individually instead!
                return None

            results.append(result)

        # Massage individual results into a batched response
        keys, examples = zip(*results)
        batch_result = {
                key: np.stack(val)
                for key, val in zip(keys[0], zip(*examples))
        }

        return batch_result

    async def batch(self, queue, max_batch_size):
        '''
        Greedily fills up batch before processing, but processes partial batch
        if the input queue is empty.
        '''
        batch = []
        while True:
            if queue.empty() and len(batch) == 0:
                batch.append(await queue.get())

            if not queue.empty() and len(batch) < max_batch_size:
                batch.append(await queue.get())
                continue

            keys, examples, result_params, queues = zip(*batch)
            batched_examples = zip(*examples)
            query_params = {
                    key: np.stack(val)
                    for key, val in zip(keys[0], batched_examples)
            }

            try:
                result = await self.model.query(query_params, result_params[0])

                keys, values = zip(*sorted(result.items()))

                for result_value, q in zip(zip(*values), queues):
                    q.put_nowait((keys, result_value))

            except Exception as e:
                print(e)
                for q in queues:
                    q.put_nowait(None)

            batch = []

    async def info_handler(self, request):
        signatures = self.model.list_signatures()
        return web.json_response(signatures)

    async def stats_handler(self, request):
        ''' TODO: Implement runtime stats '''
        signatures = self.model.list_signatures()
        return web.json_response(signatures)
