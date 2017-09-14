import base64
import json
import urllib.request
import numpy as np
import asyncio
from aiohttp import web

class Batcher:
    def __init__(self, model, loop, batch_size=32, batch_transpose=False):
        self.loop = loop
        self.batch_size = batch_size
        self.model = model
        self.batch_transpose = batch_transpose

        batched_methods, direct_methods = self.find_batched_methods()
        self.direct_methods = set(map(lambda x: x['name'], direct_methods))

        self.batched_queues = {signature['name']: asyncio.Queue(maxsize=batch_size, loop=loop)
            for signature in batched_methods}

        for queue in self.batched_queues:
            loop.create_task(self.batch(self.batched_queues[queue], batch_size))

    def find_batched_methods(self):
        batched_methods = []
        direct_methods = []
        for signature in self.model.list_signatures():
            for _, tensor in list(signature['inputs'].items()) + list(signature['outputs'].items()):
                if isinstance(tensor['shape'], list) and len(tensor['shape']) and tensor['shape'][0] == -1:
                    continue
                else:
                    direct_methods.append(signature)
                    break
            else:
                batched_methods.append(signature)

        return batched_methods, direct_methods

    def decoder(self, data_str):
        def hook(d):
            if 'b64' in d:
                return base64.b64decode(d['b64'])
            if 'url' in d:
                return urllib.request.urlopen(d['url']).read()
            return d
        return json.loads(data_str, object_hook=hook)

    def encoder(self, data):
        class JSONBase64Encoder(json.JSONEncoder):
            ''' Base64 encodes strings that aren't valid UTF-8 '''
            def default(self, obj):
                def test_utf8(obj):
                    try:
                        return obj.decode('utf-8')
                    except UnicodeDecodeError:
                        return base64.b64encode(obj).decode('ascii')
                if isinstance(obj, (np.ndarray, np.generic)):
                    if obj.dtype == np.dtype(object):
                        obj = np.vectorize(test_utf8)(obj)
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        return json.dumps(data, ensure_ascii=False, cls=JSONBase64Encoder)

    async def handler(self, request):
        try:
            method = request.match_info.get('method', 'classify')
            if method not in self.batched_queues and method not in self.direct_methods:
                return web.json_response(
                    {'error': 'Method %s not found' % method}, status=400)

            data = await request.json(loads=self.decoder)

            if method in self.direct_methods:
                query_params, result_params = await self.model.parse(method, data, False)
                result = self.model.query(query_params, result_params)
                return web.json_response(result, dumps=self.encoder)

            unwrapped_example = False
            if self.batch_transpose:
                if type(data) == dict:
                    unwrapped_example = True
                    data = [data]
                # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
                data = dict(zip(data[0],zip(*[d.values() for d in data])))

            query_params, result_params = await self.model.parse(method, data, True)

            # Split in into examples and pass them through!
            task = asyncio.Queue(loop=self.loop)
            keys, values = zip(*sorted(query_params.items()))
            num_items = 0
            for example in zip(*values):
                num_items += 1
                await self.batched_queues[method].put((keys, example, result_params, task))

            # Get results back
            results = []
            for _ in range(num_items):
                result = await task.get()
                if result is None:
                    return web.json_response({'error': 'Batch failed'}, status=500)
                results.append(result)

            # Massage individual results into a batched response
            keys, examples = zip(*results)
            batch_result = { key: np.stack(val)
                    for key, val in zip(keys[0], zip(*examples)) }

            if self.batch_transpose:
                batch_result = [dict(zip(batch_result,t)) for t in zip(*batch_result.values())]
                if unwrapped_example:
                    batch_result = batch_result.pop()

            return web.json_response(batch_result, dumps=self.encoder)
        except ValueError as e:
            return web.json_response({'error': str(e)}, status=400)
        except TypeError as e:
            return web.json_response({'error': str(e)}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

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
            query_params = { key: np.stack(val)
                    for key, val in zip(keys[0], batched_examples) }

            try:
                result = self.model.query(query_params, result_params[0])

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
        signatures = self.model.list_signatures()
        return web.json_response(signatures)
