import base64
import json
import urllib.request

import numpy as np

from aiohttp import web
import asyncio


class JsonHandler(object):
    def __init__(self, model, batcher, batch_transpose=False):
        self.model = model
        self.batcher = batcher
        self.batch_transpose = batch_transpose

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

    def shutdown(self):
        for task in asyncio.Task.all_tasks():
            task.cancel()

    async def handler(self, request):
        try:
            method = request.match_info.get('method', 'serving_default')
            if method not in self.batcher.batched_queues and \
               method not in self.batcher.direct_methods:
                return web.json_response(
                        {
                                'error': 'Method %s not found' % method
                        }, status=400)

            data = await request.json(loads=self.decoder)

            if method in self.batcher.direct_methods:
                query_params, result_params = await self.model.parse(
                        method, data, False)
                result = await self.model.query(query_params, result_params)
                return web.json_response(result, dumps=self.encoder)

            unwrapped_example = False
            if self.batch_transpose:
                if type(data) == dict:
                    unwrapped_example = True
                    data = [data]
                # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
                data = dict(zip(data[0], zip(*[d.values() for d in data])))

            batch_result = await self.batcher.batch_query(method, data)

            if not batch_result:
                query_params, result_params = await self.model.parse(
                        method, data, False)
                batch_result = await self.model.query(query_params,
                                                      result_params)

            if not batch_result:
                return web.json_response(
                        {
                                'error': 'Batch request failed'
                        }, status=400)

            if self.batch_transpose:
                batch_result = [
                        dict(zip(batch_result, t))
                        for t in zip(*batch_result.values())
                ]
                if unwrapped_example:
                    batch_result = batch_result.pop()

            return web.json_response(batch_result, dumps=self.encoder)
        except ValueError as e:
            return web.json_response({'error': str(e)}, status=400)
        except TypeError as e:
            return web.json_response({'error': str(e)}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
