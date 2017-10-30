import tensorflow as tf

from grpclib.server import Server

from service_pb2 import ModelQuery, ModelResult
from service_grpc import ModelBase

class GrpcHandler(ModelBase):
    def __init__(self, model, batcher):
        self.model = model
        self.batcher = batcher

    async def Infer(self, stream):
        request = await stream.recv_message()
        method = request.method
        data = {key: tf.make_ndarray(val) for key, val in request.query.items()}

        if method in self.batcher.direct_methods:
            result = await self.single_query(method, data)
        else:
            result = await self.batch_query(method, data)

        if not result:
            await stream.send_message(ModelResult(status="failed"))
            return

        result = {key: tf.make_tensor_proto(val) for key, val in result.items()}
        await stream.send_message(ModelResult(status="success", result=result))

    async def single_query(self, method, data):
        try:
            query_params, result_params = await self.model.parse(method, data, False)
            return self.model.query(query_params, result_params)
        except Exception as e:
            print(e)
            return None

    async def batch_query(self, method, data):
        try:
            result = await self.batcher.batch_query(method, data)
            if not result:
                result = await self.single_query(method, data)
        except Exception as e:
            print(e)
            retur None

        return result
