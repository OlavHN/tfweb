import tensorflow as tf

from service_pb2 import PredictResponse
from service_grpc import ModelBase

# from predict_service_pb2 import PredictResponse
# from predict_service_grpc import PredictionServiceBase


class GrpcHandler(ModelBase):
    def __init__(self, model, batcher):
        self.model = model
        self.batcher = batcher

    async def Predict(self, stream):
        request = await stream.recv_message()
        method = request.model_spec.signature_name
        data = {
                key: tf.make_ndarray(val)
                for key, val in request.inputs.items()
        }

        if method in self.batcher.direct_methods:
            result = await self.single_query(method, data)
        else:
            result = await self.batch_query(method, data)

        if not result:
            await stream.send_message(PredictResponse())
            return

        result = {
                key: tf.make_tensor_proto(val)
                for key, val in result.items()
        }
        await stream.send_message(PredictResponse(result=result))

    async def single_query(self, method, data):
        try:
            query_params, result_params = await self.model.parse(
                    method, data, False)
            return await self.model.query(query_params, result_params)
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
            return None

        return result
