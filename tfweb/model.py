import tensorflow as tf
import numpy as np
import functools

dir(tf.contrib)  # contrib ops lazily loaded


class Model:

    default_tag = tf.saved_model.tag_constants.SERVING

    def __init__(self, path, tags, loop):
        self.sess = tf.Session()
        self.loop = loop
        try:
            self.graph_def = tf.saved_model.loader.load(self.sess, tags, path)
        except Exception as e:
            raise IOError('Couldn\'t load saved_model')

    async def parse(self, method, request, validate_batch):
        signature = tf.contrib.saved_model.get_signature_def_by_key(
                self.graph_def, method)

        inputs = signature.inputs
        outputs = signature.outputs

        query_params = {}
        batch_length = 0
        for key, value in inputs.items():
            if key not in request:
                raise ValueError(
                        'Request missing required key %s for method %s' %
                        (key, method))

            input_json = request[key]

            # input_json = list(map(base64.b64decode, input_json))
            dtype = tf.as_dtype(inputs[key].dtype).as_numpy_dtype
            try:
                tensor = np.asarray(input_json, dtype=dtype)
            except ValueError as e:
                raise ValueError('Incompatible types for key %s: %s' % (key,
                                                                        e))
            correct_shape = tf.TensorShape(inputs[key].tensor_shape)
            input_shape = tf.TensorShape(tensor.shape)
            if not correct_shape.is_compatible_with(input_shape):
                raise ValueError(
                        'Shape of input %s %s not compatible with %s' %
                        (key, input_shape.as_list(), correct_shape.as_list()))
            if validate_batch:
                try:
                    if batch_length > 0 and \
                       batch_length != input_shape.as_list()[0]:
                        raise ValueError(
                                'The outer dimension of tensors did not match')
                    batch_length = input_shape.as_list()[0]
                except IndexError:
                    raise ValueError(
                            '%s is a scalar and cannot be batched' % key)
            query_params[value.name] = tensor

        result_params = {
                key: self.sess.graph.get_tensor_by_name(val.name)
                for key, val in outputs.items()
        }

        return query_params, result_params

    async def query(self, query_params, result_params):
        ''' TODO: Interface via FIFO queue '''
        return await self.loop.run_in_executor(None,
                                               functools.partial(
                                                       self.sess.run,
                                                       result_params,
                                                       feed_dict=query_params))

    def list_signatures(self):
        signatures = []
        signature_def_map = self.graph_def.signature_def
        for key in signature_def_map.keys():
            signature_def = tf.contrib.saved_model.get_signature_def_by_key(
                    self.graph_def, key)

            signature = {}
            signature['name'] = key
            signature['inputs'] = {}
            signature['outputs'] = {}
            for key, tensor_info in signature_def.inputs.items():
                signature['inputs'][key] = {
                        'type':
                        tf.as_dtype(tensor_info.dtype).name,
                        'shape':
                        'unkown' if tensor_info.tensor_shape.unknown_rank else
                        [dim.size for dim in tensor_info.tensor_shape.dim]
                }
            for key, tensor_info in signature_def.outputs.items():
                signature['outputs'][key] = {
                        'type':
                        tf.as_dtype(tensor_info.dtype).name,
                        'shape':
                        'unkown' if tensor_info.tensor_shape.unknown_rank else
                        [dim.size for dim in tensor_info.tensor_shape.dim]
                }
            signatures.append(signature)
        return signatures
