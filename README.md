# Infer

Web server for Tensorflow inference in python.

```
$ python src/infer.py --help
usage: infer.py [-h] [--model MODEL] [--tags TAGS] [--batch_size BATCH_SIZE]
                [--static_path STATIC_PATH] [--batch_transpose] [--no_cors]
                [--request_size REQUEST_SIZE]

tf-infer

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         path to saved_model directory
  --tags TAGS           Comma separated SavedModel tags
  --batch_size BATCH_SIZE
                        Maximum batch size for batchable methods
  --static_path STATIC_PATH
                        Path to static content, eg. html files
  --batch_transpose     Provide and return each example in batches separately
  --no_cors             Turn off blanket CORS headers
  --request_size REQUEST_SIZE
                        Max size per request
```

## Usage

Tensorflow has a standard format for persisting models called SavedModel. Any model persistend in this format which specifies it signatures can then automatically be exposed as a web service with tf-infer.

Create a SavedModel that contains signature_defs (Look in the `examples` folder) then start a server exposing the model over JSON with `$ python infer.py --model path/to/savedmodel --batch_transpose`

To see what sort of APIs the model exposes one can query it directly:

`$ curl localhost:8080 | python -m json.tool`
```
[
    {
        "name": "features",
        "inputs": {
            "image": {
                "type": "string",
                "shape": [
                    -1
                ]
            }
        },
        "outputs": {
            "features": {
                "type": "float32",
                "shape": [
                    -1,
                    2048
                ]
            }
        }
    },
    {
        "name": "names",
        "inputs": {
            "image": {
                "type": "string",
                "shape": [
                    -1
                ]
            }
        },
        "outputs": {
            "names": {
                "type": "string",
                "shape": [
                    -1,
                    5
                ]
            }
        }
    }
]
```

Here we see the model has exposed two methods, `features` and `names` which accepts batches of strings. The model is in fact Inception v3 trained on OpenImages, meaning those batches of strings are batches of JPEG images. We cannot encode JPEG data as JSON, so we can either let the server fetch the data from a URL or we can base64 encode the image data before sending.

Thus we can query the method `names` like this:

`curl -d '{"image": {"url": "https://i.imgur.com/ekNNNjN.jpg"}}' localhost:8080/names | python -m json.tool`
```
{
    "names": [
        "mammal",
        "animal",
        "pet",
        "cat",
        "vertebrate"
    ]
}
```

And we received 5 strings corresponding to the best inception matches.

Since the method accepts batches we can send multiple queries in the same request:

`curl -d '[{"image": {"url": "https://i.imgur.com/ekNNNjN.jpg"}}, {"image": {"url": "https://i.imgur.com/JNo5tHj.jpg"}}]' localhost:8080/names | python -m json.tool`
```
[
    {
        "names": [
            "mammal",
            "animal",
            "pet",
            "cat",
            "vertebrate"
        ]
    },
    {
        "names": [
            "mammal",
            "animal",
            "pet",
            "vertebrate",
            "dog"
        ]
    }
]
```

In fact, for batched methods (identified by all inputs and outputs having a variable outer dimension, i.e. `-1`) tf-infer will batch up queries from multiple HTTP requests and process them as a single batch as soon as the backing tensorflow session is ready. Afterwards it will split the batch again and send the results to the correct recipients.

## Functionality

- Pure python - same as the tensorflow API!
- Reads tensorflow saved_model and exposes a HTTP API based on the signature definitions
- Batches across multiple requests for GPU utilization without delay
- Can read binary data over JSON either wrapped in `{"b64": "..."}` or `{"url": "..."}`
- Also base64 encodes JSON results that aren't valid UTF-8

## TODO
- tests
- drop requests based on queue length
- expose metrics for auto scaling
