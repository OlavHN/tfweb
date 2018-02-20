# tfweb

Web server for Tensorflow model inference in python.

## Quickstart

```
$ pip install tensorflow
$ pip install tfweb
$ tfweb --model s3://tfweb-models/hotdog --batch_transpose
$ curl -d '{"image": {"url": "https://i.imgur.com/H37kxPH.jpg"}}' localhost:8080/predict
{
  "class": ["no hotdog"],
  "prediction": [0.7314095497131348]
}
```

Might take some time to download the model from `s3://tfweb-models`.

```
$ tfweb -h

usage: tfweb [-h] [--model MODEL] [--tags TAGS] [--batch_size BATCH_SIZE]
                [--static_path STATIC_PATH] [--batch_transpose] [--no_cors]
                [--request_size REQUEST_SIZE] [--grpc_port GRPC_PORT]

tfweb

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         path to saved_model directory (can also be S3, GCS or hdfs)
  --tags TAGS           Comma separated SavedModel tags. Defaults to `serve`
  --batch_size BATCH_SIZE
                        Maximum batch size for batchable methods
  --static_path STATIC_PATH
                        Path to static content, eg. html files served on GET
  --batch_transpose     Provide and return each example in batches separately
  --no_cors             Accept HTTP requests from all domains
  --request_size REQUEST_SIZE
                        Max size per request
  --grpc_port GRPC_PORT
                        Port accepting grpc requests
```

## Why?

tfweb aims to be easier to setup, easier to tinker with and easier to integrate with than tf-serving. Thanks to being written in pure python 3 it's possible to interact with tensorflow though it's flexible python bindings.

## Usage

Tensorflow has a standard format for persisting models called SavedModel. Any model persisted in this format which specifies it signatures can then automatically be exposed as a web service with tfweb.

Create a SavedModel that contains signature_defs (Look in the `examples` folder) then start a server exposing the model over JSON with `$ tfweb --model s3://tfweb-models/openimages --batch_transpose`

To see what sort of APIs the model exposes one can query it to get its type information:

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

## Batching

By default tfweb doesn't do any batching, but if a method (signature definition) has a variable outer dimension for all inputs and outputs (i.e. shape is [-1, ..]) then the method is assumed to be batchable and tfweb will optimistically queue up requests for batching while the tensorflow session is busy doing other stuff (like running the previous batch).

If a method accepts batches we can also send multiple queries in the same request:

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

## Functionality

- Pure python - same as the most mature tensorflow API!
- Reads tensorflow saved_model and exposes a HTTP API based on type information in the signature definitions
- Batches across multiple requests for GPU utilization without delay
- Can read binary data over JSON either wrapped in `{"b64": "..."}` or `{"url": "..."}`
- Also base64 encodes JSON results that aren't valid UTF-8
- Also accepts the Predict gRPC signature. Check out `test.py` for an example.

## TODO
- More tests (both coded and real world!)
- Drop requests when
- expose metrics for auto scaling
- when downloading URLs, keep track of content size
