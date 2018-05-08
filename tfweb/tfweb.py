import sys
import argparse

import asyncio

from aiohttp import web
import aiohttp_cors

from model import Model
from batcher import Batcher

from json_handler import JsonHandler
from grpc_handler import GrpcHandler

from grpclib.server import Server


async def on_shutdown(app):
    for task in asyncio.Task.all_tasks():
        task.cancel()


async def init(loop, args):
    if args.tags:
        tags = args.tags.split(',')
    else:
        tags = [Model.default_tag]
    model = Model(args.model, tags, loop)
    batcher = Batcher(model, loop, args.batch_size)

    web_app = web.Application(loop=loop, client_max_size=args.request_size)
    web_app.on_shutdown.append(on_shutdown)
    web_app.router.add_get('/stats', batcher.stats_handler)

    json_handler = JsonHandler(model, batcher, args.batch_transpose)

    if args.no_cors:
        web_app.router.add_get('/', batcher.info_handler)
        web_app.router.add_post('/{method}', json_handler.handler)
    else:
        cors = aiohttp_cors.setup(
                web_app,
                defaults={
                        "*":
                        aiohttp_cors.ResourceOptions(
                                allow_credentials=True,
                                expose_headers="*",
                                allow_headers="*")
                })

        get_resource = cors.add(web_app.router.add_resource('/'))
        cors.add(get_resource.add_route("GET", batcher.info_handler))

        post_resource = cors.add(web_app.router.add_resource('/{method}'))
        cors.add(post_resource.add_route("POST", json_handler.handler))

    if args.static_path:
        web_app.router.add_static(
                '/web/', path=args.static_path, name='static')

    grpc_app = Server([GrpcHandler(model, batcher)], loop=loop)

    return web_app, grpc_app


def main(args):
    parser = argparse.ArgumentParser(description='tfweb')
    parser.add_argument(
            '--model',
            type=str,
            default='./examples/basic/model',
            help='path to saved_model directory (can be GCS)')
    parser.add_argument(
            '--tags',
            type=str,
            default=None,
            help='Comma separated SavedModel tags. Defaults to `serve`')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            help='Maximum batch size for batchable methods')
    parser.add_argument(
            '--static_path',
            type=str,
            default=None,
            help='Path to static content, eg. html files served on GET')
    parser.add_argument(
            '--batch_transpose',
            action='store_true',
            help='Provide and return each example in batches separately')
    parser.add_argument(
            '--no_cors',
            action='store_true',
            help='Accept HTTP requests from all domains')
    parser.add_argument(
            '--request_size',
            type=int,
            default=10 * 1024**2,
            help='Max size per request')
    parser.add_argument(
            '--grpc_port',
            type=int,
            default=50051,
            help='Port accepting grpc requests')
    args = parser.parse_args(args)

    loop = asyncio.get_event_loop()

    web_app, grpc_app = loop.run_until_complete(init(loop, args))

    loop.run_until_complete(grpc_app.start('0.0.0.0', args.grpc_port))

    try:
        web.run_app(web_app)
    except asyncio.CancelledError:
        pass


if __name__ == '__main__':
    main(args=sys.argv[1:])
