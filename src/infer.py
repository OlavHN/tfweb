import argparse

import asyncio
from aiohttp import web
import aiohttp_cors

from model import Model
from batcher import Batcher

async def init(loop, args):
    if args.tags:
        tags = args.tags.split(',')
    else:
        tags = Model.default_tag
    model = Model(args.model, [tags])
    batcher = Batcher(model, loop, args.batch_size, args.batch_transpose)

    app = web.Application(loop=loop, client_max_size=args.request_size)
    app.router.add_get('/stats', batcher.stats_handler)

    if args.no_cors:
        app.router.add_get('/', batcher.info_handler)
        app.router.add_post('/{method}', batcher.handler)
    else:
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
        })

        get_resource = cors.add(app.router.add_resource('/'))
        cors.add(get_resource.add_route("GET", batcher.info_handler))

        post_resource = cors.add(app.router.add_resource('/{method}'))
        cors.add(post_resource.add_route("POST", batcher.handler))

    if args.static_path:
        app.router.add_static(
            '/web/',
            path=args.static_path,
            name='static')

    return app

def main():
    parser = argparse.ArgumentParser(description='tf-infer')
    parser.add_argument('--model', type=str, default='./model',
                        help='path to saved_model directory')
    parser.add_argument('--tags', type=str, default=None,
                    help='Comma separated SavedModel tags')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Maximum batch size for batchable methods')
    parser.add_argument('--static_path', type=str, default=None,
                        help='Path to static content, eg. html files')
    parser.add_argument('--batch_transpose', action='store_true',
                        help='Provide and return each example in batches separately')
    parser.add_argument('--no_cors', action='store_true',
                        help='Turn off blanket CORS headers')
    parser.add_argument('--request_size', type=int, default=10*1024**2,
                        help='Max size per request')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init(loop, args))
    web.run_app(app)

if __name__ == '__main__':
    main()
