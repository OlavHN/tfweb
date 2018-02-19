from distutils.core import setup
setup(name='infer',
      packages=['infer'],
      version='0.1',
      description='Server for exposing tensorflow models though HTTP JSON API',
      author='Olav Nymoen',
      author_email='olav@olavnymoen.com',
      url='https://github.com/olavhn/tf-infer',
      download_url='https://github.com/olavhn/tf-infer/archive/0.2.tar.gz',
      keywords=['serving', 'tensorflow', 'asyncio', 'aiohttp', 'grpc'],
      classifiers=[],
      scripts=['bin/infer'])
