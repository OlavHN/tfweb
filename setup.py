# from distutils.core import setup
from setuptools import setup

setup(name='tfweb',
      packages=['tfweb'],
      version='0.4',
      description='Server for exposing tensorflow models though HTTP JSON API',
      author='Olav Nymoen',
      author_email='olav@olavnymoen.com',
      url='https://github.com/olavhn/tfweb',
      download_url='https://github.com/olavhn/tfweb/archive/0.4.tar.gz',
      keywords=['serving', 'tensorflow', 'asyncio', 'aiohttp', 'grpc'],
      classifiers=[],
      install_requires=[
              'aiohttp',
              'aiohttp_cors',
              'grpclib==0.1.0rc2',
      ],
      python_requires='>=3.5',
      scripts=['bin/tfweb'])
