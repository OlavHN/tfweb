from distutils.core import setup
setup(name='tfweb',
      packages=['tfweb'],
      version='0.3',
      description='Server for exposing tensorflow models though HTTP JSON API',
      author='Olav Nymoen',
      author_email='olav@olavnymoen.com',
      url='https://github.com/olavhn/tfweb',
      download_url='https://github.com/olavhn/tfweb/archive/0.3.tar.gz',
      keywords=['serving', 'tensorflow', 'asyncio', 'aiohttp', 'grpc'],
      classifiers=[],
      scripts=['bin/tfweb'])
