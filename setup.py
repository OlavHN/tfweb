from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

print(find_packages())

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='tfweb',
      version='0.4.4',
      description='Server for exposing tensorflow models though HTTP JSON API',
      long_description=long_description,
      url='https://github.com/olavhn/tfweb',
      author='Olav Nymoen',
      author_email='olav@olavnymoen.com',
      license='MIT',
      classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Topic :: Software Development :: Build Tools',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3.6',
      ],
      keywords=['serving', 'tensorflow', 'asyncio', 'aiohttp', 'grpc'],
      packages=["tfweb"],
      install_requires=['aiohttp>=2', 'aiohttp_cors>=0.7', 'grpclib>=0.1'],
      python_requires='>=3.5',
      scripts=['bin/tfweb'])

