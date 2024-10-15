# setup.py
from setuptools import setup, find_packages

setup(
    name='cs285',
    version='0.1.0',
    packages=find_packages(),
    package_data = {'': ['*.xml', '**/*.xml', '*.ttl', '**/*.ttl']}
)