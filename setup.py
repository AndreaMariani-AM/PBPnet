from setuptools import setup

setup(
    name='PBPnet',
    version='0.0.1',
    author='Andrea Mariani',
    author_email='andrea.mariani1@icloud.com',
    packages=['bpnetlite'],
    scripts=['bpnet', 'chrombpnet'],
    url='https://github.com/AndreaMariani-AM/PBPnet',
    license='LICENSE.txt',
    description='PBPnet is a modification of the original bpnet-lite.',
    python_requires='>=3.9',
    install_requires=[
        "bpnet-lite >= 0.8.1"
    ],
)
