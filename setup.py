from setuptools import find_packages, setup


install_requires = []

setup(
    name='pimmatching',
    version='1.0.0',
    packages=find_packages(
        exclude=[]
    ),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'pimmatching = cli:main'
        ]
    }
)