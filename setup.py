from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='ddplib',
    packages=find_packages(include=['ddplib']),
    version='0.1.0',
    description='Library for disturbance decoupling problem',
    long_description=Path(__file__).with_name('README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Deepak Patil',
    license='GPL-3.0',
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy', 'control'],
    extras_require={'test': ['pytest']},
)
