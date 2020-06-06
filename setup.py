from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("configs",)

# for install, do: pip install -ve .

setup(
    name='xmuda',
    version="0.0.1",
    url="https://github.com/maxjaritz/xmuda",
    description="xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation",
    install_requires=['yacs', 'nuscenes-devkit', 'tabulate'],
    packages=find_packages(exclude=exclude_dirs),
)