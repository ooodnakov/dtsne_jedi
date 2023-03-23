
from setuptools import setup, find_packages
 
setup(name = "dtsnejedi",
      author="Aleksandr Odnakov",
      author_email="me@dnakov.ooo",
      packages = find_packages(),
      install_requires = ['matplotlib','numpy','scipy','tqdm'],
      version='0.1.1')