import io
import os
import sys

from setuptools import setup
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(name='musket_core',
      version='0.498',
      description='The core of Musket ML',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/musket-ml/musket_core',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/aleju/imgaug'],
      entry_points={
            'console_scripts': [
                  'musket = musket_core.cli:main'
            ]
      },
      install_requires=[
            "hyperopt>=0.1.2,<=0.2.1",
            "PyYAML>=3.13,<=5.1.2",
            "numpy>=1.15.4,<=1.17.3",
            "Keras>=2.2.4,<=2.3.1",
            "tqdm>=4.28.1,<=4.36.1",
            "pandas>=0.23.4,<=0.25.2",
            "scikit-learn>=0.20.2,<=0.21.3",
            "scikit-image==0.15.0",
            "Pillow>=5.4.0,<=6.2.1",
            "scikit-image>=0.14.2",
            "Shapely>=1.6.4.post1,<=1.6.4.post2",
            "imgaug==0.3.0",
            "lightgbm>=2.2.3,<=2.3.0",
            "py4j==0.10.8.1",
            "async-promises==1.1.1",
            "scipy>=1.2.0,<=1.3.1",
            "pandas>=0.23.4,<=0.25.2",
            "matplotlib>=3.0.2,<=3.1.1",
            "imageio>=2.4.1,<=2.6.1",
            "h5py>=2.9.0,<=2.10.0",
            "opencv-python>=3.4.5.20,<=4.1.1.26",
            "Cython>=0.29.2,<=0.29.13",
            "segmentation-models>=0.2.0,<=0.2.1"
            "image-classifiers>=0.2.0,<=0.2.1",
      ],
      zip_safe=False)