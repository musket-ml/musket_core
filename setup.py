from setuptools import setup
import setuptools
setup(name='musket_core',
      version='0.42',
      description='Common parts of my pipelines',
      url='https://github.com/petrochenko-pavel-a/musket_core',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/aleju/imgaug'],
      install_requires=["numpy", "scipy","Pillow", "cython","pandas","matplotlib", "scikit-image","keras>=2.2.4","imageio",
"opencv-python",
"h5py",
"tqdm",
"segmentation_models", "lightgbm", "async_promises"],
      zip_safe=False)