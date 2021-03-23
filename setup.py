from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

print(find_packages())

setup(name='lardon',
      version='0.1.1',
      description='numpy memmap front-end for large data indexing',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/domkkirke/lardon',
      author='domkirke',
      author_email='domkirke@wolfgang.wang',
      license='GPLv3',
      packages=find_packages(),
      install_requires=['numpy',
        'dill',
        'tqdm'
     ],
     classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
     ],
     python_requires='>=3.6',
     keywords='python, memory, data, import',
     zip_safe=False)
