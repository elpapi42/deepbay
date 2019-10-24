import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

version="v0.2"

setuptools.setup(
	name="deepbay",
	version=version,
	author="Whitman Bohorquez",
	author_email="whitman-2@hotmail.com",
	description="Tensorflow/Keras Plug-N-Play Deep Learning Models Compilation",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/ElPapi42/DeepBay",
	download_url = "https://github.com/ElPapi42/DeepBay/archive/" + version + ".tar.gz",
	keywords = ['Keras', 'Model', 'Plug And Play', "Tensorflow"],
	packages=setuptools.find_packages(),
	install_requires=[
		"tensorflow-gpu"
	],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.7',
)