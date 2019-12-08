[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Development Status](https://github.com/ElPapi42/DeepBay/workflows/build/badge.svg?branch=development)](https://github.com/ElPapi42/DeepBay/actions?query=workflow%3A%22Test+Package%22)
[![GitHub version](https://badge.fury.io/gh/Elpapi42%2FDeepBay.svg)](https://badge.fury.io/gh/Elpapi42%2FDeepBay)
[![Last Commit](https://badgen.net/github/last-commit/ElPapi42/DeepBay)](https://github.com/ElPapi42/DeepBay/graphs/commit-activity)


# DeepBay
This project was created with the objective of compile Machine Learning Architectures created using Tensorflow or Keras. The architectures must be provided as a ready-to-use Plug-and-Play module that can be easily integrated into any existing project or architecture design.

## Installation
You can use pip for install this from [PyPi](https://pypi.org/project/deepbay/):
```
pip install deepbay
```

## Quick Start
You can use any architecture inside deepbay as an self-contained model ready to be trained:
```python
import tensorflow as tf
import deepbay

denseblock = deepbay.DenseBlock(units=1)
```
Or you can integrate it to any existing architecture, just use it as any other keras layer:
```python
import tensorflow as tf
import deepbay

model = tf.keras.models.Sequential()
model.add(deepbay.DenseBlock(units=1))
```
