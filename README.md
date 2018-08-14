# ts-samples

Just learning footprints.

## Installing Python 3.6

ML Engine only supports Python 3.5.x, but it causes a warning and annoying.
```
/Users/tachiba/.pyenv/versions/3.5.6/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.6 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
```

https://github.com/pyenv/pyenv
https://github.com/tensorflow/tensorflow/issues/14273 
https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#python_version_support

`brew install python3` enforces to use the latest python.

```
brew install pyenv pyenv-virtualenv
pyenv install 3.6.6 
```

Need to tweak shell config.

https://github.com/pyenv/pyenv#basic-github-checkout

## Building environment

https://github.com/pyenv/pyenv-virtualenv#usage

```
pyenv virtualenv 3.6.6 3.6.6-ts
echo "3.6.6-ts" > .python-version
```

## Installing TensorFlow

https://www.tensorflow.org/install/install_mac?hl=ja

```
pip3 install --upgrade tensorflow
```

## Validate the installation

https://www.tensorflow.org/install/install_mac?hl=ja#ValidateYourInstallation
