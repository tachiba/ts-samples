# ts-samples

Just learning footprints.

## Installing Python 3.5

Decide the version by
https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#python_version_support

`brew install python3` enforces to use the latest python.

https://github.com/pyenv/pyenv

```
brew install pyenv
pyenv install 3.5.6
brew install pyenv-virtualenv
```

Need to tweak shell config.

https://github.com/pyenv/pyenv#basic-github-checkout

## Building environment

https://github.com/pyenv/pyenv-virtualenv#usage

```
pyenv local 3.5.6
pyenv virtualenv 3.5.6 3.5.6-ts
echo "3.5.6-ts" > .python-version
```

## Installing TensorFlow

https://www.tensorflow.org/install/install_mac?hl=ja

```
pip3 install --upgrade tensorflow

```

## Validate the installation

https://www.tensorflow.org/install/install_mac?hl=ja#ValidateYourInstallation
