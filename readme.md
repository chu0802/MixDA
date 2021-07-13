# MixDA
My research related to the mixup strategy for Domain Adaptation

## Installation

#### Python

Install python 3.8.2 on any virtual environment you like.

*e.g. pyenv virtualenv*

* First, install [pyenv](https://github.com/pyenv/pyenv), and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
* Install python 3.8.2
    ```
    pyenv install 3.8.2
    ```
* Create a blank virtual environment
    ```
    pyenv virtualenv 3.8.2 <NAME>
    ```
* Start up the virtual environment
    ```
    pyenv shell <NAME>
    ```

#### Packages

After starting up the virtual environments, install all dependent packages through the below steps.

1. Update pip to the latest version
    ```
    pip install --upgrade pip
    ```

1. Install pytorch from [here](https://pytorch.org/get-started/previous-versions) according to the corresponding cuda version.
    ```
    torch==1.7.1
    torchaudio==0.7.2
    torchvision==0.8.2
    ```
2. Install the remain packages through `requirements.txt`
    ```
    pip install -r requirements.txt
    ```

#### Exception

If there are the following warnings when importing torchvision:
```
No module named 'backports'
No module named 'backports.lzma'
```

Install the following packages:
```
pip install backports.weakref
pip install backports.lzma
```
