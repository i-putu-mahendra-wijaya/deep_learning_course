# Deep Learning

This repository contains code snippets used in my `Deep Learning` course

There are few preparation steps before running the code in this repository: 

> **Note:** The following steps are for Linux / MacOS users.
> For Windows users, I would suggest using [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

## 0. Install `git` and your favorite IDE

Please refer to the [official documentation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
for more details how to install `git` for your operating system.

And then, install an Integrated Development Environment (IDE) of your choice.

Few popular IDEs are:

1. VS Code. Please refer to the [official documentation](https://code.visualstudio.com/docs/setup/setup-overview) 
for more details, or
2. PyCharm. Please refer to the [official documentation](https://www.jetbrains.com/help/pycharm/installation-guide.html) 
for more details.

## 1. Install `pyenv` and `virtualenv`

Pyenv is a python version manager. It allows you to install multiple versions of python on your system. 
This is useful since different versions of python may be required for different projects.

For those who have had experience with Python before; `pyenv` CAN NOT be installed from `pip` or `conda`.
You need to install `pyenv` from its source. 

Please refer to the [official documentation](https://github.com/pyenv/pyenv#installation) for more details, 
and follow the steps to install `pyenv` according to your operating system.

On another hand, `virtualenv` is a tool that allows you to create isolated python environments.
It is useful to prevent packages from one project from affecting another project.

Unlike `pyenv`, `virtualenv` can be installed from `pip` or `conda`. 
Please refer to the [official documentation](https://virtualenv.pypa.io/en/latest/) for more details on how to 
install `virtualenv`.

## 2. Create a folder for the `deep-learning` course`materials

Launch a terminal and run the following commands:

```shell
mkdir -p ~/deep_learning_materials && cd ~/deep_learning_materials
```
