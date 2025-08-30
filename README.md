# Deep Learning Course

## Document History

| Date       | Version | Description | Author                 |
|------------|---------| ----------- |------------------------|
| 2025-08-30 | 0.0     | Initial version | I Putu Mahendra Wijaya |

---

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

> **Note:** `pyenv` is not a python package manager. It's a python version manager.

Verify that `pyenv` is installed by running the following command from your terminal:

```shell
pyenv --version # returns pyenv 2.5.5 in my machine. your output show a different version.
```

If you see the version number, then `pyenv` is installed successfully.

Install python 3.11.5 or later using the following command:

```shell
pyenv install 3.11.5
```

Verify that Python 3.11.5 is installed by running the following command from your terminal:

```shell
pyenv versions # returns 3.11.5 and other versions installed on my machine.
```

Once finished istalling `pyenv`, we need to install `virtualenv` - a tool that allows you to create isolated python environments.
It is useful to prevent packages from one project from affecting another project.

Unlike `pyenv`, `virtualenv` can be installed from `pip` or `conda`. 
Please refer to the [official documentation](https://virtualenv.pypa.io/en/latest/) for more details on how to 
install `virtualenv`.

## 2. Create a folder for the `deep-learning` course`materials

Launch a terminal and run the following commands:

```shell
mkdir -p ~/deep_learning_course && cd ~/deep_learning_course
```
The above command creates a folder named `deep_learning_course` in your home directory, 
and then changes the current working directory to the newly created folder.

> **Note:** `~/` is a shortcut for your home directory.
> **Note:** once you are inside the `deep_learning_materials` folder, your terminal prompt will show the current working directory: `~/deep_learning_course`.

## 3. Clone this repository

within the `deep_learning_course` folder, run the following command:

```shell
git clone https://github.com/i-putu-mahendra-wijaya/deep_learning_course.git
```

Inside the `deep_learning_course` folder, you will find `.python-version` file. 
This file is used by `pyenv` to automatically set the correct version of python for the project.

## 4. Create a virtual environment

Within the `deep_learning_course` folder, run the following command:

```shell
python3 -m virtualenv venv_deep_learning
source venv_deep_learning/bin/activate
```

The above command creates a virtual environment named `venv_deep_learning` in the `deep_learning_course` folder.
And then activates the virtual environment.

> **Note:** once you have activated the virtual environment, your terminal prompt will show : `(venv_deep_learning) ~/deep_learning_course/:`.

