# Tools Basics

Repository containing basic examples of various tools used for robotics.

# Install Prerequisites

## Linux

How to install latest python via apt package manager.

Add deadsnakes personal package archive for latest python releases.

```python
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```

Install python 3.12 or greater.

```python
sudo apt update
sudo apt install python3.12-dev python3.12-venv
```

(*Note*: Do not change your system default `python3` instead call the installed python binaries via the full name e.g. `python3.12`)

## MacOS

How to install latest python via Homebrew.

```python
brew install python@3.12
```

## Windows

For windows installation refer to the official documentation: <https://docs.python.org/3/using/windows.html>

# Setup Virtual Environment and Install Packages:

## Linux and MacOS

Navigate to the directory where you want to setup the environment. It is recommended to set it up in the root of this repository.

Setup the virtual environment:

```python
python3.12 -m venv env
```

Source the virtual environment:

```python
source env/bin/activate
```

Upgrade pip and install packages:

```python3
pip install --upgrade pip
pip install numpy
pip install mujoco
```

## Windows

Refer to the official documentation: <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>

(*Note*: Make sure you are using the python installed in the previous step)
