<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [PyClConvolve](#pyclconvolve)
- [How to use](#how-to-use)
- [Notes on how the wrapper works](#notes-on-how-the-wrapper-works)
- [To install from pip](#to-install-from-pip)
- [To build directly](#to-build-directly)
  - [Pre-requisites:](#pre-requisites)
    - [Compilers](#compilers)
    - [Python packages](#python-packages)
  - [To build:](#to-build)
- [Considerations for Python wrapper developers](#considerations-for-python-wrapper-developers)
- [To build, obsolete method](#to-build-obsolete-method)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# PyClConvolve

Python wrapper for  [ClConvolve](https://github.com/hughperkins/ClConvolve)

# How to use

See [test_clconvolve.py](PyClConvolve/test_clconvolve.py) for an example of:

* creating a network, with several layers
* loading mnist data
* training the network using a higher-level interface (`NetLearner`)

For examples of using lower-level entrypoints, see [test_lowlevel.py](https://github.com/hughperkins/PyClConvolve/blob/master/test_lowlevel.py):

* creating layers directly
* running epochs and forward/backprop directly

# Notes on how the wrapper works

* [cClConvolve.pxd](https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/cClConvolve.pxd) contains the definitions of the underlying ClConvolve c++ libraries classes
* [PyClConvolve.pyx](https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/PyClConvolve.pyx) contains Cython wrapper classes around the underlying c++ classes
* [setup.py](https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/setup.py) is a setup file for compiling the `PyClConvolve.pyx` Cython file

# To install from pip

```bash
pip install DeepCL 
```

* related pypi page: [https://pypi.python.org/pypi/PyClConvolve](https://pypi.python.org/pypi/PyClConvolve)

# To build directly

## Pre-requisites:

### Compilers
* on Windows:
  * Python 2.7 build: need [Visual Studio 2008 for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266) from Microsoft
  * Python 3.4 build: need Visual Studio 2010, eg [Visual C++ 2010 Express](https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4)

### Python packages

* Need the following python packages installed, eg via `pip install`:
  * cython

## To build:

```bash
cd python
python setup.py build_ext -i
```

# Considerations for Python wrapper developers

* By default, cython disables ctrl-c, so need to handle this ourselves somehow, eg see `NetLearner.learn` for an example
* To handle ctrl-c, we need to use `nogil`, which means we cant use the `except +` syntax, I think, hence need to handle this ourselves too :-)  see again `Netlearner.learn for an example

