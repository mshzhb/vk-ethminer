# Vulkan-ethminer

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

> Ethereum/Ethereum Classic miner with Vulkan support

**Vulkan-ethminer** is the first an Ethash GPU mining worker based on Vulkan: with the miner you can mine every coin which relies on an Ethash Proof of Work thus including Ethereum (deprecated), Ethereum Classic, Metaverse, Musicoin, Ellaism, Pirl, Expanse and others. This is the actively maintained version of Vulkan-ethminer. It originates from [cpp-ethereum] project (where GPU mining has been discontinued) and builds on vulkan api. This project should be able to build on most of the platforms with VK support.

## Features

* Vulkan mining
* on-GPU DAG generation (no more DAG files on disk)
* stratum mining without proxy
* zero dev fee


## Table of Contents

* [Install](#install)
* [Usage](#usage)
* [Build](#build)
* [Maintainers & Authors](#maintainers--authors)
* [Contribute](#contribute)
* [F.A.Q.](#faq)


## Install

Standalone **executables** for *Linux*, *macOS* and *Windows* are provided in
the [Releases](https://github.com/mshzhb/vulkan-ethminer/releases) section.
Download an archive for your operating system and unpack the content to a place
accessible from command line. The ethminer is ready to go.

## Usage

The **vulkan-ethminer** is a command line program. This means you launch it either
from a Windows command prompt or Linux console, or create shortcuts to
predefined command lines using a Linux Bash script or Windows batch/cmd file.
For a full list of available command, please run:

```sh
./vulkan_ethminer --help
```

common run cli
```sh
#please replace 0x.. with your wallet
#On x86 platforms it will automatically run on all discrete gpus. On arm (e.g. apple) it will run on all gpus. 
./vulkan_ethminer --server us1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle
```

list all devices
```sh
./vulkan_ethminer -l
```

explicitly run miner on GPU 0 and GPU 1
```sh
./vulkan_ethminer -d 0 1 ....
```

## Build
### Building from source

This project uses [CMake] and [Hunter] package manager.

### Common

1. [CMake](https://cmake.org/) >= 3.6
2. [Git](https://git-scm.com/downloads)
3. [Vulkan-SDK](https://vulkan.lunarg.com) >= 1.3
4. python3
5. build-tools

### Linux & Mac OS
requires gcc/clang and make on linux/macOS    
  
using build script
``` shell
./setup.py
```
manual build
``` shell
cd vulkan-etherminer
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
### Windows
requires visual studio 2019+ on windows  
  
using build script
``` shell
setup.py
```
manual build
``` shell
cd vulkan-etherminer
mkdir build
cd build
cmake ..
cmake --build . --config Release
Xcopy shaders Release\shaders\ /E /H /C /I
```
## Maintainers & Authors

| Name                  | Contact                                                      |     |
| --------------------- | ------------------------------------------------------------ | --- |
| Tong Liu              | [@mshzhb](https://github.com/mshzhb/vulkan-ethminer)         |   0x0d405dc4889de1512bfdefa0007c3b6aa468e31a  |


## Contribute

All bug reports, pull requests and code reviews are very much welcome.


## License

Licensed under the [MIT License](LICENSE).

## F.A.Q
