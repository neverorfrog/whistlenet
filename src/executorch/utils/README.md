Install Steps (Tested on Ubuntu 22.04)

1) Install Miniconda
```shell 
bash -i miniconda-install.sh
```
2) Install Executorch and Libtorch with the necessary packages in a conda env
```shell
bash -i install.sh
```
- Follow the instructions
- It will take a few minutes

3) Build Executorch with CMake
```shell
bash -i build.sh
```