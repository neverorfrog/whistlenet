#!/bin/bash

#distinguishes between zsh and bash
DEFAULT_SHELL=$(echo $SHELL | awk -F/ '{print $NF}')

#conda env
conda create -yn executorch python=3.10.0
conda activate executorch
conda install ipykernel ipywidgets scikit-learn matplotlib pyyaml -y

#user input
echo "Absolute path to which you want to download the executorch folder"
read TORCH_PATH
echo 'export TORCH_PATH='${TORCH_PATH} >> ~/.${DEFAULT_SHELL}rc
echo 'export PATH="'${TORCH_PATH}':${PATH}"' >> ~/.${DEFAULT_SHELL}rc
mkdir -p ${TORCH_PATH}

#libtorch clone
cd /tmp
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest -d ${TORCH_PATH}

#buck2
pip3 install zstd
wget https://github.com/facebook/buck2/releases/download/2024-02-15/buck2-x86_64-unknown-linux-musl.zst
zstd -cdq buck2-*.zst > ${TORCH_PATH}/buck2e && chmod +x ${TORCH_PATH}/buck2e

#executorch clone
cd ${TORCH_PATH}
git clone -b v0.2.0 https://github.com/pytorch/executorch.git
cd executorch
git submodule sync
git submodule update --init

#requirements of executorch
./install_requirements.sh
export PATH="'${TORCH_PATH}'/executorch/third-party/flatbuffers/cmake-out:${PATH}"
echo 'export PATH="'${TORCH_PATH}'/executorch/third-party/flatbuffers/cmake-out:${PATH}"' >> ~/.${DEFAULT_SHELL}rc
./build/install_flatc.sh