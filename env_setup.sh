#!/bin/bash
set -e
case "$1" in
    "cpu") gpu=false;;
    "gpu") gpu=true;;
    *) echo "incorrect args. Supply either \"cpu\" or \"gpu\""
       exit 1;;
esac
 
conda remove -y --name 4YP --all
conda create -y --name 4YP python
conda install -n 4YP -y -c anaconda networkx
if [ $gpu = true ]; then
    # pytorch gpu install
    # pytorch.org keeps making this mistake, whereby they neglect to mention that -c conda-forge is necessary!
    conda install -n 4YP -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge
    conda install -n 4YP -c conda-forge faiss-gpu
    torch_geom_flavour="cu111"
else
    conda install -n 4YP -y pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
    torch_geom_flavour="cpu"
    conda install -n 4YP -c conda-forge faiss-cpu
fi
conda install -n 4YP -c conda-forge pycurl
conda run -n 4YP python3 -m pip install --upgrade pip
conda run -n 4YP python3 -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${torch_geom_flavour}.html
conda run -n 4YP python3 -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${torch_geom_flavour}.html
conda run -n 4YP python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${torch_geom_flavour}.html
conda run -n 4YP python3 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${torch_geom_flavour}.html
conda run -n 4YP python3 -m pip install torch-geometric
conda run -n 4YP python3 -m pip install -r kilt_requirements.txt
conda run -n 4YP python3 -m pip install -r requirements.txt # 4YP specific requirements