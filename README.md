# vh

## Third party
- Download SPOT mesh urdf and related meshes.
- Download FRANKA mesh urdf and related meshes from the [official repository](https://github.com/frankaemika/franka_description/tree/main/robots/fr3)

## Setup 

**Spot**
```
python3 -m pip install virtualenv
python3 -m virtualenv --python=/usr/bin/python3 my_spot_env
source my_spot_env/bin/activate

python3 -m pip install --upgrade bosdyn-client bosdyn-mission bosdyn-choreography-client bosdyn-orbit
```
**Packages**
```
pip install lightning
pip install transformers -U
pip install tokenizers==0.20.0 matplotlib
pip install torch torchvision torchaudio 
pip install scikit-learn urdfpy
pip install networkx --upgrade
pip install open3d seaborn 
pip install -U 'tensorboard'
```