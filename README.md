# UniTac: Whole-Robot Touch Sensing Without Tactile Sensors

## Installation
```
git clone git@github.com:julia-fu0528/UniTac.git
```

## Virtual Environment

```
python3 -m pip install virtualenv
python3 -m virtualenv --python=/usr/bin/python3 unitac_env
source unitac_env/bin/activate

python3 -m pip install --upgrade bosdyn-client bosdyn-mission bosdyn-choreography-client bosdyn-orbit
```
**Packages**
```
pip install lightning==2.6.0
pip install transformers -U
pip install tokenizers==0.20.0 matplotlib==3.9.4
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install scikit-learn==1.6.1 urdfpy==0.0.22
pip install networkx --upgrade
pip install open3d==0.18.0 seaborn==0.13.2
pip install -U 'tensorboard'
pip install natsort
pip install "numpy<1.24"
```

## Downloads
**Third Party**
- Download SPOT mesh urdf and related meshes [here](https://drive.google.com/drive/folders/1IT3_eVo6WOz9hAca2uvpmRWlpXH9W97E?usp=sharing) into the root directory of this repository
- Download FR3 mesh urdf and related meshes from the [official repository](https://github.com/frankaemika/franka_description/tree/main/robots/fr3) into the root director of this repository.

**Dataset**
- Downlaod the sampled ground truth touch locations on [Spot](https://drive.google.com/file/d/1StENgdsREB_C42rrWShTXtaZ_9jWJE9l/view?usp=drive_link) and [FR3](https://drive.google.com/file/d/1IeHhX6Kyk7Yhe_Ho4edR7fjRe1DOHGl4/view?usp=drive_link) into data/ of the repository's root.
- Download the dataset of [Spot](https://drive.google.com/drive/folders/1WWiTWrCvt33Famq9ckpXRQK160WsTJ2r?usp=drive_link) and [FR3](https://drive.google.com/drive/folders/1gK3e4d8Hmp2NV2_uDqsMIq30uzYho65u?usp=drive_link) into spot_dataset/ and fr3_dataset/ of the data directory.


**Preprocessed Data**

If you want to train a model on preprocessed data, you can use the regression model of the Spot as an example.

Download the [preprocessed data](https://drive.google.com/drive/folders/1zwsH1k3OcTa9yv2RlmUCfLDm_fMVFNjO?usp=drive_link) into preprocessed_data/ of the repository's root directory.

**Model Checkpoint**

If you want to test the model on your robot, you can use the regression model of the Spot as an example.

Download the [model checkpoint](https://drive.google.com/drive/folders/1gu0QhYxPT4Lt_cYvq38_O4iZJHX51y-0?usp=drive_link) into unitac_net_logs/spot/regression/ of the repository's root.

