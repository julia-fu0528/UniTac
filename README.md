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

## Run

**Step1: Data Collection**

```
cd scripts/

python store_robot_state.py --markers_path your_marker_path --output_dir your_output_path --robot_type spot_or_franka --duration 2 --hostname xxx.xxx.xxx.xxx
```

**Step2: Data Preprocessing**
```
python dataset.py --session spot_dataset --data_dir ../data --markers_path your_marker_path --seq 1 --robot_type spot_or_franka
```

**Step3: Training**
```
python train.py --session spot_dataset --data_dir ../data --markers_path your_marker_path --device "gpu" --seq 1 --robot_type spot_or_franka
```

**Step4: Online Prediction**
Without physical Human Robot Interaction:
```
python predict.py --ckpts_path ../unitac_net_logs/spot/regression/version_0/checkpoints/best.ckpt --markers_path your_marker_path --data_dir ../data/spot_dataset --device cpu --seq 1 --robot_type spot_or_franka --hostname xxx.xxx.xxx.xxx
```

With physical Human Robot Interaction:
```
python predict.py --ckpts_path ../unitac_net_logs/spot/regression/version_0/checkpoints/best.ckpt --markers_path your_marker_path --data_dir ../data/spot_dataset --device cpu --seq 1 --robot_type spot_or_franka --hostname xxx.xxx.xxx.xxx \
--choreo --choreography-filepaths ../choreo/lay_down.txt ../choreo/sit.txt\
../choreo/pace_right.txt ../choreo/pace_left.txt\
../choreo/turn_left_back.txt ../choreo/turn_right_front.txt ../choreo/turn_right_back.txt ../choreo/turn_left_front.txt\
../choreo/tilt_left_back.txt ../choreo/tilt_left_front.txt ../choreo/tilt_right_back.txt ../choreo/tilt_right_front.txt\
../choreo/step_back_left.txt ../choreo/step_back_right.txt ../choreo/step_front_left.txt ../choreo/step_front_right.txt\
../choreo/play_bow.txt ../choreo/play_bow_happy.txt

```

**Run Everything**
```
sh scripts/run.sh
```