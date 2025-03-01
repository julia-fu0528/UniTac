
# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py --markers_path ../data/gouger_markers_pos.txt\
#                             --output_dir ../data/test \
#                             --robot_type spot --duration 2 \
#                             # --hostname 138.16.161.22 \


# echo "################# DATALOADER ###############################"
# python dataset.py --session gouger1209 --data_dir ../data \
#                     --markers_path ../data/gouger_markers_pos.txt \
#                     --seq 1  --robot_type spot --classify \

echo "################# TRAINING ###############################"
python train.py --session gouger1209 --data_dir ../data \
                --markers_path ../data/gouger_markers_pos.txt --device "gpu" \
                --seq 1 --robot_type spot --classify \


# echo "################# PREDICTING ################################"
# python predict.py --ckpts_path ../gouger_logs/franka/regression/version_611/checkpoints/best.ckpt\
#                   --markers_path ../data/franka_10markers_pos.txt --data_dir ../data/franka_right --device cpu --seq 1 --robot_type franka \
#                   --hostname 138.16.161.22 \
#                   --choreo \
#                   --choreography-filepaths ../choreo/lay_down.txt ../choreo/sit.txt\
#                                            ../choreo/pace_right.txt ../choreo/pace_left.txt\
#                                            ../choreo/turn_left_back.txt ../choreo/turn_right_front.txt ../choreo/turn_right_back.txt ../choreo/turn_left_front.txt\
#                                            ../choreo/tilt_left_back.txt ../choreo/tilt_left_front.txt ../choreo/tilt_right_back.txt ../choreo/tilt_right_front.txt\
#                                            ../choreo/step_back_left.txt ../choreo/step_back_right.txt ../choreo/step_front_left.txt ../choreo/step_front_right.txt\
#                                            ../choreo/play_bow.txt ../choreo/play_bow_happy.txt






# nc 20 seconds all tusker




# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
# pip install urdfpy
# pip install lightning
# pip install open3d
# pip install seaborn
# pip install -U 'tensorboard'



# numpy 2.0.2
