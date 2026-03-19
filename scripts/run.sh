
echo "################# COLLECT DATA ###########################"
python store_robot_state.py --markers_path ../data/spot_markers_pos.txt \
                            --output_dir ../data/test \
                            --robot_type spot --duration 2 \
                            --hostname 128.148.138.22 \


echo "################# DATALOADER ###############################"
python dataset.py --session spot_dataset_tusker --data_dir ../data \
                    --markers_path ../data/spot_markers_pos.txt\
                    --seq 1 --robot_type spot  

echo "################# TRAINING ###############################"
python train.py --session spot_dataset_tusker --data_dir ../data \
                --markers_path ../data/spot_markers_pos.txt --device "gpu" \
                --seq 1 --robot_type spot \


echo "################# PREDICTING ################################"
python predict.py --ckpts_path ../unitac_net_logs/spot/regression/version_0/checkpoints/best.ckpt\
                  --markers_path ../data/spot_markers_pos.txt --data_dir ../data/spot_dataset_tusker --device cpu --seq 1 --robot_type spot \
                  --hostname 128.148.138.22 \
                  --choreo \
                  --choreography-filepaths ../choreo/lay_down.txt ../choreo/sit.txt\
                                           ../choreo/pace_right.txt ../choreo/pace_left.txt\
                                           ../choreo/turn_left_back.txt ../choreo/turn_right_front.txt ../choreo/turn_right_back.txt ../choreo/turn_left_front.txt\
                                           ../choreo/tilt_left_back.txt ../choreo/tilt_left_front.txt ../choreo/tilt_right_back.txt ../choreo/tilt_right_front.txt\
                                           ../choreo/step_back_left.txt ../choreo/step_back_right.txt ../choreo/step_front_left.txt ../choreo/step_front_right.txt\
                                           ../choreo/play_bow.txt ../choreo/play_bow_happy.txt





