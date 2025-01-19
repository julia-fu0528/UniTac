
# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py state --hostname 138.16.161.22 \
#                             --markers_path ../data/franka_10markers_pos.txt\
#                             --output_dir ../data/franka_right/test \
#                             --robot_type franka --duration 2\


# echo "################# DATALOADER ###############################"
# python dataset.py --session franka_right --data_dir ../data \
#                     --markers_path ../data/franka_10markers_pos.txt \
#                     --seq 1  --robot_type franka --classify \

echo "################# TRAINING ###############################"
python train.py --session franka_right --data_dir ../data \
                --markers_path ../data/franka_10markers_pos.txt --device "gpu" \
                --seq 1 --robot_type franka --classify


# echo "################# PREDICTING ################################"
# python predict.py --ckpts_path ../gouger_logs/franka/regression/version_493/checkpoints/best.ckpt\
#                   --markers_path ../data/franka_10markers_pos.txt --data_dir ../data/franka_right --device cpu --seq 1 --robot_type franka\
#                 #   --choreography-filepaths ../choreo/step.txt ../choreo/trot.txt ../choreo/turn_2step.txt ../choreo/twerk.txt ../choreo/unstow.txt\







# nc 20 seconds all tusker




# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
# pip install urdfpy
# pip install lightning
# pip install open3d
# pip install seaborn
# pip install -U 'tensorboard'



# numpy 2.0.2
