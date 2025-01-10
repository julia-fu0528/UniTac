
# echo "################# COLLECT DATA ###########################"
# python store_robot_state.py 138.16.161.22 state \
#                             --markers_path ../data/gouger_markers_pos.txt\
#                             --output_dir ../data/gouger1209/tf0 \
#                             --robot_type spot \


# echo "################# DATALOADER ###############################"
# python dataset.py --session gouger1209 --data_dir ../data \
#                     --markers_path ../data/gouger_markers_pos.txt --seq 1 \

# echo "################# TRAINING ###############################"
# python train.py --session gouger1209 --data_dir ../data \
#                 --markers_path ../data/gouger_markers_pos.txt --device "gpu" --seq 1 \


echo "################# PREDICTING ################################"
python predict.py 138.16.161.22 --ckpts_path ../gouger_logs/regression/version_190/checkpoints/best.ckpt\
                  --markers_path ../data/gouger_markers_pos.txt --data_dir ../data/gouger1209 --device cpu --seq 1 \
                  --choreography-filepaths ../choreo/step.txt ../choreo/trot.txt ../choreo/turn_2step.txt ../choreo/twerk.txt ../choreo/unstow.txt\







# nc 20 seconds all tusker




# pip3 install torch torchvision torchaudio
# pip install pytorch_lightning==1.6.0
# pip install urdfpy
# pip install lightning
# pip install open3d
# pip install seaborn
# pip install -U 'tensorboard'




