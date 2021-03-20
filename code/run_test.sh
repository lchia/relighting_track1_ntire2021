EPOCH1=0128
EPOCH2=0090
EPOCH3=0123
EPOCH4=0124
EPOCH5=0134

CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --test_only \
    --test_checkpoint ../ckpts/epoch_${EPOCH1}_checkpoint.pth.tar \
    --test_save_dir ../TMP/${EPOCH1} \
    --pynet_level 0 \
    --num_gpus 1 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss '1*L1_Charbonnier_loss_color' \
    --log_tag 'rgbin' \
    --data_root ../data \
    --dataset track1 \
    --data_format RGB \
    --model pynet3 \
    --n_colors 8 \
    --n_features 16 \
    --n_features_level_1 2 \
    --n_blocks 16 \
    --trainer_mode dp \
    --GT hdr \
    --loss_supervised diffE2E \
    --save_dir ./experiment \
    --profile_H 1024 \
    --profile_W 1024 \
    --start_epoch 1 \
    --end_epoch 500 \
    --metrics 'PSNR|SSIM' \
    --with_in \
    --save_dense \

"""
CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --test_only \
    --test_checkpoint ../ckpts/epoch_${EPOCH2}_checkpoint.pth.tar \
    --test_save_dir ../TMP/${EPOCH2} \
    --pynet_level 0 \
    --num_gpus 1 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss '1*L1_Charbonnier_loss_color' \
    --log_tag 'rgbin' \
    --data_root ../data \
    --dataset track1 \
    --data_format RGB \
    --model pynet3 \
    --n_colors 8 \
    --n_features 16 \
    --n_features_level_1 2 \
    --n_blocks 16 \
    --trainer_mode dp \
    --GT hdr \
    --loss_supervised diffE2E \
    --save_dir ./experiment \
    --profile_H 1024 \
    --profile_W 1024 \
    --start_epoch 1 \
    --end_epoch 500 \
    --metrics 'PSNR|SSIM' \
    --with_in \
    --save_dense \


CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --test_only \
    --test_checkpoint ../ckpts/epoch_${EPOCH3}_checkpoint.pth.tar \
    --test_save_dir ../TMP/${EPOCH3} \
    --pynet_level 0 \
    --num_gpus 1 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss '1*L1_Charbonnier_loss_color' \
    --log_tag 'rgbin' \
    --data_root ../data \
    --dataset track1 \
    --data_format RGB \
    --model pynet3 \
    --n_colors 8 \
    --n_features 16 \
    --n_features_level_1 2 \
    --n_blocks 16 \
    --trainer_mode dp \
    --GT hdr \
    --loss_supervised diffE2E \
    --save_dir ./experiment \
    --profile_H 1024 \
    --profile_W 1024 \
    --start_epoch 1 \
    --end_epoch 500 \
    --metrics 'PSNR|SSIM' \
    --with_in \
    --save_dense \


CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --test_only \
    --test_checkpoint ../ckpts/epoch_${EPOCH4}_checkpoint.pth.tar \
    --test_save_dir ../TMP/${EPOCH4} \
    --pynet_level 0 \
    --num_gpus 1 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss '1*L1_Charbonnier_loss_color' \
    --log_tag 'rgbin' \
    --data_root ../data \
    --dataset track1 \
    --data_format RGB \
    --model pynet3 \
    --n_colors 8 \
    --n_features 16 \
    --n_features_level_1 2 \
    --n_blocks 16 \
    --trainer_mode dp \
    --GT hdr \
    --loss_supervised diffE2E \
    --save_dir ./experiment \
    --profile_H 1024 \
    --profile_W 1024 \
    --start_epoch 1 \
    --end_epoch 500 \
    --metrics 'PSNR|SSIM' \
    --with_in \
    --save_dense \


CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --test_only \
    --test_checkpoint ../ckpts/epoch_${EPOCH5}_checkpoint.pth.tar \
    --test_save_dir ../TMP/${EPOCH5} \
    --pynet_level 0 \
    --num_gpus 1 \
    --batch_size 4 \
    --lr 0.0001 \
    --loss '1*L1_Charbonnier_loss_color' \
    --log_tag 'rgbin' \
    --data_root ../data \
    --dataset track1 \
    --data_format RGB \
    --model pynet3 \
    --n_colors 8 \
    --n_features 16 \
    --n_features_level_1 2 \
    --n_blocks 16 \
    --trainer_mode dp \
    --GT hdr \
    --loss_supervised diffE2E \
    --save_dir ./experiment \
    --profile_H 1024 \
    --profile_W 1024 \
    --start_epoch 1 \
    --end_epoch 500 \
    --metrics 'PSNR|SSIM' \
    --with_in \
    --save_dense \

python fuse.py

# rm -rf ../TMP
"""