#!/bin/bash

#########################################
# run this script use command following:
# yhbatch -N 1-1 -n 1 -p gpu_v100 startrun.sh
#########################################

#export CUDA_VISIBLE_DEVICES=2,3
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --model AAN --scale 4 --patch_size 256 --batch_size 32  --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --reset --save ann_x4


#export CUDA_VISIBLE_DEVICES=0
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV11 --reset --save biann_v11g_x2

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNPSV3 --reset --save bifsrcnnps_v3a_ls_x2

#python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNLIV4 --reset --save bifsrcnnLI_v4_x2

# ###################################
# SRARN settings like ConvNeXt
# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 4 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --depths 3+3+27+3 --dims 128+256+512+1024 --model SRARNV2 --save ../srarn/srarn_v2e7_x2 --reset

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --skip_threshold 1e6 --epochs 3000 --data_test Set5+Set14 --loss 1*SmoothL1 --lr 4e-3 --n_colors 1 --optimizer AdamW --weight_decay 0.05 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_div_opt_x2 --reset

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --skip_threshold 1e6 --epochs 3000 --data_train DF2K --data_range 1-3550 --data_test Set5+Set14 --loss 1*SmoothL1 --lr 4e-3 --n_colors 1 --optimizer AdamW --weight_decay 0.05 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_df_opt_x2 --reset

# 得分比较高的设置
# nohup python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --n_up_feat 64 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_x2 --reset > ../srarn/v2d5.log 2>&1 &


# nohup python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV3 --save ../srarn/srarn_v3d5_x2 --reset > ../srarn/v3d5_x2.log 2>&1 &

# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 0 --depths 3+3+27+3 --dims 128+256+512+1024 --model SRARNV3 --save ../srarn/srarn_v3e8_x2 --reset > ../srarn/v3e8_x2.log 2>&1 &


##################################################################################
######################      SRARN V4       ######################
##################################################################################
# SRARN V4 settings like SwinIR-B
# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV4 --save ../srarn/srarn_v4g12_x2 --reset > ../srarn/v4g12_x2.log 2>&1 &

# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV3 --save ../srarn/srarn_v3g12_x2 --pre_train ../srarn/srarn_v3g12_x2/model/model_best.pt --test_only --save_result --inf_switch

# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV3 --save ../srarn/srarn_v3g12_x2 --pre_train ../srarn/srarn_v3g12_x2/model/inf_model.pt --test_only --save_result --load_inf

# #####################################
# SRARN V4 settings like SwinIR-S
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV4 --save ../srarn/srarn_v4s_f11_x2 --reset > ../srarn/v4s_f11_x2.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV4 --save ../srarn/srarn_v4s_f11_x3 --reset > ../srarn/v4s_f11_x3.log 2>&1 &

# nohup python main.py --n_GPUs 3 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV4 --save ../srarn/srarn_v4s_f11_x4 --reset > ../srarn/v4s_f11_x4.log 2>&1 &


# #####################################
# SRARN V4 settings for tiny size (T)
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV4 --save ../srarn/srarn_v4t_c14_x2 --reset > ../srarn/v4t_c14_x2.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV4 --save ../srarn/srarn_v4t_c14_x3 --reset > ../srarn/v4t_c14_x3.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV4 --save ../srarn/srarn_v4t_c14_x4 --reset > ../srarn/v4t_c14_x4.log 2>&1 &

# #####################################
# SRARN V4 settings for extremly tiny size (XT) inf:67.6K
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV4 --save ../srarn/srarn_v4xt_j15_x2 --reset > ../srarn/v4xt_j15_x2.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV4 --save ../srarn/srarn_v4xt_j15_x3 --reset > ../srarn/v4xt_j15_x3.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV4 --save ../srarn/srarn_v4xt_j15_x4 --reset > ../srarn/v4xt_j15_x4.log 2>&1 &


################################################################################
######################      SRARN V5       ######################
################################################################################

# #####################################
# like SwinIR-B
# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 500 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x2 --reset > ../srarn/v5_ps_g12_x2.log 2>&1 &
# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 500 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x2_test --pre_train ../srarn/srarn_v5_ps_g12_x2/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 500 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x2_test --pre_train ../srarn/srarn_v5_ps_g12_x2_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 2 --scale 3 --patch_size 144 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x3 --reset > ../srarn/v5_ps_g12_x3.log 2>&1 &
python main.py --n_GPUs 2 --scale 3 --patch_size 144 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x3_test --pre_train ../srarn/srarn_v5_ps_g12_x3/model/model_best.pt --test_only --inf_switch
python main.py --n_GPUs 2 --scale 3 --patch_size 144 --batch_size 32 --accumulation_step 2 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x3_test --pre_train ../srarn/srarn_v5_ps_g12_x3_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 2 --scale 4 --patch_size 192 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x4 --reset > ../srarn/v5_ps_g12_x4.log 2>&1 &
python main.py --n_GPUs 2 --scale 4 --patch_size 192 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x4_test --pre_train ../srarn/srarn_v5_ps_g12_x4/model/model_best.pt --test_only --inf_switch
python main.py --n_GPUs 2 --scale 4 --patch_size 192 --batch_size 32 --accumulation_step 2 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_g12_x4_test --pre_train ../srarn/srarn_v5_ps_g12_x4_test/model/inf_model.pt --test_only --save_result --load_inf

# #####################################
# like SwinIR-B and same lr decay
# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 500 --decay 250-400-450-475 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV5 --save ../srarn/srarn_v5_ps_lrdc_g12_x2 --reset > ../srarn/v5_ps_lrdc_g12_x2.log 2>&1 &

# #####################################
# like SwinIR-S
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x2 --reset > ../srarn/v5s_f11_x2.log 2>&1 &
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x2_test --pre_train ../srarn/srarn_v5s_f11_x2/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x2_test --pre_train ../srarn/srarn_v5s_f11_x2_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x3 --reset > ../srarn/v5s_f11_x3.log 2>&1 &
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x3_test --pre_train ../srarn/srarn_v5s_f11_x3/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1250-1500-1750-1875 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x3_test --pre_train ../srarn/srarn_v5s_f11_x3_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1000-1250-1375 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x4 --reset > ../srarn/v5s_f11_x4.log 2>&1 &
# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 1375 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --load ../srarn/srarn_v5s_f11_x4 --resume -1 >> ../srarn/v5s_f11_x4.log 2>&1 &
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1000-1250-1375 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x4_test --pre_train ../srarn/srarn_v5s_f11_x4/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1000-1250-1375 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV5 --save ../srarn/srarn_v5s_f11_x4_test --pre_train ../srarn/srarn_v5s_f11_x4_test/model/inf_model.pt --test_only --save_result --load_inf

# #####################################
# for tiny size (T) c14
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_c14_x2 --reset > ../srarn/v5t_c14_x2.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_c14_x3 --reset > ../srarn/v5t_c14_x3.log 2>&1 &

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_c14_x4 --reset > ../srarn/v5t_c14_x4.log 2>&1 &


# for tiny size (T) i14
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x2 --reset > ../srarn/v5t_i14_x2.log 2>&1 &
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x2_test --pre_train ../srarn/srarn_v5t_i14_x2/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x2_test --pre_train ../srarn/srarn_v5t_i14_x2_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x3 --reset > ../srarn/v5t_i14_x3.log 2>&1 &
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x3_test --pre_train ../srarn/srarn_v5t_i14_x3/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x3_test --pre_train ../srarn/srarn_v5t_i14_x3_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x4 --reset > ../srarn/v5t_i14_x4.log 2>&1 &
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x4_test --pre_train ../srarn/srarn_v5t_i14_x4/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV5 --save ../srarn/srarn_v5t_i14_x4_test --pre_train ../srarn/srarn_v5t_i14_x4_test/model/inf_model.pt --test_only --save_result --load_inf


# #####################################
# for extremly tiny size (XT) inf:K
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 2750 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x2 --reset > ../srarn/v5xt_j15_x2.log 2>&1 &
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 2750 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x2 --pre_train ../srarn/srarn_v5xt_j15_x2/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 2750 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x2_test --pre_train ../srarn/srarn_v5xt_j15_x2/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x3 --reset > ../srarn/v5xt_j15_x3.log 2>&1 &
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x3_test --pre_train ../srarn/srarn_v5xt_j15_x3/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x3_test --pre_train ../srarn/srarn_v5xt_j15_x3_test/model/inf_model.pt --test_only --save_result --load_inf

# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x4 --reset > ../srarn/v5xt_j15_x4.log 2>&1 &
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x4_test --pre_train ../srarn/srarn_v5xt_j15_x4/model/model_best.pt --test_only --inf_switch
# python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5+Set14+B100+Urban100+Manga109 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../srarn/srarn_v5xt_j15_x4_test --pre_train ../srarn/srarn_v5xt_j15_x4_test/model/inf_model.pt --test_only --save_result --load_inf

