#!/bin/bash

################################################################################
######################      SRARN V5       ######################
################################################################################
# !!!!!!!!!!!!!!!!!! ATTENTION for multi-step LR decay when use resume !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ./scripts/train_srarn_v5.sh [mode] [cuda_device] [accummulation_step] [model] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [dataset] [loss]
# run example for v5test_x2: ./scripts/train_srarn_v5.sh train 0 1 test s 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# ########### training commands ###########

# run example for v5bn_x3: ./scripts/train_srarn_v5.sh train 0,1 1 bn b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5bn_x4: ./scripts/train_srarn_v5.sh train 2,3 1 bn b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5bn_x2: ./scripts/train_srarn_v5.sh train 0,1,2,3 1 bn b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5b_Nrst_x2: ./scripts/train_srarn_v5.sh train 0,1,2 1 b n 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5b_PxSh_x2: ./scripts/train_srarn_v5.sh train 0 4 b p 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5ba_x2: ./scripts/train_srarn_v5.sh train 0,1 1 ba b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5bL_x2: ./scripts/train_srarn_v5.sh train 0,1,2 1 bL b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5bnL_x2: ./scripts/train_srarn_v5.sh train 0 4 bnL b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5LBbL_x2: ./scripts/train_srarn_v5.sh train 1 4 LBbL b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5l_x2: ./scripts/train_srarn_v5.sh train 0,1 2 l b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5s_Nrst_x2: ./scripts/train_srarn_v5.sh train 3 1 s n 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5   # better
# run example for v5s_PxSh_x2: ./scripts/train_srarn_v5.sh train 3 1 s p 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5   # bad

# run example for v5bL_x2: ./scripts/train_srarn_v5.sh train 0,1,2 1 bL b 2 48 ms 1acb3 DF2K SmoothL1 SRARNV5
# run example for v5s_Nrst_x2: ./scripts/train_srarn_v5.sh train 3 1 s n 2 64 ms 1acb3 DF2K SmoothL1 SRARNV5
# run example for v5s_x2: ./scripts/train_srarn_v5.sh train 3 1 s b 2 64 ms 1acb3 DF2K SmoothL1 SRARNV5

# run example for v5s_s64_x2: ./scripts/train_srarn_v5.sh train 0 1 s b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done bad for s64
# run example for v5s_s64_x3: ./scripts/train_srarn_v5.sh resume 0,1 1 s b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup shen bad for s64
# run example for v5s_s64_x4: ./scripts/train_srarn_v5.sh resume 0 1 s b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup shen a100 bad for s64

# run example for v5s_Nrst_x2: ./scripts/train_srarn_v5.sh resume 0,1 1 s n 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5   # done t640 better for nearest interpolation add
# run example for v5s_Nrst_x4: ./scripts/train_srarn_v5.sh train 1 2 s n 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup shen 

# run example for v5t_x3: ./scripts/train_srarn_v5.sh train 0 1 t b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5t_x4: ./scripts/train_srarn_v5.sh train 0 1 t b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5t_x2: ./scripts/train_srarn_v5.sh train 0 1 t b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5xt_x2: ./scripts/train_srarn_v5.sh train 0 1 xt b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5xt_x3: ./scripts/train_srarn_v5.sh train 1 1 xt b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5xt_x4: ./scripts/train_srarn_v5.sh train 1 1 xt b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5

# run example for v5lt_x2: ./scripts/train_srarn_v5.sh train 1 1 lt b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5lt_x3: ./scripts/train_srarn_v5.sh train 1 1 lt b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5lt_x4: ./scripts/train_srarn_v5.sh train 1 1 lt b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5

# #####################################
# fixed layer number is 6 in each block versions
# run example for v5fblt_s64_x2: ./scripts/train_srarn_v5.sh train 0 1 fblt b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fblt_s64_x3: ./scripts/train_srarn_v5.sh train 1 1 fblt b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fblt_s64_x4: ./scripts/train_srarn_v5.sh train 0 1 fblt b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fbt_s64_x2: ./scripts/train_srarn_v5.sh train 1 1 fbt b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5fbt_s64_x3: ./scripts/train_srarn_v5.sh train 0 1 fbt b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fbt_s64_x4: ./scripts/train_srarn_v5.sh train 1 1 fbt b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fbxt_s64_x2: ./scripts/train_srarn_v5.sh train 1 1 fbxt b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# run example for v5fbxt_s64_x3: ./scripts/train_srarn_v5.sh train 0 1 fbxt b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# run example for v5fbxt_s64_x4: ./scripts/train_srarn_v5.sh train 1 1 fbxt b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen


# run example for v5fbxt_s64_Nrst_x2: ./scripts/train_srarn_v5.sh train 1 1 fbxt n 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # waiting t640 add following
# run example for v5fbxt_s64_x2: ./scripts/train_srarn_v5.sh train 1 1 fbxt b 2 64 ms 3acb3 DIV2K SmoothL1 SRARNV5  # done shen

# run example for v5fs_s64_x2: ./scripts/train_srarn_v5.sh train 0,1 1 fs b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup bad for s64
# run example for v5fs_s64_x3: ./scripts/train_srarn_v5.sh train 2,3 1 fs b 3 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup
# run example for v5fs_s64_x4: ./scripts/train_srarn_v5.sh train 0,1 1 fs b 4 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup

# #####################################
# fixed layer number is 6 in each block versions, with patch size 48
# run example for v5fbxt_x2: ./scripts/train_srarn_v5.sh train 1 1 fbxt b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done t640
# run example for v5fbxt_x3: ./scripts/train_srarn_v5.sh train 2 1 fbxt b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fbxt_x4: ./scripts/train_srarn_v5.sh train 1 1 fbxt b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fbt_x2: ./scripts/train_srarn_v5.sh train 0 1 fbt b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  #done t640
# run example for v5fbt_x3: ./scripts/train_srarn_v5.sh train 0 1 fbt b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fbt_x4: ./scripts/train_srarn_v5.sh train 0 1 fbt b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen

# run example for v5fs_x2: ./scripts/train_srarn_v5.sh train 0,1 1 fs b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup shen, bad for multi-GPU
# run example for v5fs_x3: ./scripts/train_srarn_v5.sh train 2,3 1 fs b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup bad for multi-GPU
# run example for v5fs_x4: ./scripts/train_srarn_v5.sh train 0,1 1 fs b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # giveup bad for multi-GPU
# run example for v5fs_Nrst_x2: ./scripts/train_srarn_v5.sh train 0 1 fs n 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fs_Nrst_x3: ./scripts/train_srarn_v5.sh train 1 1 fs n 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fs_Nrst_x4: ./scripts/train_srarn_v5.sh train 0 1 fs n 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen a100
# run example for v5fs_x2: ./scripts/train_srarn_v5.sh train 2 1 fs b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done            # test single GPU, may multi-GPU damage perf
# run example for v5fs_x3: ./scripts/train_srarn_v5.sh train 3 1 fs b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # down            # test single GPU  # init with multi-GPU resume with single-GPU
# run example for v5fs_x4: ./scripts/train_srarn_v5.sh train 0 1 fs b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # down shen a100  # test single GPU



# run example for v5fblt_x2: ./scripts/train_srarn_v5.sh train 3 1 fblt b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fblt_x3: ./scripts/train_srarn_v5.sh train 0 1 fblt b 3 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fblt_x4: ./scripts/train_srarn_v5.sh train 0 1 fblt b 4 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen

# #####################################
# ablation
# v5s30_x2: ./scripts/train_srarn_v5.sh train 0 1 s30 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s90_x2: ./scripts/train_srarn_v5.sh resume 0,1 1 s90 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s120_x2: ./scripts/train_srarn_v5.sh resume 1,2 1 s120 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# v5s150_x2: ./scripts/train_srarn_v5.sh resume 0,1 2 s150 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# v5s180_x2: ./scripts/train_srarn_v5.sh train 0 4 s180 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # pause
# v5s210_x2: ./scripts/train_srarn_v5.sh train 1 4 s210 b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # pause
# v5s1B_x2: ./scripts/train_srarn_v5.sh train 1 1 s1B b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s2B_x2: ./scripts/train_srarn_v5.sh train 0 1 s2B b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s6B_x2: ./scripts/train_srarn_v5.sh train 0,1 1 s6B b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s8B_x2: ./scripts/train_srarn_v5.sh resume 0,1 1 s8B b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen restart with epoch 200
# nohup python main.py --n_GPUs 2 --accumulation_step 1 --scale 2 --patch_size 128 --epochs 1300 --decay 550-1000-1150-1225 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6+6+6+6 --dims 60+60+60+60+60+60+60+60 --batch_size 32  --data_train DIV2K --data_range 1-900 --res_connect 1acb3 --loss 1*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class MultiStepLR --model SRARNV5 --pre_train ../srarn_v5/v5s8B_s64_SmoothL1_x2_epoch200/model/model_latest.pt --save ../srarn_v5/v5s8B_s64_SmoothL1_x2 --reset > ../srarn_v5/logs/v5s8B_s64_SmoothL1_x2.log 2>&1 &
# v5s10B_x2: ./scripts/train_srarn_v5.sh train 0 1 s10B b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # pause
# v5s1L_x2: ./scripts/train_srarn_v5.sh train 0 1 s1L b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s2L_x2: ./scripts/train_srarn_v5.sh train 0 1 s2L b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5
# v5s4L_x2: ./scripts/train_srarn_v5.sh resume 0 1 s4L b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# v5s8L_x2: ./scripts/train_srarn_v5.sh resume 2,3 1 s8L b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # done shen
# v5s10L_x2: ./scripts/train_srarn_v5.sh train 0 1 s10L b 2 64 ms 1acb3 DIV2K SmoothL1 SRARNV5  # pause


# run example for v5fbxtnopa_x2: ./scripts/train_srarn_v5.sh train 0 1 fbxtnopa b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fbxtps_x2: ./scripts/train_srarn_v5.sh train 1 1 fbxtps b 2 48 ms 1acb3 DIV2K SmoothL1 SRARNV5  # retraining shen
# run example for v5fbxt_skip_x2: ./scripts/train_srarn_v5.sh train 0 1 fbxt b 2 48 ms skip DIV2K SmoothL1 SRARNV5  # give up, no implemented skip res_connect, replaced with v9xt version

# no head; BN; before LN; for batch_befLN/v9fbxt_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 48 ms skip 1acb3 batch befln nolr  # giveup shen
# no head; no BN; before LN; for no_befLN/v9fbxt_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms skip 1acb3 no befln nolr  # giveup shen
# no head; no BN; No LN; for no_noLN/v9fbxt_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 48 ms skip 1acb3 no no nolr  # giveup
# no head; no BN; after LN; for no/v9fbxt_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms skip 1acb3 no ln nolr  # giveup
# 1acb3 head; BN; No LN; for batch_noLN/v9fbxt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 48 ms 1acb3 1acb3 batch no nolr  # retraining shen
# 1acb3 head; BN; after LN; for batch/v9fbxt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms 1acb3 1acb3 batch ln nolr  # retraining shen
# 1acb3 head; BN; No LN; for batch_noLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 3 1 fbxt b 2 64 ms 1acb3 1acb3 batch no nolr  # retraining shen
# 1acb3 head; BN; after LN; for batch/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 1acb3 1acb3 batch ln nolr  # done shen


# ################### test ######################
# 1acb3 head; BN; before LN; for batch_befLN/v9fbxt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms 1acb3 1acb3 batch befln nolr  # retraining shen
# 1acb3 head; BN; before LN; for batch_befLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 1acb3 1acb3 batch befln nolr  # retraining shen

# 1acb3 head; BN; No LN; for batch_noLN/v9fbxt_s64_1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 64 ms 1acb3 skip batch no nolr  # pause shen
# 1acb3 head; BN; after LN; for batch/v9fbxt_s64_1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 1acb3 skip batch ln nolr  # pause shen

# #####################################
# accept input
# input 2 params, first is run mode, 
mode=$1
# second is devices of gpu to use
device=$2
n_device=`expr ${#device} / 2 + 1`
# third is accumulation_step number
accum=$3
# forth is model size
size=$4
# ############## model_b #############
if [ $size = "b" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "bL" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 180 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "ba" ]; then  # model_b use PixelShuffle upsampling with activate layer, same as version 5
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "baL" ]; then  # model_b use PixelShuffle upsampling with activate layer, same as version 5
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --srarn_up_feat 180 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "bn" ]; then  # model_b with nearest+conv upsampling
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "bnL" ]; then  # model_b with nearest+conv upsampling
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --srarn_up_feat 180 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "LBbL" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 180 --depths 10+10+10+10 --dims 180+180+180+180 --batch_size 32"
# ############## model_l #############
elif [ $size = "l" ]; then  # model_l use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --batch_size 32"
elif [ $size = "tl" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 120 --depths 8+8+8+8+8+8+8+8+8+8 --dims 120+120+120+120+120+120+120+120 --batch_size 32"
# ############## model_s #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "fs" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --batch_size 32"
# ############## model_s ablation #############
elif [ $size = "s30" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 30 --depths 6+6+6+6 --dims 30+30+30+30 --batch_size 32"
elif [ $size = "s90" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 90 --depths 6+6+6+6 --dims 90+90+90+90 --batch_size 32"
elif [ $size = "s120" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 120 --depths 6+6+6+6 --dims 120+120+120+120 --batch_size 32"
elif [ $size = "s150" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 150 --depths 6+6+6+6 --dims 150+150+150+150 --batch_size 32"
elif [ $size = "s180" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 150 --depths 6+6+6+6 --dims 180+180+180+180 --batch_size 32"
elif [ $size = "s210" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 150 --depths 6+6+6+6 --dims 210+210+210+210 --batch_size 32"
elif [ $size = "s1B" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6 --dims 60 --batch_size 32"
elif [ $size = "s2B" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6 --dims 60+60 --batch_size 32"
elif [ $size = "s6B" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6+6 --dims 60+60+60+60+60+60 --batch_size 32"
elif [ $size = "s8B" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6+6+6+6 --dims 60+60+60+60+60+60+60+60 --batch_size 32"
elif [ $size = "s10B" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6+6+6+6+6+6 --dims 60+60+60+60+60+60+60+60+60+60 --batch_size 32"
elif [ $size = "s1L" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 1+1+1+1 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "s2L" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 2+2+2+2 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "s4L" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 4+4+4+4 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "s8L" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 8+8+8+8 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "s10L" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 10+10+10+10 --dims 60+60+60+60 --batch_size 32"
# ############## model_lt larger tiny #############
elif [ $size = "fblt" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --batch_size 32"
# ############## model_t #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --batch_size 32"
# ############## fixed block model_t #############
elif [ $size = "fbt" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 30 --depths 6+6+6 --dims 30+30+30 --batch_size 32"
# ############## model_xt #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --batch_size 32"
# ############## fixed block model_xt #############
elif [ $size = "fbxt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 6+6 --dims 24+24 --batch_size 32"
elif [ $size = "fbxtnopa" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling NearestNoPA --srarn_up_feat 24 --depths 6+6 --dims 24+24 --batch_size 32"
elif [ $size = "fbxtps" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling PixelShuffle --srarn_up_feat 24 --depths 6+6 --dims 24+24 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add bicubic interpolation of input
interpolation=$5
if [ $interpolation = "b" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "n" ]; then
  interpolation_print="_Nrst"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "s" ]; then
  interpolation_print="_Skip"
  interpolation="--interpolation Skip"
elif [ $interpolation = "p" ]; then
  interpolation_print="_PxSh"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (b | n | s)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
if [ $patch = 48 ]; then
  patch_print=""
else
  patch_print="_s$patch"
fi

# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
lr=$8
if [ $lr = "cosre" ]; then  # for CosineWarmRestart
  lr_class="CosineWarmRestart"
  lr_print="_CR"
elif [ $lr = "cos" ]; then  # for CosineWarm
  lr_class="CosineWarm"
  lr_print="_C"
else  # $lr = "ms"
  lr_class="MultiStepLR"
  lr_print=""
fi
# res_connect choice, other version default is 1acb3(same as version 5). 
res=$9
if [ $res = "1acb3" ]; then
  res_print=""
else
  res_print="_$res"
fi
# training dataset options --data_train --data_range 
dataset=${10}
if [ $dataset = "DIV2K" ]; then
  train="--data_train DIV2K --data_range 1-900"
  dataset_print=""
elif [ $dataset = "DF2K" ]; then
  train="--data_train DF2K --data_range 1-3550"
  dataset_print="_$dataset"
elif [ $dataset = "Flickr2K" ]; then
  train="--data_train Flickr2K --data_range 1-2650"
  dataset_print="_$dataset"
fi
# loss function
loss=${11}
if [ $loss = "L1" ]; then
  loss_print=""
else
  loss_print="_$loss"
fi
method=${12}  # SRARNV5 | SRARNV5OLD


# #####################################
# prepare program options parameters
# v5 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation $train --res_connect $res --loss 1*$loss --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class --model $method"
# seems SmoothL1 is better than L1

save_dir="../srarn_v5$dataset_print/v5${size}${patch_print}${interpolation_print}${res_print}${lr_print}${loss_print}_x${scale}"
log_file="../srarn_v5$dataset_print/logs/v5${size}${patch_print}${interpolation_print}${res_print}${lr_print}${loss_print}_x${scale}.log"

if [ ! -d "../srarn_v5$dataset_print" ]; then
  mkdir "../srarn_v5$dataset_print"
fi
if [ ! -d "../srarn_v5$dataset_print/logs" ]; then
  mkdir "../srarn_v5$dataset_print/logs"
fi


# #####################################
# run train/eval program
export CUDA_VISIBLE_DEVICES=$device
echo "CUDA GPUs use: No.'$CUDA_VISIBLE_DEVICES' devices."

if [ $mode = "train" ]
then
  if [ -f "$save_dir/model/model_latest.pt" ]; then
    echo "$save_dir seems storing some model files trained before, please change the save dir!"
  else
    echo "start training from the beginning:"
    echo "nohup $run_command --save $save_dir --reset > $log_file 2>&1 &"
    nohup $run_command --save $save_dir --reset > $log_file 2>&1 &
  fi
elif [ $mode = "resume" ]
then
  echo "resume training:"
  echo "nohup $run_command --load $save_dir --resume -1 > $log_file 2>&1 &"
  nohup $run_command --load $save_dir --resume -1 >> $log_file 2>&1 &
elif [ $mode = "switch" ]
then
  echo "switch acb from training to inference mode:"
  echo "$run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch"
  $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch
elif [ $mode = "eval" ]
then
  echo "load inference version of acb to eval:"
  echo "$run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
elif [ $mode = "eval_plus" ]
then
  echo "load inference version of acb to eval:"
  echo "$run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble"
  $run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble
elif [ $mode = "runtime" ]
then
  # echo "load inference version of acb to test the runtime:"
  # echo "$run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --times ${11}
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi

