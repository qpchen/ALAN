#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr 2e-4 bicubic 0 0       ######################
################################################################################
# ./scripts/train_srarn_v9.sh [mode] [cuda_device] [accummulation_step] [model] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [block_conv] [deep_conv] [acb_norm] [LN] [addLR]
# run example v9test_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 test b 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# ########### training commands ###########
# ###### on Starlight: ######
# run example v9ba_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 ba b 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # 38.243 v8  # PixelShuffleAct+BicubicAdd same as v5
# run example v9b_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 b b 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # 38.257 v8  # PixelShuffleNOAct+BicubicAdd

# run example v9bn_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 bn b 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # 38.266 v8: Nearest+BicubicAdd 
# run example v9bn_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0,1 1 bn b 3 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9bn_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1 1 bn b 4 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9bnL_Nrst_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 bnL n 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # 38.266 v8: Nearest+BicubicAdd 
# run example v9bL_Nrst_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 4 bL n 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # 38.257 v8  # PixelShuffleNOAct+BicubicAdd

# run example v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s b 2 64 ms skip 1acb3 inst befln nolr 2e-4 bicubic 0 0  # comparing
# run example v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s b 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s b 2 64 ms skip 1acb3 batch befln addlr 2e-4 bicubic 0 0  # comparing
# run example v9s_s64_Nrst_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s n 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0  # comparing

# run example v9s_s64_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 s b 3 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9s_s64_D1acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 s b 4 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0

# run example v9lt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 lt b 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9lt_s64_D3acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 lt b 3 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9lt_s64_D3acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 lt b 4 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0

# run example v9t_s64_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 t b 4 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9t_s64_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0 1 t b 3 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0

# ###### on t640: ######
# run example v9t_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 t b 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9xt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 xt b 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9xt_s64_D3acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 xt b 3 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# run example v9xt_s64_D3acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 xt b 4 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0

# run example v9xt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 xt n 2 64 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0

# run example v9xt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 xt b 2 64 ms skip 1acb3 batch befln addlr 2e-4 bicubic 0 0  # bad

# ################### test: 1acb3 head; BN; before LN;  ######################
# batch_befLN/v9l2_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 l2 b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # giveup shen a100

# batch_befLN/v9l3_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3 b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen  # larger batch multi-GPU
# batch_befLN/v9l3_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3 b 3 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # pause shen  # larger batch multi-GPU
# batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1,2,3 2 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # giveup shen # accum_step may bad  # larger batch multi-GPU
# batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 2 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 4e-4 bicubic 0 0  # giveup shen a100 # accum_step may bad  # larger lr
# batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1 1 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 4e-4 bicubic 0 0  # pause shen a100  # larger lr

# batch_befLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fbxt_s64_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fbxt_s64_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 2 1 fbxt b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fbxt_s64_3acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 3acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done t640  # param less than 100K
# batch_befLN/v9fbxt_s64_3acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 3 64 ms 3acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done t640
# batch_befLN/v9fbxt_s64_3acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 4 64 ms 3acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fblt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fblt b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fblt_s64_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 2 1 fblt b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fblt_s64_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 3 1 fblt b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fs_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fs b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fs_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 2 1 fs b 3 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen
# batch_befLN/v9fs_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 3 1 fs b 4 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # done shen

  # ############## test larger init LR ###############  !!! fs paused at epoch 1000 and good enough, l3 need further train
  # batch_befLN/v9fs_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fs b 2 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # done shen
  # batch_befLN/v9fs_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh resume 0 1 fs b 3 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # done shen no6
  # batch_befLN/v9fs_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh resume 1 1 fs b 4 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # done shen no6
  # batch_befLN/v9l3_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh resume 0,1,2,3 1 l3 b 2 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # giveup shen no2  # bad: too large lr multi-GPU
  # batch_befLN/v9l3_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh resume 0,1,2,3 1 l3 b 3 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # giveup shen no4  # bad: too large lr multi-GPU
  # batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh resume 0,1 1 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 16e-4 bicubic 0 0  # giveup shen a100  # bad: too large lr
  ###################################################
  # ########## change upsampling to PS && larger init LR #############
  # batch_befLN/v9l3_UpPS_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3_UpPS b 2 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # pause shen  # larger batch multi-GPU
  # batch_befLN/v9l3_UpPS_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3_UpPS b 3 48 ms 1acb3 1acb3 batch befln nolr 8e-4 bicubic 0 0  # pause shen  # larger batch multi-GPU
  # batch_befLN/v9l3_UpPS_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1 1 l3_UpPS b 4 48 ms 1acb3 1acb3 batch befln nolr 16e-4 bicubic 0 0  # pause shen a100  # larger lr
  ##################################################
  # ########## Image Restoration IR #############
    # BD, 
    # ./scripts/train_srarn_v9.sh train 1 1 fs b 3 48 ms 1acb3 1acb3 batch befln nolr 5e-4 BD 0 0
    # DN, 
    # ./scripts/train_srarn_v9.sh train 0 1 fs b 3 48 ms 1acb3 1acb3 batch befln nolr 5e-4 DN 0 0
    # Noise, 
    # ./scripts/train_srarn_v9.sh train 0 1 fs b 1 48 ms 1acb3 1acb3 batch befln nolr 5e-4 Noise 15 0
    # Gray_Noise
    # ./scripts/train_srarn_v9.sh train 1 1 fs b 1 48 ms 1acb3 1acb3 batch befln nolr 5e-4 Gray_Noise 15 0
    # Blur, 
    # ./scripts/train_srarn_v9.sh train 0 1 fs b 1 48 ms 1acb3 1acb3 batch befln nolr 5e-4 Blur 0 0
    # JPEG, 
    # ./scripts/train_srarn_v9.sh train 1 1 fs b 1 48 ms 1acb3 1acb3 batch befln nolr 5e-4 JPEG 0 10


# batch_befLN/v9fbxt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # pause shen  # bad p48
# batch_befLN/v9fblt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fblt b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0  # pause shen  # bad p48
# batch/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 64 ms 1acb3 1acb3 batch ln nolr 2e-4 bicubic 0 0  # done shen  # bad 37.899  p48:37.871
# batch_noLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 64 ms 1acb3 1acb3 batch no nolr 2e-4 bicubic 0 0  # done shen  # better 37.919 p48:37.903
# batch/v9fblt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fblt b 2 64 ms 1acb3 1acb3 batch ln nolr 2e-4 bicubic 0 0  # waiting shen

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
# ############## model_l #############
elif [ $size = "l1" ]; then  # model_l use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --batch_size 32"
elif [ $size = "l2" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --no_act_ps --srarn_up_feat 120 --depths 8+8+8+8+8+8+8+8 --dims 120+120+120+120+120+120+120+120 --batch_size 32"
elif [ $size = "l2_ps" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 120 --depths 8+8+8+8+8+8+8+8 --dims 120+120+120+120+120+120+120+120 --batch_size 32"
elif [ $size = "l3" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --no_act_ps --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --batch_size 32"
elif [ $size = "l3_UpPS" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --batch_size 32"
elif [ $size = "l4" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --no_act_ps --srarn_up_feat 180 --depths 8+8+8+8+8+8+8+8+8+8 --dims 180+180+180+180+180+180+180+180+180+180 --batch_size 32"
# ############## model_s #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "fs" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --batch_size 32"
# ############## fixed block model_lt larger tiny #############
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
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "b" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "bl" ]; then
  interpolation_print="_Biln"
  interpolation="--interpolation Bilinear"
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
  echo "no valid $interpolation ! Please input (b | bl | n | p | s)."
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
# if [ $# == 8 ]; then
  lr=$8
  if [ $lr = "cosre" ]; then  # CosineWarmRestart
    lr_class="CosineWarmRestart"
    lr_print="_CR"
  elif [ $lr = "cos" ]; then  # CosineWarm
    lr_class="CosineWarm"
    lr_print="_C"
  else  # $lr = "ms"
    lr_class="MultiStepLR"
    lr_print=""
  fi
# else
#   lr_class="MultiStepLR"
#   lr_print=""
# fi
# res_connect choice, other version default is 1acb3(same as version 5). 
# But v9 default 'skip'. other choices are 1conv1 3acb3
res=$9
if [ $res = "skip" ]; then
  res_print=""
else
  res_print="_$res"
fi
deep=${10}
# the last conv at end of deep feature module, before skip connect
if [ $deep = "skip" ]; then
  deep_print=""
else
  deep_print="_D$deep"
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_$acb"
# backbone norm choices
norm=${12}
if [ $norm = "ln" ]; then
  norm_opt="--norm_at after"
  norm_print=""
elif [ $norm = "no" ]; then  # "no": means do not use norm
  norm_opt="--no_layernorm"
  norm_print="_noLN"
elif [ $norm = "befln" ]; then
  norm_opt="--norm_at before"
  norm_print="_befLN"
fi
# add lr to upsampling, can be "addlr" or "nolr"
addlr=${13}
if [ $addlr = "addlr" ]; then
  addlr_opt="--add_lr"
  addlr_print="_addlr"
elif [ $addlr = "nolr" ]; then
  addlr_opt=""
  addlr_print=""
fi
initlr=${14}
if [ $initlr = "2e-4" ]; then
  initlr_print=""
else
  initlr_print="_$initlr"
fi
# degradation option bicubic, BD, DN, Noise, Blur, JPEG, Gray_Noise
deg=${15}
deg_opt="--degradation $deg"
deg_print="_$deg"
if [ $deg = "bicubic" ]; then
  deg_print=""
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "BD" ]; then 
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "DN" ]; then 
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "Noise" ]; then 
  val_set="McMaster"
  test_set="McMaster+Kodak24+CBSD68+Urban100"
elif [ $deg = "Gray_Noise" ]; then 
  deg_opt="$deg_opt --n_colors 1"
  val_set="Set12"
  test_set="Set12+BSD68+Urban100_Gray"
elif [ $deg = "Blur" ]; then 
  val_set="McMaster"
  test_set="McMaster+Kodak24+Urban100"
elif [ $deg = "JPEG" ]; then 
  val_set="Classic5"
  test_set="Classic5+LIVE1"
fi
# number of sigma for Noise degradation (15 | 25 | 50)
sigma=${16}
if [ $sigma = 0 ]; then
  sigma_print=""
else
  sigma_print="_N$sigma"
fi
# number of quality for JPEG degradation (10 | 20 | 30 | 40)
quality=${17}
if [ $quality = 0 ]; then
  quality_print=""
else
  quality_print="_Q$quality"
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --res_connect $res --deep_conv $deep --acb_norm $acb --loss 1*SmoothL1 --lr $initlr --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $norm_opt $addlr_opt --data_train DIV2K_IR --data_test $val_set $deg_opt --sigma $sigma --quality $quality --model SRARNV9"
# run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --res_connect $res --deep_conv $deep --loss 1*L1 --lr $initlr --optimizer ADAM --skip_threshold 1e6 --lr_class CosineWarmRestart --model SRARNV9"
save_dir="../srarn_v9${acb_print}${norm_print}${initlr_print}/v9${size}${patch_print}${addlr_print}${interpolation_print}${res_print}${deep_print}${lr_print}${deg_print}${sigma_print}${quality_print}_x${scale}"
log_file="../srarn_v9${acb_print}${norm_print}${initlr_print}/logs/v9${size}${patch_print}${addlr_print}${interpolation_print}${res_print}${deep_print}${lr_print}${deg_print}${sigma_print}${quality_print}_x${scale}.log"

if [ ! -d "../srarn_v9${acb_print}${norm_print}${initlr_print}" ]; then
  mkdir "../srarn_v9${acb_print}${norm_print}${initlr_print}"
fi
if [ ! -d "../srarn_v9${acb_print}${norm_print}${initlr_print}/logs" ]; then
  mkdir "../srarn_v9${acb_print}${norm_print}${initlr_print}/logs"
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
    echo "nohup python main.py $run_command --save $save_dir --reset > $log_file 2>&1 &"
    nohup python main.py $run_command --save $save_dir --reset > $log_file 2>&1 &
  fi
elif [ $mode = "resume" ]
then
  echo "resume training:"
  echo "nohup python main.py $run_command --load $save_dir --resume -1 > $log_file 2>&1 &"
  nohup python main.py $run_command --load $save_dir --resume -1 >> $log_file 2>&1 &
elif [ $mode = "switch" ]
then
  echo "switch acb from training to inference mode:"
  echo "python main.py $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch"
  python main.py $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch
elif [ $mode = "eval" ]
then
  echo "load inference version of acb to eval:"
  echo "python main.py $run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python main.py $run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
elif [ $mode = "eval_plus" ]
then
  echo "load inference version of acb to eval:"
  echo "python main.py $run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble"
  python main.py $run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble
elif [ $mode = "runtime" ]
then
  # echo "load inference version of acb to test the runtime:"
  # echo "python main.py $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python main.py $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --times ${11}
elif [ $mode = "lam" ]
then
  echo "doing Local attribution map (LAM) analysis:"
  echo "python lam.py $run_command --data_test Demo --dir_demo ../lams/imgs --save ${save_dir}_lam --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python lam.py $run_command --data_test Demo --dir_demo ../lams/imgs --save ${save_dir}_lam --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi

