#!/bin/bash

################################################################################
######################      SRARN V8       ######################
################################################################################
# ./scripts/train_srarn_v8.sh [mode] [cuda_device] [accummulation_step] [model] [use_bicubic] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [deep_conv] [acb_norm] [LN]
# run example for v8test_x2: ./scripts/train_srarn_v8.sh train 0 1 test nb 2 48 ms skip skip no ln
# ########### training commands ###########

# run example for v8ba_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 2 48 ms skip skip no ln
# run example for v8ba_x2: ./scripts/train_srarn_v8.sh train 2,3 2 ba ab 2 48 ms skip skip batch ln
# run example for v8ba_x3: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 3 48 ms skip skip no ln
# run example for v8ba_x4: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 4 48 ms skip skip no ln
# run example for v8ba_nb_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba nb 2 48 ms skip skip no ln

# run example for v8bn_x2: ./scripts/train_srarn_v8.sh train 0,1 1 bn ab 2 48 ms skip skip no ln
# run example for v8b_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b ab 2 48 ms skip skip no ln

# run example for v8s_x2: ./scripts/train_srarn_v8.sh train 0 1 s ab 2 48 ms skip skip no ln
# run example for v8s_x3: ./scripts/train_srarn_v8.sh train 1 1 s ab 3 48 ms skip skip no ln

# run example for v8s_B1acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 s ab 2 48 ms skip 1acb3 no ln  # VS v8s_x2, turns out B1acb3 is better
# run example for v8s_x4: ./scripts/train_srarn_v8.sh train 1 1 s ab 4 48 ms skip skip no ln  # paused
# run example for v8t_x4: ./scripts/train_srarn_v8.sh train 0 1 t ab 4 48 ms skip skip no ln
# run example for v8t_x3: ./scripts/train_srarn_v8.sh train 0 1 t ab 3 48 ms skip skip no ln

# run example for v8t_x2: ./scripts/train_srarn_v8.sh train 1 1 t ab 2 48 ms skip skip batch
# run example for v8t_x2: ./scripts/train_srarn_v8.sh train 0 1 t ab 2 48 ms skip skip no ln
# run example for v8xt_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms skip skip no ln
# run example for v8xt_x3: ./scripts/train_srarn_v8.sh train 1 1 xt ab 3 48 ms skip skip no ln
# run example for v8xt_x4: ./scripts/train_srarn_v8.sh train 1 1 xt ab 4 48 ms skip skip no ln


# run example for v8xt_1conv1_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 1conv1 no ln
# run example for v8xt_3acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 3acb3 no ln
# run example for v8xt_1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 1acb3 no ln

# run example for v8lt_x2: ./scripts/train_srarn_v8.sh train 0 1 lt ab 2 48 ms skip skip no ln
# run example for v8lt_x3: ./scripts/train_srarn_v8.sh train 1 1 lt ab 3 48 ms skip skip no ln
# run example for v8lt_x4: ./scripts/train_srarn_v8.sh train 1 1 lt ab 4 48 ms skip skip no ln


################################################################################
######################      SRARN V8_D1acb3 noACBnorm noLN       ######################
################################################################################
# ./scripts/train_srarn_v8.sh [mode] [cuda_device] [accummulation_step] [model] [use_bicubic] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [deep_conv] [acb_norm] [LN]
# run example for v8test_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 test nb 2 48 ms skip 1acb3 no no
# ########### training commands ###########
# on Starlight:
# run example for v8s_s64_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 s ab 2 64 ms skip 1acb3 no no
# run example for v8s_s64_D1acb3_x3: ./scripts/train_srarn_v8.sh train 1 1 s ab 3 64 ms skip 1acb3 no no

# run example for v8s_s64_D1acb3_x4: ./scripts/train_srarn_v8.sh train 1 1 s ab 4 64 ms skip 1acb3 no no
# run example for v8t_s64_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0 1 t ab 4 64 ms skip 1acb3 no no
# run example for v8t_s64_D1acb3_x3: ./scripts/train_srarn_v8.sh train 0 1 t ab 3 64 ms skip 1acb3 no no

# run example for v8ba_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 2 48 ms skip 1acb3 no no  # STOP!!!  # PixelShuffle+Bicubic fix PSNR, while Nearest+Bicubic is okay
# run example for v8ba_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba nb 2 48 ms skip 1acb3 no no  # STOP!!! # PixelShuffle+NOBicubicAdd PSNR lower than 35.709 during epoch 62-151
# run example for befLN/v8ba_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba nb 2 48 ms skip 1acb3 no befln  # comparing: # beforeLN!!! and PixelShuffleAct+NOBicubicAdd

# run example for v8bn_D1acb3_x2: ./scripts/train_srarn_v8.sh train 2,3 1 bn ab 2 48 ms skip 1acb3 no no  # comparing: Nearest+Bicubic

# run example for v8b_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b ab 2 48 ms skip 1acb3 no no  # STOP!!!  # PixelShuffle+Bicubic fix PSNR, while Nearest+Bicubic is okay
# run example for v8b_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b nb 2 48 ms skip 1acb3 no no  # STOP!!!: # PixelShuffle+NOBicubicAdd Again fix PSNR lower than 13 
# PixelShuffle with no activate is not suit for convnext when not LN at backbone?
# run example for befLN/v8b_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b nb 2 48 ms skip 1acb3 no befln  # comparing: # beforeLN!!! and PixelShuffle+NOBicubicAdd

# run example for v8ba_D1acb3_x3: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 3 48 ms skip 1acb3 no no  # STOP!!!  # NO TRAIN. but same reason
# run example for v8ba_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 4 48 ms skip 1acb3 no no  # STOP!!!  # PixelShuffle+Bicubic fix PSNR, while Nearest+Bicubic is okay
# run example for v8b_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0,1 1 b ab 4 48 ms skip 1acb3 no no  # STOP!!!  # NO TRAIN. but same reason
# run example for v8bn_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 bn nb 2 48 ms skip 1acb3 no no  # comparing: Nearest+NOBicubicAdd

# run example for v8s_s64_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 s nb 2 64 ms skip 1acb3 no no  # comparing: Nearest with no Bicubic adding
# run example for v8t_s64_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 t nb 2 64 ms skip 1acb3 no no  # comparing: Nearest with no Bicubic adding
# run example for v8xt_s64_nb_D3acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 xt nb 2 64 ms skip 3acb3 no no  # comparing: Nearest with no Bicubic adding

# on t640:
# run example for v8t_s64_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 t ab 2 64 ms skip 1acb3 no no
# run example for v8xt_s64_D3acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 64 ms skip 3acb3 no no
# run example for v8xt_s64_D3acb3_x3: ./scripts/train_srarn_v8.sh train 1 1 xt ab 3 64 ms skip 3acb3 no no
# run example for v8xt_s64_D3acb3_x4: ./scripts/train_srarn_v8.sh train 1 1 xt ab 4 64 ms skip 3acb3 no no
# run example for v8xt_s64_nb_D3acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 xt nb 2 64 ms skip 3acb3 no no  # comparing



# ########### may try commands ###########

# run example for v8xt_1conv1_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 1conv1 no no
# run example for v8xt_3acb3_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 3acb3 no no
# run example for v8xt_1acb3_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 48 ms 1acb3 no no

# run example for v8lt_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 lt ab 2 48 ms skip 1acb3 no no
# run example for v8lt_D1acb3_x3: ./scripts/train_srarn_v8.sh train 1 1 lt ab 3 48 ms skip 1acb3 no no
# run example for v8lt_D1acb3_x4: ./scripts/train_srarn_v8.sh train 1 1 lt ab 4 48 ms skip 1acb3 no no


################################################################################
######################      SRARN V8_D1acb3 noACBnorm befLN       ######################
################################################################################
# ./scripts/train_srarn_v8.sh [mode] [cuda_device] [accummulation_step] [model] [use_bicubic] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [deep_conv] [acb_norm] [LN]
# run example for v8test_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 test nb 2 48 ms skip 1acb3 no befln
# ########### training commands ###########
# on Starlight:
# run example for v8s_s64_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 s ab 2 64 ms skip 1acb3 no befln
# run example for v8s_s64_D1acb3_x3: ./scripts/train_srarn_v8.sh train 1 1 s ab 3 64 ms skip 1acb3 no befln

# run example for v8ba_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 2 48 ms skip 1acb3 no befln  # comparing  # PixelShuffleAct+BicubicAdd
# run example for v8ba_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 ba nb 2 48 ms skip 1acb3 no befln  # waiting # PixelShuffleAct+NOBicubicAdd 

# run example for v8bn_D1acb3_x2: ./scripts/train_srarn_v8.sh train 2,3 1 bn ab 2 48 ms skip 1acb3 no befln  # comparing: Nearest+BicubicAdd 

# run example for v8s_s64_D1acb3_x4: ./scripts/train_srarn_v8.sh train 1 1 s ab 4 64 ms skip 1acb3 no befln
# run example for v8t_s64_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0 1 t ab 4 64 ms skip 1acb3 no befln
# run example for v8t_s64_D1acb3_x3: ./scripts/train_srarn_v8.sh train 0 1 t ab 3 64 ms skip 1acb3 no befln

# run example for v8b_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b ab 2 48 ms skip 1acb3 no befln  # comparing  # PixelShuffleNOAct+Bicubic
# run example for v8b_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 b nb 2 48 ms skip 1acb3 no befln  # waiting # PixelShuffleNOAct+NOBicubicAdd

# run example for v8ba_D1acb3_x3: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 3 48 ms skip 1acb3 no befln  # waiting to training better version of x2
# run example for v8ba_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0,1 1 ba ab 4 48 ms skip 1acb3 no befln  # waiting to training better version of x2
# run example for v8b_D1acb3_x4: ./scripts/train_srarn_v8.sh train 0,1 1 b ab 4 48 ms skip 1acb3 no befln  # waiting to training better version of x2
# run example for v8bn_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0,1 1 bn nb 2 48 ms skip 1acb3 no befln  # comparing: Nearest+NOBicubicAdd

# run example for v8s_s64_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 s nb 2 64 ms skip 1acb3 no befln  # comparing: Nearest with no Bicubic adding
# run example for v8t_s64_nb_D1acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 t nb 2 64 ms skip 1acb3 no befln  # comparing: Nearest with no Bicubic adding
# run example for v8xt_s64_nb_D3acb3_x2: ./scripts/train_srarn_v8.sh train 1 1 xt nb 2 64 ms skip 3acb3 no befln  # comparing: Nearest with no Bicubic adding

# on t640:
# run example for v8t_s64_D1acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 t ab 2 64 ms skip 1acb3 no befln
# run example for v8xt_s64_D3acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt ab 2 64 ms skip 3acb3 no befln
# run example for v8xt_s64_D3acb3_x3: ./scripts/train_srarn_v8.sh train 1 1 xt ab 3 64 ms skip 3acb3 no befln
# run example for v8xt_s64_D3acb3_x4: ./scripts/train_srarn_v8.sh train 1 1 xt ab 4 64 ms skip 3acb3 no befln
# run example for v8xt_s64_nb_D3acb3_x2: ./scripts/train_srarn_v8.sh train 0 1 xt nb 2 64 ms skip 3acb3 no befln  # comparing