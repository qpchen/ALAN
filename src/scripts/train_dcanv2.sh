#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr 2e-4       ######################
################################################################################
# ./scripts/train_dcanv2.sh [mode] [cuda_device] [accummulation_step] [model_size] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [stage Res] [acb_norm] [upsampling]
# run example for v9test_D1acb3_x2: ./scripts/train_dcanv2.sh train 0 1 test nr 2 48 ms skip 1acb3 batch befln nolr 2e-4

#done No1: ./scripts/train_dcanv2.sh resume 1 1 t2 nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2

#done No1: ./scripts/train_dcanv2.sh eval 1 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No1: ./scripts/train_dcanv2.sh eval 1 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No2: ./scripts/train_dcanv2.sh lam 2 1 t2 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No2: ./scripts/train_dcanv2.sh eval 3 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No3: ./scripts/train_dcanv2.sh eval 0 1 xt2_dep4 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No3: ./scripts/train_dcanv2.sh eval 2 1 xt2_dep4 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#pause No8: ./scripts/train_dcanv2.sh train 1 1 b2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No4: ./scripts/train_dcanv2.sh resume 0 1 b26 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No4: ./scripts/train_dcanv2.sh resume 1 1 b26 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
#done No4: ./scripts/train_dcanv2.sh resume 2 1 b26 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2

# resume with L1 to try improve faster
#done No5: ./scripts/train_dcanv2.sh resume 1 1 b26 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no L1 DIV2K_IR V2
#done No5: ./scripts/train_dcanv2.sh resume 3 1 b26 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no L1 DIV2K_IR V2
#done No5: ./scripts/train_dcanv2.sh resume 2 1 b26 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no L1 DIV2K_IR V2

# Try better upsamling for large size model
#done No4: ./scripts/train_dcanv2.sh resume 3 1 b26 nr 2 48 ms 5e-4 useStageRes no PSnA noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2

  # Training Image Restoration IR (!! dataset only DIV2K caused overfit in 70-200 epoch!!)
    # Noise, 
    # training No1: ./scripts/train_dcanv2.sh train 5 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Noise 15 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 6 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Noise 25 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh resume 7 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Noise 50 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 0 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Noise 70 0 no L1 DIV2K_IR V2
    # Gray_Noise
    # training No1: ./scripts/train_dcanv2.sh train 2 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Gray_Noise 15 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 3 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Gray_Noise 25 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 4 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Gray_Noise 50 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 1 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Gray_Noise 70 0 no L1 DIV2K_IR V2
    # Blur, 
    # training No1: ./scripts/train_dcanv2.sh train 0 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN Blur 0 0 no L1 DIV2K_IR V2
    # JPEG, 
    # training No1: ./scripts/train_dcanv2.sh train 1 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN JPEG 0 10 no L1 DIV2K_IR V2
    # training No2: ./scripts/train_dcanv2.sh train 0 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN JPEG 0 20 no L1 DIV2K_IR V2
    # training No2: ./scripts/train_dcanv2.sh train 1 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN JPEG 0 30 no L1 DIV2K_IR V2
    # training No2: ./scripts/train_dcanv2.sh train 2 1 t2 nr 1 128 ms 2e-4 useStageRes no NN noACB 23 BN JPEG 0 40 no L1 DIV2K_IR V2
    # BD, 
    # ./scripts/train_dcanv2.sh train 3 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN BD 0 0 no L1 DIV2K_IR V2
    # DN, 
    # ./scripts/train_dcanv2.sh train 1 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 BN DN 0 0 no L1 DIV2K_IR V2


# Ablation
# 1. no Linear Attention
# done No2: ./scripts/train_dcanv2.sh resume 3 1 xt2_dep4_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
# 2. backbone norm: LN | no | noAll
# done No3: ./scripts/train_dcanv2.sh resume 0 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no SmoothL1 DIV2K_IR V2
##### error for save loss, so retrain use pre_training option: nohup python main.py --n_GPUs 1 --accumulation_step 1 --scale 2 --patch_size 96 --epochs 2000 --decay 500-1400-1700-1850 --srarn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 4+4 --batch_size 32 --interpolation Nearest --acb_norm no --stage_res --upsampling Nearest --loss 1*SmoothL1 --lr 5e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class MultiStepLR   --bb_norm LN  --data_train DIV2K_IR --data_range 1-900 --data_test Set5 --degradation bicubic --sigma 0 --quality 0 --model V2 --pre_train ../DCAN/v2_UpNN_noACB_ACBno_StgRes_AddNr_MS_5e-4/dcanv2_xt2_dep4_p48_bbLN_x2_backup/model/model_latest.pt --save ../DCAN/v2_UpNN_noACB_ACBno_StgRes_AddNr_MS_5e-4/dcanv2_xt2_dep4_p48_bbLN_x2 --reset > ../DCAN/v2_UpNN_noACB_ACBno_StgRes_AddNr_MS_5e-4/logs/dcanv2_xt2_dep4_p48_bbLN_x2.log 2>&1 &
# done No3: ./scripts/train_dcanv2.sh resume 1 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no SmoothL1 DIV2K_IR V2
# done No3: ./scripts/train_dcanv2.sh resume 2 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no SmoothL1 DIV2K_IR V2
# 3. MLP level change to 2
# done No3: ./scripts/train_dcanv2.sh resume 3 1 xt2_dep4_mlp2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2
# 4. L1 loss
# done No2: ./scripts/train_dcanv2.sh resume 0 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no L1 DIV2K_IR V2
# 5. local-then-global at block
# done No2: ./scripts/train_dcanv2.sh resume 2 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no SmoothL1 DIV2K_IR V2AA
# 

# try LN, no, or all noNorm
# LN !!!!!!!!!! maybe the best:
#done: ./scripts/train_dcanv2.sh train 0 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#done: ./scripts/train_dcanv2.sh train 1 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No3: ./scripts/train_dcanv2.sh train 1 1 t2 nr 2 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No3: ./scripts/train_dcanv2.sh train 0 1 t2 nr 3 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No3: ./scripts/train_dcanv2.sh train 2 1 t2 nr 4 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#done No3: ./scripts/train_dcanv2.sh eval 1 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#done No3: ./scripts/train_dcanv2.sh eval 2 1 xt2_dep4 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#done No3: ./scripts/train_dcanv2.sh eval 3 1 xt2_dep4 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#pause No1: ./scripts/train_dcanv2.sh resume 0 1 b26 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2   # giveup: retrain at 796(38.290 @763)
#done No1: ./scripts/train_dcanv2.sh resume 1 1 b26 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2   # giveup: retrain at 818(34.818 @566)
#pause No1: ./scripts/train_dcanv2.sh resume 2 1 b26 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2   # giveup: retrain at 760(32.804 @554)
#debug: ./scripts/train_dcanv2.sh train 5 1 b26 nr 8 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#giveup No1: ./scripts/train_dcanv2.sh train 0 1 n nr 2 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#giveup No1: ./scripts/train_dcanv2.sh train 1 1 n nr 3 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No1: ./scripts/train_dcanv2.sh train 0 1 b26 nr 2 48 ms 1e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No1: ./scripts/train_dcanv2.sh train 1 1 n nr 3 48 ms 1e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
#training No1: ./scripts/train_dcanv2.sh train 2 1 n nr 4 48 ms 2e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2

  # Training Image Restoration IR (!! dataset only DIV2K caused overfit in 70-200 epoch!!)
    # Noise, 
    # pause No1: ./scripts/train_dcanv2.sh train 5 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Noise 10 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 6 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Noise 30 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 5 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Noise 15 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 6 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Noise 25 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh resume 7 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Noise 50 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 7 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Noise 70 0 no L1 DIV2K_IR V2
    # Gray_Noise
    # pause No2: ./scripts/train_dcanv2.sh train 0 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Gray_Noise 10 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 3 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Gray_Noise 30 0 no L1 DIV2K_IR V2
    # waiting No4: ./scripts/train_dcanv2.sh train 0 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Gray_Noise 15 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 3 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Gray_Noise 25 0 no L1 DIV2K_IR V2
    # training No1: ./scripts/train_dcanv2.sh train 4 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Gray_Noise 50 0 no L1 DIV2K_IR V2
    # pause No1: ./scripts/train_dcanv2.sh train 5 1 b26 nr 1 48 ms 4e-7 useStageRes no NN noACB 23 LN Gray_Noise 70 0 no L1 DIV2K_IR V2
    # Blur, 
    # pause No2: ./scripts/train_dcanv2.sh resume 3 1 b26 nr 1 48 ms 1e-5 useStageRes no NN noACB 23 LN Blur 0 0 no L1 DIV2K_IR V2
    # waiting No2: ./scripts/train_dcanv2.sh resume 3 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN Blur 0 0 no L1 DIV2K_IR V2
    # JPEG, 
    # training No2: ./scripts/train_dcanv2.sh train 0 1 n nr 1 48 ms 4e-5 useStageRes no NN noACB 23 LN JPEG 0 10 no L1 DIV2K_IR V2
    # training No2: ./scripts/train_dcanv2.sh train 1 1 n nr 1 48 ms 4e-5 useStageRes no NN noACB 23 LN JPEG 0 20 no L1 DIV2K_IR V2
    # training No2: ./scripts/train_dcanv2.sh resume 2 1 n nr 1 48 ms 4e-5 useStageRes no NN noACB 23 LN JPEG 0 30 no L1 DIV2K_IR V2
    # waiting No2: ./scripts/train_dcanv2.sh train 0 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN JPEG 0 10 no L1 DIV2K_IR V2
    # waiting No2: ./scripts/train_dcanv2.sh train 1 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN JPEG 0 20 no L1 DIV2K_IR V2
    # waiting No2: ./scripts/train_dcanv2.sh resume 2 4 n nr 1 128 ms 2e-4 useStageRes no NN noACB 23 LN JPEG 0 30 no L1 DIV2K_IR V2
    # waiting No4: ./scripts/train_dcanv2.sh train 0 1 n nr 1 48 ms 4e-5 useStageRes no NN noACB 23 LN JPEG 0 40 no L1 DIV2K_IR V2
    # BD, 
    # ./scripts/train_dcanv2.sh train 3 1 b26 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 LN BD 0 0 no L1 DIV2K_IR V2
    # DN, 
    # ./scripts/train_dcanv2.sh train 1 1 b26 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 LN DN 0 0 no L1 DIV2K_IR V2


# blk no norm:
#training No2: ./scripts/train_dcanv2.sh train 0 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no L1 DIV2K_IR V2
#training No2: ./scripts/train_dcanv2.sh train 1 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no L1 DIV2K_IR V2
#training No2: ./scripts/train_dcanv2.sh train 2 1 t2 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no L1 DIV2K_IR V2
#training No2: ./scripts/train_dcanv2.sh train 3 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no L1 DIV2K_IR V2

# all noNorm (t2在200左右epoch梯度消失，xt2倒是没有，但也放弃继续训练):
#giveup No2: ./scripts/train_dcanv2.sh train 0 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V2
#giveup No2: ./scripts/train_dcanv2.sh train 1 1 t2 nr 3 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V2
#giveup No2: ./scripts/train_dcanv2.sh train 2 1 t2 nr 4 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V2
#giveup No2: ./scripts/train_dcanv2.sh train 3 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V2

# removed DWConv in FFN or add DWConv before DCN within Attention, with LN or noNorm

#training No3: ./scripts/train_dcanv2.sh train 0 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V3
#training No3: ./scripts/train_dcanv2.sh train 1 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V3
#training No3: ./scripts/train_dcanv2.sh train 2 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V4
#training No3: ./scripts/train_dcanv2.sh train 3 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V4
#training No1: ./scripts/train_dcanv2.sh train 2 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V5
#training No1: ./scripts/train_dcanv2.sh train 3 1 xt2_dep4 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V5


#training No5: ./scripts/train_dcanv2.sh train 1 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V3
#training No5: ./scripts/train_dcanv2.sh train 2 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V4
#training No5: ./scripts/train_dcanv2.sh train 3 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V5
#waiting No1: ./scripts/train_dcanv2.sh train 0 1 t2 nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V5

# try noAttn
# training No4: ./scripts/train_dcanv2.sh train 0 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 BN bicubic 0 0 no L1 DIV2K_IR V2
# training No4: ./scripts/train_dcanv2.sh train 2 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V2
# giveup No4: ./scripts/train_dcanv2.sh train 3 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V2
# giveup No5: ./scripts/train_dcanv2.sh train 0 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V3
# giveup No5: ./scripts/train_dcanv2.sh train 1 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V4
# giveup No5: ./scripts/train_dcanv2.sh train 2 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 noAll bicubic 0 0 no L1 DIV2K_IR V5

# training No4: ./scripts/train_dcanv2.sh train 1 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V3
# training No4: ./scripts/train_dcanv2.sh train 3 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V4
# training No5: ./scripts/train_dcanv2.sh train 0 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 LN bicubic 0 0 no L1 DIV2K_IR V5

# training No6: ./scripts/train_dcanv2.sh train 0 1 t2_noAttn nr 2 48 ms 5e-4 useStageRes no NN noACB 23 no bicubic 0 0 no L1 DIV2K_IR V5

# #####################################
# accept input
# first is run mode, 
mode=$1
# second is devices of gpu to use
device=$2
n_device=`expr ${#device} / 2 + 1`
# third is accumulation_step number
accum=$3
# forth is model size
size=$4
# ############## model_large #############
if [ $size = "l" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 180 --depths 8+8+8+8+8+8+8+8+8+8 --dims 180+180+180+180+180+180+180+180+180+180 --mlp_ratios 4+4+4+4+4+4+4+4+4+4 --batch_size 32"
# ############## model_base #############
elif [ $size = "b" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --mlp_ratios 4+4+4+4+4+4+4+4 --batch_size 32"
elif [ $size = "b2" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 128 --depths 6+6+6+6+6+6+6+6 --dims 128+128+128+128+128+128+128+128 --mlp_ratios 4+4+4+4+4+4+4+4 --batch_size 32"
elif [ $size = "b26" ]; then
  options="--epochs 1000 --decay 200-500-800-900-950 --srarn_up_feat 128 --depths 6+6+6+6+6+6 --dims 128+128+128+128+128+128 --mlp_ratios 4+4+4+4+4+4 --batch_size 32"
elif [ $size = "n" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 128 --depths 6+6+6+6+6+6 --dims 128+128+128+128+128+128 --mlp_ratios 4+4+4+4+4+4 --batch_size 32"
# elif [ $size = "b26" ]; then
#   options="--epochs 1200 --decay 350-700-1000-1100-1150 --srarn_up_feat 128 --depths 6+6+6+6+6+6 --dims 128+128+128+128+128+128 --mlp_ratios 4+4+4+4+4+4 --batch_size 32"
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --mlp_ratios 4+4+4+4+4 --batch_size 32"
elif [ $size = "xs" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --mlp_ratios 4+4+4+4 --batch_size 32"
elif [ $size = "xs2" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 64 --depths 6+6+6+6 --dims 64+64+64+64 --mlp_ratios 4+4+4+4 --batch_size 32"
elif [ $size = "s2" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 64 --depths 4+4+4 --dims 64+64+64 --mlp_ratios 4+4+4 --batch_size 32"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "t_mlp2" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 2+2+2 --batch_size 32"
elif [ $size = "t_noAttn" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32 --no_attn"
# elif [ $size = "t2" ]; then
#   options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 48 --depths 6+6+6 --dims 48+48+48 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "t2" ]; then
  options="--epochs 2000 --decay 800-1200-1600-1800-1900 --srarn_up_feat 48 --depths 6+6+6 --dims 48+48+48 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "t2_noAttn" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 48 --depths 6+6+6 --dims 48+48+48 --mlp_ratios 4+4+4 --batch_size 32 --no_attn"
# ############## model_xt extremely tiny #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32"
elif [ $size = "xt2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 32 --depths 5+5 --dims 32+32 --mlp_ratios 4+4 --batch_size 32"
# elif [ $size = "xt2_dep4" ]; then
#   options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 4+4 --batch_size 32"
elif [ $size = "xt2_dep4" ]; then
  options="--epochs 3000 --decay 1200-1800-2400-2600-2800-2900 --srarn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 4+4 --batch_size 32"
elif [ $size = "xt2_dep4_mlp2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 2+2 --batch_size 32"
elif [ $size = "xt2_dep4_noAttn" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 4+4 --batch_size 32 --no_attn"
elif [ $size = "xt2_mlp2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 32 --depths 5+5 --dims 32+32 --mlp_ratios 2+2 --batch_size 32"
elif [ $size = "xt_mlp2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 2+2 --batch_size 32"
elif [ $size = "xt_noAttn" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32 --no_attn"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 6 --depths 2+4 --dims 6+12 --mlp_ratios 4+4 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "bc" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "bl" ]; then
  interpolation_print="_AddBL"
  interpolation="--interpolation Bilinear"
elif [ $interpolation = "nr" ]; then
  interpolation_print="_AddNr"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "sk" ]; then
  interpolation_print="_AddSk"
  interpolation="--interpolation Skip"
elif [ $interpolation = "ps" ]; then
  interpolation_print="_AddPS"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (bc | bl | nr | ps | sk)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
patch_print="_p$patch"
# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
lr=$8
if [ $lr = "cosre" ]; then  # for CosineWarmRestart
  lr_class="CosineWarmRestart"
  lr_print="_CWRe"
elif [ $lr = "cos" ]; then  # for CosineWarm
  lr_class="CosineWarm"
  lr_print="_CW"
else  # $lr = "ms"
  lr_class="MultiStepLR"
  lr_print="_MS"
fi
initlr=$9
if [ $initlr = "2e-4" ]; then
  initlr_print=""
else
  initlr_print="_$initlr"
fi
# stage level residual connect
stageres=${10}
if [ $stageres = "useStageRes" ]; then
  stageres_opt="--stage_res"
  stageres_print="_StgRes"
elif [ $stageres = "noStageRes" ]; then  # better on test! better on other?
  stageres_opt=""
  stageres_print="_noStgRes"
else
  echo "no valid $stageres ! Please input (useStageRes | noStageRes)."
  exit
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_ACB$acb"
# upsampling optionsNearestNoPA
upsam=${12}
if [ $upsam = "NN" ]; then  # best? use Nearest-Neibor
  upsam_print="_UpNN"
  upsam_opt="Nearest"
elif [ $upsam = "PSnA" ]; then  # nA better? use PixelShuffle with no activate layer, same as SwinIR
  upsam_print="_UpPSnA"
  upsam_opt="PixelShuffle --no_act_ps"
elif [ $upsam = "PS" ]; then  # worst? use PixelShuffle with activate layer
  upsam_print="_UpPS"
  upsam_opt="PixelShuffle"
elif [ $upsam = "NNnPA" ]; then  # worse? use Nearest-Neibor without pixel attention
  upsam_print="_UpNNnPA"
  upsam_opt="NearestNoPA"
else
  echo "no valid $upsam ! Please input (NN | PS | PSnA | NNnPA)."
  exit
fi
# use ACB or not
use_acb=${13}
if [ $use_acb = "ACB" ]; then  # best? use Nearest-Neibor
  use_acb_print=""
  use_acb_opt="--use_acb"
elif [ $use_acb = "noACB" ]; then 
  use_acb_print="_noACB"
  use_acb_opt=""
elif [ $use_acb = "DBB" ]; then 
  use_acb_print="_DBB"
  use_acb_opt="--use_acb --use_dbb"
else
  echo "no valid $use_acb ! Please input (ACB | DBB | noACB)."
  exit
fi
# set large kernel (LKA) size
LKAk=${14}
if [ $LKAk = "23" ]; then  # default
  LKAk_print=""
  LKAk_opt=""
elif [ $LKAk = "7" ]; then 
  LKAk_print="_LK7"
  LKAk_opt="--DWDkSize 7 --DWDdil 1"
elif [ $LKAk = "15" ]; then 
  LKAk_print="_LK15"
  LKAk_opt="--DWDkSize 7 --DWDdil 2"
elif [ $LKAk = "31" ]; then 
  LKAk_print="_LK31"
  LKAk_opt="--DWDkSize 7 --DWDdil 4"
elif [ $LKAk = "39" ]; then 
  LKAk_print="_LK39"
  LKAk_opt="--DWDkSize 7 --DWDdil 5"
elif [ $LKAk = "47" ]; then 
  LKAk_print="_LK47"
  LKAk_opt="--DWDkSize 7 --DWDdil 6"
elif [ $LKAk = "55" ]; then 
  LKAk_print="_LK55"
  LKAk_opt="--DWDkSize 7 --DWDdil 7"
else
  echo "no valid $LKAk ! Please input (7 | 15 | 23 | 31 | 39 | 47 | 55)."
  exit
fi
# backbone norm use BN | LN | no, without '--no_layernorm' to keep LN in each stage
bb_norm=${15}
# bb_norm_opt="--bb_norm $bb_norm"
if [ $bb_norm = "BN" ]; then  # best? use Nearest-Neibor
  bb_norm_opt="--bb_norm BN"
  bb_norm_print=""
elif [ $bb_norm = "LN" ]; then 
  bb_norm_opt="--bb_norm LN"
  bb_norm_print="_bbLN"
elif [ $bb_norm = "no" ]; then 
  bb_norm_opt="--bb_norm no"
  bb_norm_print="_bbnoN"
elif [ $bb_norm = "noAll" ]; then 
  bb_norm_opt="--bb_norm no --no_layernorm"
  bb_norm_print="_allnoN"
else
  echo "no valid $bb_norm ! Please input (BN | LN | no | noAll)."
  exit
fi
# degradation option bicubic, BD, DN, Noise, Blur, JPEG, Gray_Noise
deg=${16}
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
sigma=${17}
if [ $sigma = 0 ]; then
  sigma_print=""
else
  sigma_print="_N$sigma"
fi
# number of quality for JPEG  (10 | 20 | 30 | 40)
quality=${18}
if [ $quality = 0 ]; then
  quality_print=""
else
  quality_print="_Q$quality"
fi
# use wavelet end-to-end, and up after (wave | wavePUp | no)
usewave=${19}
if [ $usewave = "wave" ]; then  # old code down first
  usewave_opt="--use_wave"
  usewave_print="_E2Ewave"
elif [ $usewave = "no" ]; then 
  usewave_opt=""
  usewave_print=""
elif [ $usewave = "wavePUp" ]; then
  usewave_opt="--use_wave --wave_patchup"
  usewave_print="_E2EwavePUp"
else
  echo "no valid $usewave ! Please input (wave | wavePUp | no)."
  exit
fi
# loss function 
loss=${20}
if [ $loss = "SmoothL1" ]; then
  loss_print=""
else
  loss_print="_$loss"
fi
# training dataset options --data_train --data_range 
dataset=${21}
if [ $dataset = "DIV2K_IR" ]; then
  train="--data_train DIV2K_IR --data_range 1-900"
  dataset_print=""
elif [ $dataset = "DF2K" ]; then
  train="--data_train DF2K --data_range 1-3550"
  dataset_print="_$dataset"
elif [ $dataset = "Flickr2K" ]; then
  train="--data_train Flickr2K --data_range 1-2650"
  dataset_print="_$dataset"
fi
# model options
model=${22}
if [ $model = "V2" ]; then
  model_print=""
  model_v="v2"
elif [ $model = "V2AA" ]; then
  model_print="_blkLTG"
  model_v="v2"
elif [ $model = "V3" ]; then
  model_print="_LFFN"  # light FFN, remove DWConv
  model_v="v3"
elif [ $model = "V4" ]; then
  model_print="_DWDA"  # add DWConv in DCN attention
  model_v="v4"
elif [ $model = "V5" ]; then
  model_print="_LFFN_DWDA"  # add DWConv in DCN attention
  model_v="v5"
else
  echo "no valid $model ! Please input (V2 | V2AA | V3 | V4 | V5)."
  exit
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --acb_norm $acb $stageres_opt --upsampling $upsam_opt --loss 1*$loss --lr $initlr --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $use_acb_opt $LKAk_opt $bb_norm_opt $usewave_opt $train --data_test $val_set $deg_opt --sigma $sigma --quality $quality --model DCAN$model"
father_dir="../DCAN/${model_v}${upsam_print}${use_acb_print}${acb_print}${stageres_print}${interpolation_print}${lr_print}${initlr_print}${dataset_print}"
file_name="dcan${model_v}_${size}${model_print}${patch_print}${LKAk_print}${bb_norm_print}${usewave_print}${deg_print}${sigma_print}${quality_print}${loss_print}_x${scale}"
save_dir="${father_dir}/${file_name}"
log_file="${father_dir}/logs/${file_name}.log"

if [ ! -d "../DCAN" ]; then
  mkdir "../DCAN"
fi
if [ ! -d "${father_dir}" ]; then
  mkdir "${father_dir}"
fi
if [ ! -d "${father_dir}/logs" ]; then
  mkdir "${father_dir}/logs"
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
elif [ $mode = "resumeBest" ]
then
  echo "resume training from best epoch (copy the model_best.pt to model_1.pt):"
  echo "nohup python main.py $run_command --load $save_dir --resume 1 > $log_file 2>&1 &"
  nohup python main.py $run_command --load $save_dir --resume 1 >> $log_file 2>&1 &
elif [ $mode = "retrainBest" ]
then
  echo "retraining from best epoch:"
  echo "nohup python main.py $run_command --save $save_dir --pre_train $save_dir/model/model_best.pt > $log_file 2>&1 &"
  nohup python main.py $run_command --save $save_dir --pre_train $save_dir/model/model_best.pt >> $log_file 2>&1 &
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

