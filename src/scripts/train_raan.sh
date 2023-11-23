#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr 2e-4       ######################
################################################################################
# ./scripts/train_srarn_v9.sh [mode] [cuda_device] [accummulation_step] [model_size] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [stage Res] [acb_norm] [upsampling]
# run example for v9test_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 test b 2 48 ms skip 1acb3 batch befln nolr 2e-4
# ########### test add bicubic ########### (may best on s/xs/xt??) _t: 38.069 327, 38.090 476, 38.108 711, 38.133 1612
# pause: ./scripts/train_raan.sh train 0,1 1 s bc 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# pause: ./scripts/train_raan.sh resume 1 1 t bc 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0  (Best: 38.133 @epoch 1594)
# pause: ./scripts/train_raan.sh resume 0 1 t bc 3 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# pause: ./scripts/train_raan.sh train 0 1 t bc 4 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# done t640: ./scripts/train_raan.sh resume 0 1 xt bc 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# training t640: ./scripts/train_raan.sh resume 1 1 xt bc 3 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

    # ################# ablation ##################
    # 1. no ACB, 2. ACB no, 3. StageRes, 4. 31x31, 5. replace ACB with DDB in backbone, 6. noACB + StageRes + 31, 7. DBB + StageRes + 31, 8. backbone no norm, 9. ACB no + backbone no/LN
    # 10. ACB no + backbone no/LN + stage noLN + no addLR
    # Hybrid version? noACB + StageRes? 
    # add unShufflePixel?

    # 1. done No5: ./scripts/train_raan.sh train 0 1 t bc 2 48 ms 4e-4 noStageRes batch NN noACB 23 BN bicubic 0 0  (Best: 38.168 @epoch 1614)
    # 2. done No5: ./scripts/train_raan.sh train 1 1 t bc 2 48 ms 4e-4 noStageRes no NN ACB 23 BN bicubic 0 0  (Best: 38.165 @epoch 1952)
    # 3. training No3: ./scripts/train_raan.sh train 0 1 t bc 2 48 ms 4e-4 useStageRes batch NN ACB 23 BN bicubic 0 0  (Best: 38.164 @epoch 1717)
    # 4. training No1: ./scripts/train_raan.sh train 0 1 t bc 2 48 ms 4e-4 noStageRes batch NN ACB 31 BN bicubic 0 0  (Best: 38.150 @epoch 1208)
    # 5. training No1: ./scripts/train_raan.sh train 1,3 1 t bc 2 48 ms 4e-4 noStageRes batch NN DBB 23 BN bicubic 0 0  (Best: 38.173 @epoch 1117)
    # 6. done No1: ./scripts/train_raan.sh train 2 1 t bc 2 48 ms 4e-4 useStageRes batch NN noACB 31 BN  (Best: 38.129 @epoch 1302)
    # 7. pause: ./scripts/train_raan.sh train 3 1 t bc 2 48 ms 4e-4 useStageRes batch NN DBB 31 BN bicubic 0 0
    # 8. giveup: ./scripts/train_raan.sh train 0 1 t bc 2 48 ms 4e-4 noStageRes batch NN ACB 23 LN bicubic 0 0  # giveup, too slow, don't know how to code for speed up
    # 8. training No4: ./scripts/train_raan.sh train 1 1 t bc 2 48 ms 4e-4 noStageRes batch NN ACB 23 no bicubic 0 0  # gradient explose, so keep LN in each stage try again  (Best: 38.162 @epoch 1179)
    # 9. giveup: ./scripts/train_raan.sh train 1 1 t bc 2 48 ms 4e-4 noStageRes no NN ACB 23 no bicubic 0 0  # gradient explose at epoch 400+
    # 10. giveup No4: ./scripts/train_raan.sh train 0 1 t sk 2 48 ms 4e-4 noStageRes no NN ACB 23 no bicubic 0 0  # gradient explose at 850 epoch  (Best: 38.057 @epoch 786)
    # 10. giveup: ./scripts/train_raan.sh train 1 1 t sk 2 48 ms 4e-4 noStageRes no NN ACB 23 noAll bicubic 0 0  # gradient explose
    # 10. giveup T640: ./scripts/train_raan.sh train 0 1 t bc 2 48 ms 4e-4 noStageRes no NN ACB 23 noAll bicubic 0 0  # gradient explose  # try on t640 to test if starlight cause the gradient problem


    
    # ################# Image Restoration IR ##################
    # BD, 
    # DN, 
    # Noise, 
    # Blur, 
    # JPEG, 
    # Gray_Noise

# ##### test add bilinear ######## bad _t: 38.069 327, 38.082 476, 38.095 711, 38.118 965
# done: ./scripts/train_raan.sh resume 2 1 t bl 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# ##### test add nearest ######## best _t: 38.092 327, 38.105 476, 38.108 711, 38.140 1000
# done: ./scripts/train_raan.sh train 0,1 1 s nr 2 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done a100: ./scripts/train_raan.sh train 0 1 s nr 3 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# pause a100: ./scripts/train_raan.sh train 1 1 s nr 4 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# done: ./scripts/train_raan.sh train 0,1 1 xs nr 2 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh train 0,1 1 xs nr 3 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh train 0,1 1 xs nr 4 48 ms 8e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# done: ./scripts/train_raan.sh resume 0 1 t nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh resume 0 1 t nr 3 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh train 0 1 t nr 4 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# done: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh train 0 1 xt nr 3 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# done: ./scripts/train_raan.sh train 1 1 xt nr 4 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

  # ####ablation staty: 1. 把ACB替换回Conv和ACB用instN及不用bn；2. LKA使用不同核；3. stage使用residual connect；4. MLP ratio；5. LR 插值用不同方法；6. 上采样不同方法；7. stage、block等数量设置；8. LKA不使用attention
  # 1 done: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch NN noACB 23 BN bicubic 0 0
  # 1 done: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes inst NN ACB 23 BN bicubic 0 0
  # 1 done: ./scripts/train_raan.sh train 2 1 xt nr 2 48 ms 4e-4 noStageRes no NN ACB 23 BN bicubic 0 0
  # 2 done: ./scripts/train_raan.sh train 3 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 7 BN bicubic 0 0
  # 2 done: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 14 BN bicubic 0 0
  # 2 done: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 28 BN bicubic 0 0
  # 2 pause: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 39 BN bicubic 0 0
  # 2 pause: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 47 BN bicubic 0 0
  # 2 pause: ./scripts/train_raan.sh train 3 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 55 BN bicubic 0 0
  # 3 pause: ./scripts/train_raan.sh train 2 1 xt nr 2 48 ms 4e-4 useStageRes batch NN ACB 23 BN bicubic 0 0
  # 4 done: ./scripts/train_raan.sh train 3 1 xt_mlp2 nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
  # 6 done: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch PSnA ACB 23 BN bicubic 0 0
  # 6 done: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NNnPA ACB 23 BN bicubic 0 0
  # 8 done t640: ./scripts/train_raan.sh train 0 1 xt_noAttn nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
  # 8 done t640: ./scripts/train_raan.sh train 1 1 t_noAttn nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0


  # 1 done: ./scripts/train_raan.sh train 1 1 t nr 2 48 ms 4e-4 noStageRes batch NN noACB 23 BN bicubic 0 0
  # 1 pause: ./scripts/train_raan.sh train 2 1 t nr 2 48 ms 4e-4 noStageRes no NN ACB 23 BN bicubic 0 0

  
    # ############### optimize #################
    # 1. ACB no; 2. replace ACB with DDB in backbone; 3. StageRes; 4. backbone no norm; 5. no ACB; 6. StageRes + backbone no norm; 7. StageRes + backbone no norm + ACB no
    # 8. DDB + StageRes + backbone no norm + ACB no(upsampling); 9. DDB + StageRes + ACB no(upsampling)
    # ./scripts/train_raan.sh resume 0 1 t nr 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
    # 1. done: ./scripts/train_raan.sh train 2 1 t nr 2 48 ms 5e-4 noStageRes no NN ACB 23 BN bicubic 0 0
    # 2. pause No1-A: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 noStageRes batch NN DBB 23 BN bicubic 0 0
    # 3. done: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 useStageRes batch NN ACB 23 BN bicubic 0 0
    # 4. done: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 noStageRes batch NN ACB 23 no bicubic 0 0
    # 5. done: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 noStageRes batch NN noACB 23 BN bicubic 0 0
    # 6. done: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 useStageRes batch NN ACB 23 no bicubic 0 0
    # 7. done: ./scripts/train_raan.sh train 1 1 t nr 2 48 ms 5e-4 useStageRes no NN ACB 23 no bicubic 0 0
    # 8. pause No2-A: ./scripts/train_raan.sh train 0 1 t nr 2 48 ms 5e-4 useStageRes no NN DBB 23 no bicubic 0 0  (Best: 38.016 @epoch 152)
    # 9. pause No2-A: ./scripts/train_raan.sh train 1 1 t nr 2 48 ms 5e-4 useStageRes no NN DBB 23 BN bicubic 0 0  (Best: 38.067 @epoch 137)



# ##### test add pixel shuffle ######## _t: 38.089 327, 38.103 476, 38.127 711, 38.133 945
# done: ./scripts/train_raan.sh train 3 1 t ps 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# waiting: ./scripts/train_raan.sh train 0 1 xt ps 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0

# ##### test small mlp_ratios ########
# waiting: ./scripts/train_raan.sh train 3 1 t_mlp2 bl 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0
# waiting: ./scripts/train_raan.sh train 0 1 xt_mlp2 bl 2 48 ms 4e-4 noStageRes batch NN ACB 23 BN bicubic 0 0


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
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --mlp_ratios 4+4+4+4+4 --batch_size 32"
elif [ $size = "xs" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --mlp_ratios 4+4+4+4 --batch_size 32"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "t_mlp2" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 2+2+2 --batch_size 32"
elif [ $size = "t_noAttn" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32 --no_attn"
# ############## model_xt extremely tiny #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32"
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
# number of sigma for Noise degradation (15 | 25 | 50)
quality=${18}
if [ $quality = 0 ]; then
  quality_print=""
else
  quality_print="_Q$quality"
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --acb_norm $acb $stageres_opt --upsampling $upsam_opt --loss 1*SmoothL1 --lr $initlr --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $use_acb_opt $LKAk_opt $bb_norm_opt --data_train DIV2K_IR --data_test $val_set $deg_opt --sigma $sigma --quality $quality --model RAAN"
father_dir="../raan${upsam_print}${use_acb_print}${acb_print}${stageres_print}${interpolation_print}${lr_print}${initlr_print}"
file_name="v1${size}${patch_print}${LKAk_print}${bb_norm_print}${deg_print}${sigma_print}${quality_print}_x${scale}"
save_dir="${father_dir}/${file_name}"
log_file="${father_dir}/logs/${file_name}.log"

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
  echo "$run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
elif [ $mode = "eval_plus" ]
then
  echo "load inference version of acb to eval:"
  echo "$run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble"
  $run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble
elif [ $mode = "runtime" ]
then
  # echo "load inference version of acb to test the runtime:"
  # echo "$run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --times ${11}
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi

