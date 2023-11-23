#!/bin/bash

################################################################################
######################      SRARN V6       ######################
################################################################################
# ./train_srarn.sh [mode] [cuda_device] [accummulation_step] [model_size] [sr_scale] [lr_patch_size]
# run example as: ./train_srarn.sh train 0,1 1 test 2 48
# run example for v6b_x2: ./train_srarn.sh train 1,2,3 1 b 2 48
# run example for v6b_x3: ./train_srarn.sh train 0,1 2 b 3 48
# run example for v6b_x4: ./train_srarn.sh train 0,1 2 b 4 48
# run example for v6s_x2: ./train_srarn.sh train 0 1 s 2 48
# run example for v6s_x3: ./train_srarn.sh train 1 1 s 3 48
# run example for v6s_x4: ./train_srarn.sh train 0 1 s 4 48
# run example for v6t_x2: ./train_srarn.sh train 1 1 t 2 48
# run example for v6t_x3: ./train_srarn.sh train 0 1 t 3 48
# run example for v6t_x4: ./train_srarn.sh train 1 1 t 4 48
# run example for v6xt_x2: ./train_srarn.sh train 0 1 xt 2 48
# run example for v6xt_x3: ./train_srarn.sh train 1 1 xt 3 48
# run example for v6xt_x4: ./train_srarn.sh train 0 1 xt 4 48

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
if [ $size = "b" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "bn" ]; then  # model_b with nearest+conv upsampling
  options="--epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling Nearest --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --batch_size 32"
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --upsampling Nearest --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --batch_size 32"
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --res_connect 1acb3 --upsampling Nearest --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --batch_size 32"
elif [ $size = "test" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --res_connect 1acb3 --upsampling Nearest --srarn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# fifth is sr scale
scale=$5
# sixth is the LQ image patch size
patch=$6
patch_hr=`expr $patch \* $scale`


# #####################################
# prepare program options parameters
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options --loss 1*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --model SRARNV6"
save_dir="../srarn_v6/v6${size}_x${scale}"
log_file="../srarn_v6/logs/v6${size}_x${scale}.log"

if [ ! -d "../srarn_v6" ]; then
  mkdir "../srarn_v6"
fi
if [ ! -d "../srarn_v6/logs" ]; then
  mkdir "../srarn_v6/logs"
fi


# #####################################
# run train/eval program
export CUDA_VISIBLE_DEVICES=$device
echo "CUDA GPUs use: $CUDA_VISIBLE_DEVICES devices."

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
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi

