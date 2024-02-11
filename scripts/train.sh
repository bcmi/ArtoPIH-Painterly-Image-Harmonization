#!/usr/bin/env bash
DISPLAY_PORT=8097


G='balance'
loadSize=256
fineSize=256

L_C=1
L_S=1
L_R=100
L_CLS=100
L_TV=1e-6
L_mask=1

is_matting=1
batchs=8
lr=1e-4
load_iter=0


model_name=obadain
datasetmode=arto
content_dir="../datasets/painterly/MS-COCO/photographic_object/"
style_dir="../datasets/painterly/wikiart/"
info_dir="../datasets/painterly/wikiart/WikiArt_Split/similar_objects_train_released"

NAME="${model_name}_rec${L_R}_C${L_C}_S${L_S}_CLS${L_CLS}_batch${batchs}_lr${lr}"
checkpoint="../checkpoints/"

# OTHER="--continue_train"
CMD="python ../train.py \
--dataset_root ./here/ \

--name $NAME \
--checkpoints_dir $checkpoint \
--model $model_name \
--netG $model_name \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--info_dir $info_dir \
--is_train 1 \
--display_id 0 \
--normG instance \
--preprocess none \
--niter 50 \
--niter_decay 50 \
--input_nc 3 \
--batch_size $batchs \
--num_threads 8 \
--print_freq 200 \
--display_freq 200 \
--save_epoch_freq 1 \
--save_latest_freq 100 \
--gpu_ids 2 \
--lr $lr \
--is_matting $is_matting \
--lambda_c $L_C \
--lambda_s $L_S \
--lambda_class $L_CLS \
--lambda_rec $L_R \
--lambda_tv $L_TV \
--load_iter $load_iter \


"
echo $CMD
eval $CMD
