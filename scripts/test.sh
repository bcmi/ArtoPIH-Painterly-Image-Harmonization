#!/usr/bin/env bash
DISPLAY_PORT=8097


G='Reg_adain'
loadSize=256

is_matting=1
batchs=1
test_epoch=latest

#####network design

model_name=obadain
datasetmode=arto
content_dir="../datasets/painterly/MS-COCO/photographic_object/"
style_dir="../datasets/painterly/wikiart/"
info_dir="../examples"

NAME="pretrained"
checkpoint="../checkpoints/"


CMD="python ../test.py \
--dataset_root ./here/ \
--name $NAME \
--checkpoints_dir $checkpoint \
--model $model_name \
--netG $model_name \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--info_dir $info_dir \
--is_train 0 \
--display_id 0 \
--normD instance \
--normG instance \
--preprocess none \
--input_nc 3 \
--batch_size $batchs \
--num_threads 4 \
--print_freq 400 \
--display_freq 1 \
--gpu_ids 0 \
--load_size $loadSize \
--is_matting $is_matting \
--epoch $test_epoch  \
"

echo $CMD
eval $CMD
