#!/usr/bin/env sh
set -e
python extract_latents.py --config_path models/vqgan_imagenet_f16_1024/config.yaml --ckpt_path models/vqgan_imagenet_f16_1024/model.ckpt --data_root /data/vision/imagenet/ilsvrc_2011/pixels/train --save_path /data/vision/imagenet/ilsvrc_2011/latents/vqgan_imagenet_f16_1024_128px/train --image_size=128 --fp16
python extract_latents.py --config_path models/vqgan_imagenet_f16_1024/config.yaml --ckpt_path models/vqgan_imagenet_f16_1024/model.ckpt --data_root /data/vision/imagenet/ilsvrc_2011/pixels/val --save_path /data/vision/imagenet/ilsvrc_2011/latents/vqgan_imagenet_f16_1024_128px/val --image_size=128 --fp16
python extract_latents.py --config_path models/vqgan_imagenet_f16_1024/config.yaml --ckpt_path models/vqgan_imagenet_f16_1024/model.ckpt --data_root /data/vision/imagenet/ilsvrc_2011/pixels/test --save_path /data/vision/imagenet/ilsvrc_2011/latents/vqgan_imagenet_f16_1024_128px/test --image_size=128 --fp16
