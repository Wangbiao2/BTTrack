#!/bin/bash

# GOT-10k (Web Eval:http://got-10k.aitestunion.com/)
python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset got10k_test --threads 2 --num_gpus 1 --runid 300
python lib/test/utils/transform_got10k.py --tracker_name bttrack --cfg_name vit_tiny_ep300_300

## TrackingNet (Web Eval: https://eval.ai/web/challenges/challenge-page/1805/leaderboard)
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset trackingnet --threads 2 --num_gpus 1 --runid 300
#python lib/test/utils/transform_trackingnet.py --tracker_name FERMT --cfg_name vit_base_ep300_tokentype_300
#
## LaSOT
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset lasot --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and datset name
#
## LaSOT_ext
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset lasot_ext --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## UAV123
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset uav --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## UAV20L
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset uav20l --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## UAV10fps
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset uav10fps --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## DTB70
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset dtb --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## UAVDT
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset uavdt --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## UAVTrack112
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset uavtrack112 --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name
#
## VisDrone
#python tracking/test.py --tracker_name bttrack --tracker_param vit_tiny_ep300 --dataset visdrone --threads 2 --num_gpus 1 --runid 300
#python tracking/analysis_results.py # need to modify tracker configs and dataset name