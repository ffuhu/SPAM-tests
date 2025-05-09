export LOCAL_RANK=0;
export MASTER_ADDR="localhost";
export MASTER_PORT=12355;
export RANK=0;
export WORLD_SIZE=1

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 50 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 101
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 50 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 101
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 75 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 101
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 75 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 101
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 100 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 151
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 100 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 151
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 150 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 175 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201


# second run - steps 150 and 175 with factors 10 and 5

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 150 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 150 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201
#
#
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 175 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 1000 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 175 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 201

## third run (try with NO warmup steps) and steps: 15 and 25 and factor 5 and 1
#
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 25 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 25 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50
#
#
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 15 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 15 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50

## with gaussian  --> NO BUMP, LOW FACTOR? FEW PARAMS AFFECTED?
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 15 \
#  --grad_injection_factor 50 \
#  --grad_injection_elements 0.25 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "gaussian" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50
#
## STEP fn at step 5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## with gaussian  --> NO BUMP, LOW FACTOR? FEW PARAMS AFFECTED? --> FACTOR->5000, PARAMS->1.0
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 15 \
#  --grad_injection_factor 5000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "gaussian" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50

## with gaussian  --> NO BUMP, LOW FACTOR? FEW PARAMS AFFECTED? --> FACTOR->500_000, PARAMS->1.0
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 15 \
#  --grad_injection_factor 500000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "gaussian" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 50

## STEP fn at step 5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration=0
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 0 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## tests at step 5 with durations 1 and 2 and factors 0.5 1 and 5
## STEP fn at step 5 duration=1
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=1
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=2
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=2
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=2
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 5 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## tests with duration 1 and factors 100 and 1000
## STEP fn at step 5 duration=1 factor 100
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=1 factor 1000
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=1 factor 10000
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 10000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
#
## tests with duration 2 and factors 100 and 1000
## STEP fn at step 5 duration=2 factor 100
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 100 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=2 factor 1000
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration=2 factor 10000
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 10000 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 1 factor 0.01
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 1 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 2 factor 0.01
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 2 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 0.01
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 1.0 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25


## STEP fn at step 5 duration 5 factor 0.01 params=0.5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 0.1 params=0.5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 1 params=0.5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 10 params=0.5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.01 params=0.75
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.75 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.1 params=0.75
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.75 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 1 params=0.75
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 0.75 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 10 params=0.75
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 10 \
#  --grad_injection_elements 0.75 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 0.10 params=0.75
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 0.75 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.10 params=0.5
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 0.10 params=0.4
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 0.4 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.10 params=0.3
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 0.3 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.10 params=0.2
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 0.2 \
#  --grad_injection_layer_number -1 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 1 params=1 layers=1-4
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 1 \
#  --grad_injection_elements 1 \
#  --grad_injection_layer_number 1 2 3 4 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.50 params=1 layers=1-4
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.50 \
#  --grad_injection_elements 1 \
#  --grad_injection_layer_number 1 2 3 4 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 0.10 params=1 layers=1-4
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.10 \
#  --grad_injection_elements 1 \
#  --grad_injection_layer_number 1 2 3 4 \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25


## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.5 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25


## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.4 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.4 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.3 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 0 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 5 \
#  --grad_injection_factor 0.1 \
#  --grad_injection_elements 0.3 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 25 \
#  --finish_execution_at 25

## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.3 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.3 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400


## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400

# 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.05 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.05 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400
#
# 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400

## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400

## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.02 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400

## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.02 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400


# ---

## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.03 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400
#
## STEP fn at step 5 duration 5 factor 1 params=1 layers=all
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.03 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400

## NO GRADIENT INJECTION - ADAMW
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700
#
## NO GRADIENT INJECTION - MUON
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700


#
# 130M
#


## NO GRADIENT INJECTION - ADAMW
#python torchrun_main1_test.py --model_config configs/llama_130m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 32 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700
#
## NO GRADIENT INJECTION - MUON
#python torchrun_main1_test.py --model_config configs/llama_130m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 32 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700

#python torchrun_main1_test.py --model_config configs/llama_130m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 32 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer adamwgradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74
#
#python torchrun_main1_test.py --model_config configs/llama_130m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 32 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 400 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74

# tests with GRADIENT NORM CLIPPING

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --exp_name gradnormscaling
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_norm_scaling \
#  --exp_name gradnormscaling
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.05 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_norm_scaling \
#  --exp_name gradnormscaling
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_norm_scaling \
#  --exp_name gradnormscaling

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.05 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --exp_name gradnormscaling
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --exp_name gradnormscaling


# tests with GRADIENT CLIPPING

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --exp_name adagradclipping

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.05 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --exp_name adagradclipping

#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --exp_name adagradclipping

## W AGN and AGC no INJ
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --grad_norm_scaling \
#  --exp_name agc_gns
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --grad_norm_scaling \
#  --exp_name agc_gns
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --grad_norm_scaling \
#  --grad_centering \
#  --exp_name gc_agc_gns
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_ada_clipping \
#  --grad_norm_scaling \
#  --grad_centering \
#  --exp_name gc_agc_gns


#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step -1 \
#  --grad_injection_factor -1 \
#  --grad_injection_elements -1 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "none" \
#  --grad_injection_duration -1 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_norm_scaling \
#  --grad_centering \
#  --exp_name gc_gns
#
#python torchrun_main1_test.py --model_config configs/llama_60m.json \
#  --lr 0.0005 \
#  --galore_scale 0.25 \
#  --rank 1.0 \
#  --update_proj_gap 200000 \
#  --batch_size 64 \
#  --total_batch_size 512 \
#  --num_training_steps 10000 \
#  --warmup_steps 150 \
#  --weight_decay 0 \
#  --dtype bfloat16 \
#  --eval_every 1000 \
#  --optimizer muongradientinjection \
#  --proj_type std \
#  --updating_mask_method random \
#  --model_type llama \
#  --data_type c4 \
#  --trick none \
#  --single_gpu \
#  --grad_injection_step 200 \
#  --grad_injection_factor 0.01 \
#  --grad_injection_elements 0.01 \
#  --grad_injection_layer_number -1  \
#  --grad_injection_fn "step" \
#  --grad_injection_duration 5 \
#  --save_gradients_every 50 \
#  --finish_execution_at 700 \
#  --grad_save_layers 0 19 20 21 22 23 24 25 26 27 74 \
#  --grad_norm_scaling \
#  --grad_centering \
#  --exp_name gc_gns