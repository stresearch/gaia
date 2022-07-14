
# batch_size=$((2*96*144))

gpu=2
# 2 8 
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 1
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 2
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 4
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 8
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 8
# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 8

python run_omega.py mode="train" trainer.params.gpu=[$gpu] ckpt lightning_logs/version_9

# python run.py --mode "train,val,test" --gpu 2 --model_type encoderdecoder --dataset spcam --bottleneck 512


# python run.py --mode "train,val,test" --gpu 2 --model_type encoderdecoder --dataset cam4 --bottleneck 16


# python run.py --mode "train,val,test" --gpu 0 --batch_size $batch_size



# "Q,T,U,V,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3"
# Q,T,SHFLX,OMEGA


# python run.py --mode "train,val,test" --gpu $gpu --model_type encoderdecoder --dataset spcam --bottleneck 8






# python run.py --ignore_input_variables OMEGA --memory_variables PRECT --gpu 1 &
# python run.py --ignore_input_variables OMEGA --memory_variables PRECC --gpu 2 &
# python run.py --ignore_input_variables OMEGA --memory_variables PTEQ --gpu 3 &
# python run.py --ignore_input_variables OMEGA --memory_variables PTTEND --gpu 4 &
# python run.py --ignore_input_variables OMEGA --gpu 5 &

# python run.py --model_type memory --gpu 2 --memory_variables PRECC --dataset spcam --mode "train,test" --batch_size $batch_size

# ckpt="lightning_logs/version_13"
# batch_size=$((2*96*144))
# thres="1e-13"

# python run.py --mode "results" --ckpt lightning_logs_compare_models/spcam_nn
# python run.py --mode "results" --ckpt lightning_logs_compare_models/cam4_nn


# python run.py --mode "train,val,test" --ckpt $ckpt --num_layers 28 --hidden_size 2048 --gpu 4 --mean_thres $thres --batch_size $batch_size --max_epochs 1000
# python run.py --mode "train,val,test" --num_layers 21 --hidden_size 1536 --gpu 3 --mean_thres $thres --batch_size $batch_size --max_epochs 1000
# python run.py --mode "train,val,test" --ckpt $ckpt --num_layers 28 --hidden_size 2048 --gpu 4 --mean_thres $thres --batch_size $batch_size --max_epochs 1000

# # python run.py $params --num_layers 7  --hidden_size 1024 --gpu 2 --mean_thres $thres --batch_size $batch_size &
# # python run.py $params --num_layers 14 --hidden_size 512  --gpu 3 --mean_thres $thres --batch_size $batch_size &



              


