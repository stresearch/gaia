# "Q,T,U,V,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3"
# Q,T,SHFLX,OMEGA


python run.py
# python run.py --input_var_ignore T
# python run.py --input_var_ignore SHFLX
# python run.py --input_var_ignore OMEGA
#22 23 24
# for v in 22 23 24
# do
#     python run.py --mode predict,results --ckpt lightning_logs/version_$v --gpu 4 --dataset /ssddg1/gaia/cam4/cam4-famip-30m-timestep_4
# done