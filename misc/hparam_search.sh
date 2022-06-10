lr=".001 .0005"
batch_size="$((96*144)) $((96*144/8))"
hidden_size="256 128"
num_layers="5 3"
# echo $lr
# echo $batch_size
# echo $hidden_size
# echo $num_layers

# gpu=1
# for param in $lr
# do 
#     python run.py --lr $param --gpu $gpu
# done

# for param in $batch_size
# do 
#     python run.py --batch_size $param --gpu $gpu
# done

gpu=2
for param in $hidden_size
do 
    python run.py --hidden_size $param --gpu $gpu
done

for param in $num_layers
do 
    python run.py --num_layers $param --gpu $gpu
done