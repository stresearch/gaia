# GAIA-surrogate

## Experiment Details

*https://stresearch.github.io/gaia/*

## Documentation [under construction]

Three groups of parameters:
1. trainer_params
2. dataset_params
3. model_params

```bash
python run_omega.py \  
mode='train,val,test' \  
trainer_params.gpus=[5]  
```


When you specify model_type, it sets
`model_params.model_config` to the corresponding dict in `Config.model_type_lookup()`


```bash
python run_omega.py \
mode='train,val,test'\
model_type='memory' \
trainer_params.gpus=[5] \
model_params.lr=1e-5 \
```


You can change `model_params.model_config` values individually 
after specifying a `model_type` (or leaving it as baseline)

```bash
python run_omega.py \
mode='train,val,test'\
model_type='memory' \
trainer_params.gpus=[5] \
model_params.lr=1e-5 \
model_params.model_config.dropout=0.02
```


dataset_params are changed indidually for each 'mode'

```python
python run_omega.py \
mode='train,val,test'\
model_type='memory' \
trainer_params.gpus=[5] \
dataset_params.train.shuffle=True
dataset_params.val.shuffle=False
```


You can add arguments that don't have defaults in Config
(min_epochs is a Trainer param)

```python
python run_omega.py \
mode='train,val,test' \
trainer_params.min_epochs=5
```