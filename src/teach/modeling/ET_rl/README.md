# Multi-Layer Transformer + RL based EDH Baseline Model

The following instructions to train and evaluate an E.T. model on TEACh assume that you have the TEACh dataset downloaded. 
If running on a laptop, it might be desirable to mimic the folder structure of the TEACh dataset, but using only a small number of games from each split, and their corresponding images and EDH instances. 

Set some useful environment variables
```
source env_setup.source
```
Create a virtual environment (follow this or use Conda)

```buildoutcfg
python3 -m venv $VENV_DIR/teach_env
source $VENV_DIR/teach_env/bin/activate
cd TEACH_ROOT_DIR
pip install --upgrade pip 
pip install -r requirements.txt
export PYTHONPATH=$TEACH_SRC_DIR:$ET_ROOT:$PYTHONPATH
```

Download the ET pretrained checkpoint for Faster RCNN and Mask RCNN models
```buildoutcfg
wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
unzip et_checkpoints.zip
mv pretrained $ET_LOGS/
rm et_checkpoints.zip
```

Perform ET preprocessing (this extracts image features and does some processing of EDH jsons) (Do this, takes 2 hours. You can do it on the other directory ET directory)
```buildoutcfg
python -m alfred.data.create_lmdb \
    with args.visual_checkpoint=$ET_LOGS/pretrained/fasterrcnn_model.pth \
    args.data_input=edh_instances \
    args.task_type=edh \
    args.data_output=lmdb_edh \
    args.vocab_path=None
```
Note: If running on laptop on a small subset of the data, use `args.vocab_path=$ET_ROOT/files/human.vocab` and add `args.device=cpu`.


Train a model (adjust the `train.epochs` value in this command to specify the number of desired train epochs)
```buildoutcfg
python -m alfred.model.train_rl
```

# BELOW IS NOT SUPPORTED AT THIS MOMENT

Note: If running on laptop on a small subset of the data, add `exp.device=cpu` and `exp.num_workers=1`

Copy certain necessary files to the model folder so that we do not have to access training info at inference time.
```buildoutcfg
cp $ET_DATA/lmdb_edh/data.vocab $ET_LOGS/teach_et_trial
cp $ET_DATA/lmdb_edh/params.json $ET_LOGS/teach_et_trial
```

Evaluate the trained model
```buildoutcfg
cd $TEACH_ROOT_DIR
python src/teach/cli/inference.py \
    --model_module teach.inference.et_model \
    --model_class ETModel \
    --data_dir $ET_DATA \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_et_trial \
    --split valid_seen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_et_trial.json \
    --seed 4 \
    --model_dir teach_et_trial \
    --object_predictor $ET_LOGS/pretrained/maskrcnn_model.pth \
    --device cpu
```
