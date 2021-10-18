# MIMML
The implement of MIMML in paper: Accelerating Bioactive Peptides Discovery via Mutual Information based Meta-learning.

- **We provide the implements of  main experiments of our papers, which could run with the following commands.**
- **Users can execute the codes using the shell command or execute Python scripts directly. The latter one is recommended since it enable users to customize the hyper-parameters.**
- **User should download the project and go to the directory: '../MIMML/main/' and run script test to check if the code works properly.**



### dependency

The packages that the project depends on are listed in the file `requirements.txt`.

```shell
pandas==1.2.3
torch==1.7.1
tqdm==4.59.0
seaborn==0.11.1
numpy==1.19.5
learn2learn==0.1.5
torchmeta==1.6.1
pynvml==8.0.4
matplotlib==3.3.4
configuration==0.4.2
scikit_learn==1.0
```



## script test

To check whether the project can work normally, we can run 

shell command:

```shell
./run.sh script-test
```

or python script:

```shell
python run.py -task-type-run="script-test"
```



## pretrain

shell command:

```shell
./run.sh pretrain
```

python script:

```shell
python run.py -task-type-run="pretrain"
```



## meta train

shell command:

```shell
./run.sh meta-train
```

python script:

```shell
python run.py -task-type-run="meta-train"
```



## meta train with pretrained backbone

shell command:

```shell
./run.sh meta-train-with-pretrain
```

python script:

```shell
python run.py -task-type-run="meta-train-with-pretrain" -path-params='../result/default_pretrain/model/CNN, Epoch[20.000].pt'
```



## meta test

 shell command:

```shell
./run.sh meta-test
```

python script:

```shell
python run.py -task-type-run="meta-test" -path-params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt' -path-config='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl'
```



## meta finetune

 shell command:

```shell
./run.sh meta-finetune
```

python script:

```shell
python run.py -task-type-run="meta-finetune" -path-params="../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt"
```



## transductive inference

 shell command:

```shell
./run.sh meta-inference
```

python script:

```shell
python run.py -task-type-run="meta-inference" -path-params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt' -path-config='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl'
```



## Optional hyper-parameters

We can use run the following code to check almost optional hyper-parameters:

```shell
python run.py -h
```

Usage:

```shell
usage: run.py [-h] [-task-type-run TASK_TYPE_RUN] [-path-config PATH_CONFIG]
              [-learn-name LEARN_NAME] [-process-name PROCESS_NAME]
              [-save-best SAVE_BEST] [-cuda CUDA] [-device DEVICE]
              [-seed SEED] [-num_workers NUM_WORKERS]
              [-path-train-data PATH_TRAIN_DATA]
              [-path-test-data PATH_TEST_DATA]
              [-path-token2index PATH_TOKEN2INDEX]
              [-path-meta-dataset PATH_META_DATASET]
              [-path-params PATH_PARAMS] [-path-save PATH_SAVE]
              [-model-save-name MODEL_SAVE_NAME]
              [-save-figure-type SAVE_FIGURE_TYPE] [-num-class NUM_CLASS]
              [-dataset DATASET] [-max-len MAX_LEN]
              [-interval-log INTERVAL_LOG] [-interval-valid INTERVAL_VALID]
              [-interval-test INTERVAL_TEST]
              [-valid-start-epoch VALID_START_EPOCH]
              [-valid-interval VALID_INTERVAL] [-valid-draw VALID_DRAW]
              [-mode MODE] [-metric METRIC] [-threshold THRESHOLD]
              [-backbone BACKBONE] [-if-MIM IF_MIM]
              [-if-transductive IF_TRANSDUCTIVE]
              [-train-iteration TRAIN_ITERATION]
              [-valid-iteration VALID_ITERATION]
              [-test-iteration TEST_ITERATION]
              [-adapt-iteration ADAPT_ITERATION] [-adapt-lr ADAPT_LR]
              [-meta-batch-size META_BATCH_SIZE] [-train-way TRAIN_WAY]
              [-train-shot TRAIN_SHOT] [-train-query TRAIN_QUERY]
              [-valid-way VALID_WAY] [-valid-shot VALID_SHOT]
              [-valid-query VALID_QUERY] [-test-way TEST_WAY]
              [-test-shot TEST_SHOT] [-test-query TEST_QUERY] [-model MODEL]
              [-optimizer OPTIMIZER] [-loss-func LOSS_FUNC] [-epoch EPOCH]
              [-lr LR] [-reg REG] [-alpha ALPHA] [-lamb LAMB] [-temp TEMP]
              [-num-layer NUM_LAYER] [-num-head NUM_HEAD]
              [-dim-feedforward DIM_FEEDFORWARD] [-dim-k DIM_K] [-dim-v DIM_V]
              [-dim-embedding DIM_EMBEDDING] [-dropout DROPOUT]
              [-static STATIC] [-num-filter NUM_FILTER]
              [-filter-sizes FILTER_SIZES] [-dim-cnn-out DIM_CNN_OUT]
              [-output-extend OUTPUT_EXTEND]
```



## Example to customize the super parameters

We can use different combinations of hyper-parameters to train the model by specifying optional parameters. For example, we can set specific training epochs and learning rate:

```shell
python run.py -task-type-run="meta-train" -epoch=22 -lr=0.0001
```



For instance, if we run meta-train with customized hyper-parameters and the output path of the model parameters file is:

```
'../result/default_meta_train_with_pretrain/model/MIMML, Epoch[100.000].pt'
```

 and the corresponding config path is:

```
'../result/default_meta_train_with_pretrain/config.pkl'
```

 then we should  specify `-path-config` and `-path-params` according to the path of model parameters obtained by meta train to run meta test properly:

```shell
python run.py -task-type-run="meta-test" -path-config='../result/default_meta_train_with_pretrain/config.pkl' -path-params='../result/default_meta_train_with_pretrain/model/MIMML, Epoch[250.000].pt'
```



