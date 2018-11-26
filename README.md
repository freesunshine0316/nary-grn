# *N*-ray Relation Extraction using Graph State LSTM

This repository corresponds to code for "[N-ary Relation Extraction using Graph State LSTM](https://arxiv.org/abs/1808.09101)", which has been accpeted by EMNLP 2018.

Subdirectories "bidir_dag_lstm" and "gs_lstm" contains our bidrectional DAG LSTM baseline and graph state LSTM (we recently rename it as graph recurrent network, GRN), respectively. 

## Important directories

### Bidir DAG LSTM

Our implementation of [Peng et al., (2017)](https://www.cs.jhu.edu/~npeng/papers/TACL_17_RelationExtraction.pdf), but with a main difference on how to utilize the edge labels. Section 3.3 of our paper describes the differences.
Our DAG LSTM is implemented based on tf.while_loop, thus it is highly effecient without redundency. 

### GS LSTM (GRN)

Our graph-state LSTM model.

## How to run

Simply goes to the corresponding directory, and execute train.sh or decode.sh for training and evaluation, respectively. 
You may need to modify both scripts before executing. The hyperparameters and other settings are in config.json.

We used 5-fold cross validation to conduct our experiment. If your dataset has a training/dev/test separation, just ignore the words below.
To make things a little bit easier, we use file-of-file, where the first-level files store the locations of the data. One example is "train_list_0" and "test_list_0" in [./gs_lstm/data](./gs_lstm/data), where each line points to a file address. Our data has been segmented into 5 folds by Peng et al., thus we simply follow it.
You need to modify both "train_list_0" and "test_list_0" and make the rest, such as "train_list_1" and "test_list_1"

Other scripts within [./gs_lstm/data](./gs_lstm/data) is for extracting pretrained word embeddings. We use Glove-100d pretrained embeddings. 

For more questions, please create a issue and I'll handle it as soon as possible.

## Data

We put the data by Peng et al., (2017) inside [this repository](./peng_data) for easy access for others.

## Cite

Please cite this bib:
```
@article{song2018n,
  title={N-ary relation extraction using graph state LSTM},
  author={Song, Linfeng and Zhang, Yue and Wang, Zhiguo and Gildea, Daniel},
  journal={arXiv preprint arXiv:1808.09101},
  year={2018}
}
```

