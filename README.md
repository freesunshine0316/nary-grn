# *N*-ray Relation Extraction using Graph State LSTM

This repository corresponds to code for "[N-ary Relation Extraction using Graph State LSTM](https://arxiv.org/abs/1808.09101)", which has been accpeted by EMNLP 2018.

Subdirectories "bidir_dag_lstm" and "gs_lstm" contains our bidrectional DAG LSTM baseline and graph state LSTM (we recently rename it as graph recurrent network, GRN), respectively. 

## Important directories

### Bidir DAG LSTM

Our implementation of [Peng et al., (2017)](https://www.cs.jhu.edu/~npeng/papers/TACL_17_RelationExtraction.pdf), but with a main difference on how to utilize the edge labels. Section 3.3 of our paper describes the differences.
Our DAG LSTM is implemented based on tf.while_loop, thus it is highly effecient without redundency. 

### GS LSTM (GRN)

Our graph state lstm model

## How to run

Simply goes to the corresponding directory, and execute train.sh or decode.sh for training and evaluation, respectively. 
You may need to modify both scripts before executing.
In our experiment, we use
