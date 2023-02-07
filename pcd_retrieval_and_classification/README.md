# Point cloud retrieval and classification

We performed our experiments on point clouds using the datasets `ModelNet40`, `ShapeNet10` and `ScanNet10`.
Please contact us to get access to our prepared datasets. 
In all our experiments we use `hesiod` to handle configs. Every time that you run a command, you will
be prompted with a UI asking to set paths and other needed configs.

## Create the dataset of INRs
The first step consists in creating the dataset of INRs. To do that you can run:
```
$ cd pcd_retrieval_and_classification 
$ python3 create_inrs_dataset.py
```

## Train `inr2vec`
After creating a dataset of INRs, you can train `inr2vec` on it by running:
```
$ python3 train_inr2vec.py
```
If you need to stop the training and restart it later, you can run:
```
$ python3 train_inr2vec.py /PATH/TO/INR2VEC/RUN.YAML
```
where `/PATH/TO/INR2VEC/RUN.YAML` provides the path to the config file created by `hesiod` for the
training of `inr2vec`.

## Visualization
Aftet training, you can visualize point clouds reconstructed by `inr2vec` running:
```
$ python3 viz.py /PATH/TO/INR2VEC/RUN.YAML
```
where `/PATH/TO/INR2VEC/RUN.YAML` provides the path to the config file created by `hesiod` for the
training of `inr2vec`.

## Export embeddings
After training `inr2vec`, its encoder can be used to embed the INRs into compact embeddings, by
running:
```
$ python3 export_embeddings.py
```

## Retrieval
To perform the retrieval experiment, simply run:
```
$ python3 retrieval.py /PATH/TO/INRS/EMBEDDINGS
```
where `/PATH/TO/INRS/EMBEDDINGS` indicates the path where the the INRs embeddings have been exported.

## Classification
To perform the classification experiment, simply run:
```
$ python3 train_classifier.py
```
