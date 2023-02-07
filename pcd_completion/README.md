# Point cloud completion

We performed our point cloud completion experiment on the `MVP` dataset 
([https://mvp-dataset.github.io/MVP/Completion.html](https://mvp-dataset.github.io/MVP/Completion.html)).
Please contact us to get access to our prepared dataset. 
In all our experiments we use `hesiod` to handle configs. 
Every time that you run a command, you will be prompted with a UI asking to set paths and other
needed configs.

## Create the dataset of INRs
The first step consists in creating the dataset of INRs. To do that you can run:
```
$ cd pcd_completion
$ python3 create_inrs_dataset.py
```

## Train `inr2vec`
After creating the dataset of INRs, you can train `inr2vec` simultaneously on partial and complete
point clouds by running:
```
$ python3 train_inr2vec.py
```
If you need to stop the training and restart it later, you can run:
```
$ python3 train_inr2vec.py /PATH/TO/INR2VEC/RUN.YAML
```
where `/PATH/TO/INR2VEC/RUN.YAML` provides the path to the config file created by `hesiod` for the
training of `inr2vec`.

## Export embeddings
After training `inr2vec`, its encoder can be used to embed the INRs of both partial and complete
point clouds into compact embeddings, by running:
```
$ python3 export_embeddings.py
```

## Completion
To train the transfer network between the embeddings of partial point clouds and the embeddings of
the complete ones, simply run:
```
$ python3 train_completion.py
```

## Visualization
Finally, you can visualize the completion results by running:
```
$ python viz.py /PATH/TO/TRANSFER/RUN.YAML
```
where `/PATH/TO/TRANSFER/RUN.YAML` indicates the path to the config file created by `hesiod` for the
training of the transfer network.
