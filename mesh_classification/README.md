# Mesh classification

We performed our experiments on meshes using the `Manifold40` dataset.
Please contact us to get access to our prepared dataset. 
In all our experiments we use `hesiod` to handle configs. 
Every time that you run a command, you will be prompted with a UI asking to set paths and other
needed configs.

## Create the dataset of INRs
The first step consists in creating the dataset of INRs. To do that you can run:
```
$ cd mesh_classification 
$ python3 create_inrs_dataset.py
```

## (Optional) Compute SDF for an existing dataset of INRs
The first step (Create the dataset of INRs) saves the precomputed SDF for each shape
inside each INR h5 file. If for any reason you have a dataset of INRs without the precomputed
SDF, you can create a new version of the dataset containing also the precomputed SDF
by running:
```
$ python3 preprocess_sdf.py
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
Aftet training, you can visualize meshes reconstructed by `inr2vec` running:
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

## Classification
To perform the classification experiment, simply run:
```
$ python3 train_classifier.py
```
