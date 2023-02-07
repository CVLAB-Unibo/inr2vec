# Point cloud generation

We used the point clouds from the `ShapeNet10` dataset to perform the experiment of unconditioned
generation of point clouds.
Please contact us to get access to our prepared dataset. 
In all our experiments we use `hesiod` to handle configs. 
Every time that you run a command, you will be prompted with a UI asking to set paths and other
needed configs.

## Create the dataset of INRs, train `inr2vec` and export embeddings
Please use the instructions and the code in the directory `pcd_retrieval_and_classification` to
perform these initial steps on `ShapeNet10`.

## Export embeddings for Latent-GAN
To train a Latent-GAN on `inr2vec` embeddings, first export the embeddings in a single `.npz` file,
by running:
```
$ python3 export_embeddings.py /PATH/TO/INRS/TRAIN/EMBEDDINGS /OUT/DIR
```
where `/PATH/TO/INRS/TRAIN/EMBEDDINGS` indicates the path to the directory with the embeddings
of the training INRS, and `/OUT/DIR` provides the desired output directory.

## Train Latent-GAN
Then, you can train a Latent-GAN by using the file `train_latent_gan.py` in the official Latent-GAN
repository, i.e., [https://github.com/optas/latent_3d_points](https://github.com/optas/latent_3d_points).

## Visualize results
To visualize the generated point clouds, you can run:
```
$ python3 viz.py /PATH/TO/INR2VEC/RUN.YAML /PATH/TO/LATENT/GAN/EMBEDDINGS.NPZ
```
where `/PATH/TO/INR2VEC/RUN.YAML` provides the path to the config file created by `hesiod` for the
training of `inr2vec`, and `/PATH/TO/LATENT/GAN/EMBEDDINGS.npz` indicates the output numpy file
produced during the Latent-GAN training.
