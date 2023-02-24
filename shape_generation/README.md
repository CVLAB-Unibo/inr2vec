# Shape generation

The formulation that we adopted for `inr2vec` allows using the same approach for unconditioned
shape generation for different discrete representations of shapes, such as point clouds, meshes,
voxels, etc. In this directory, we assume that you already have a trained `inr2vec` model and the
embeddings of the training INRs. We then provide the code to train a Latent-GAN on such embeddings.

## Create the dataset of INRs, train `inr2vec` and export embeddings
Please use the instructions and the code in the  other directories.

## Export embeddings for Latent-GAN
To train a Latent-GAN on `inr2vec` embeddings, first export the embeddings in a single `.npz` file,
by running:
```
$ cd shape_generation
$ python3 export_embeddings.py /PATH/TO/INRS/TRAIN/EMBEDDINGS /OUT/DIR
```
where `/PATH/TO/INRS/TRAIN/EMBEDDINGS` indicates the path to the directory with the embeddings
of the training INRS, and `/OUT/DIR` provides the desired output directory. The script will create
one `.npz` file for each class of the dataset.

## Train Latent-GAN
Then, you can train a Latent-GAN by using the script that we provide `train_latent_gan.py`
with the code from the official Latent-GAN repository, i.e.,
[https://github.com/optas/latent_3d_points](https://github.com/optas/latent_3d_points).
You just need to set the path to one of the exported `.npz` files inside the script.

## Visualize results
To visualize the generated shapes, you can run:
```
$ python3 viz_pcd.py /PATH/TO/INR2VEC/RUN.YAML /PATH/TO/LATENT/GAN/EMBEDDINGS.NPZ
$ python3 viz_mesh.py /PATH/TO/INR2VEC/RUN.YAML /PATH/TO/LATENT/GAN/EMBEDDINGS.NPZ
$ python3 viz_voxel.py /PATH/TO/INR2VEC/RUN.YAML /PATH/TO/LATENT/GAN/EMBEDDINGS.NPZ
```
where `/PATH/TO/INR2VEC/RUN.YAML` provides the path to the config file created by `hesiod` for the
training of `inr2vec`, and `/PATH/TO/LATENT/GAN/EMBEDDINGS.npz` indicates the output numpy file
produced during the Latent-GAN training.
Please note that we provide three different scripts for visualizing the generated shapes,
depending on the discrete representation used during the training of `inr2vec`.
