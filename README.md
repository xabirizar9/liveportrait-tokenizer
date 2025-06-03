# liveportrait-tokenizer

### Setup

Run the setup.sh script to install all of the necessary dependencies:

```bash
./setup.sh
```

### Testing tokenizer

After installing the environment, you can now test the tokenizer in the `test_tokenizer.ipynb` notebook.


### Training model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_tokenizer.py --config configs/vqvae_config.yaml
```

Modify `configs/vqvae_config.yaml` to enable/disable the features you want to use during training.
You have to manually adjust the `nfeats` of the VQ-VAE based on the sum of the dimensions of the features you want to use.

### Inference

To run inference, you'll need a trained model checkpoint (`.ckpt` or `.pth` file).  Make sure you have the correct path to this file.

1.  **Open Jupyter Notebook:** Start by opening a Jupyter Notebook in your environment.

2.  **Load the Model:**  In your notebook, you'll need to load the model weights from the checkpoint file.
The checkpoints are stored by default under `outputs`.
Just select one of the checkpoints under `outputs/<RUN_NAME>/checkpoints/`.
Passing the relative path like so will load the config used in the run.
