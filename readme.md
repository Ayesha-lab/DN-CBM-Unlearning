## Unlearning using DN-CBMs
The following instructions are to help reproduce all results. But all checkpoints/important files are also provided.

Run setup commands from original README_DN_CBMs.md to ensure all dependencies are installed.

Extract features from celebA using CLIP on splits (train, test, val):
```
python extract_celeba_clip_sae.py     
--img_enc_name clip_RN50     
--expansion_factor 8     
--split test     
--celeba_root ./data/celeba     
--batch_size 64     
--device cuda
--output_dir ./data/activations_img/celeba/clip_RN50/out 
--sae_checkpoint_path checkpoints/clip_RN50_sparse_autoencoder_final.pt
```
### Train linear probe

Training results are printed in terminal, but also available directly in mlflow ui.
To access mlflow, run in separate terminal:

```
mlflow ui
```

Click on the url to open and view all experiments run.

For chosen attribute
train linear probe on celebA using CLIP_SAE ckpt and assigned names:
```
python -m train_linear_probe_celeba.main \
    --celeba_root ./data/celeba \
    --train_activations_path ./data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth \
    --val_activations_path   ./data/activations_img/celeba/clip_RN50/out/val/sae_activations.pth \
    --test_activations_path  ./data/activations_img/celeba/clip_RN50/out/test/sae_activations.pth \
    --attribute Blond_Hair \
    --num_epochs 50 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --l1_coeff 1e-3 \
    --device cuda \
    --output_dir ./train_linear_probe_celeba/outputs
```

Evaluate_concepts_celeba.ipynb contains all steps for creating the grids.

### Unlearn concepts

Simply run:

```
python Unlearning/run_celeba_unlearning.py
```

This unlearns top-k (k=100) images for top concept. All unlearning variables are hardcoded and can be adjusted in the script itself.
note: This is not all the code, simply the final code used for the results presented in the seminar. :)