Extract features from celebA using CLIP
```
python extract_celeba_clip_sae.py \
--img_enc_name clip_RN50 \
--hook_points out \
--device cuda \
--celeba_root ./data/celeba \
--save_dir_sae_ckpts ./ckpt_and_names \
--output_dir ./data/activations_img/celeba/clip_RN50/out

```
Save CelebA features from CLIP
```
python scripts/save_probe_features.py \
    --img_enc_name clip_RN50 \
    --probe_dataset celeba
```

save concept strengths using checkpoints/clip_RN50_sparse_autoenconder_final.pt. This creates sae_activations.pth for each split in data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth
```
python scripts/save_concept_strengths.py \
    --lr 5e-4 \
    --l1_coeff 3e-5 \
    --expansion_factor 8 \
    --img_enc_name clip_RN50 \
    --num_epochs 200 \
    --resample_freq 10 \
    --train_sae_bs 4096 \
    --probe_dataset celeba \
    --probe_split train

```

Train linear probe on celebA using CLIP_SAE ckpt and assigned names:
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
    --l1_coeff 0.0 \
    --device cuda \
    --output_dir ./train_linear_probe_celeba/outputs
```

