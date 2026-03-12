"""
Analysis: How to choose retain_dataset size for DN-CBM unlearning.

Given:
  - outputs/Mouth_Slightly_Open/binary_probe.pt   (trained probe)
  - outputs/Mouth_Slightly_Open/all_concepts_ranked.csv
  - outputs/Mouth_Slightly_Open/top_concepts.txt
  - SAE activations (train split)

Steps:
  1. Load probe weights → identify top concept index
  2. Load SAE train activations → score each image on top concept
  3. Sweep forget_k = [10, 50, 100, 200, 500] (top-k forget images)
  4. For each forget_k, compute:
       - forget set size
       - retain set size = n_train - forget_k
       - forget/retain ratio
       - concept activation stats in forget vs retain
       - probe accuracy on retain alone (proxy for recovery potential)
  5. Plot/print summary table
"""

import torch
import csv
import numpy as np
import os

# ── CONFIG — adjust paths ────────────────────────────────────────────────────
PROBE_PATH        = "./train_linear_probe_celeba/outputs/Mouth_Slightly_Open/binary_probe.pt"
print(PROBE_PATH, "works")
ALL_CONCEPTS_CSV  = "./train_linear_probe_celeba/outputs/Mouth_Slightly_Open/all_concepts_ranked.csv"
TRAIN_ACTS_PATH   = "./data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth"   # your SAE activations
CELEBA_ROOT       = "./data/celeba"
ATTRIBUTE         = "Mouth_Slightly_Open"

FORGET_K_SWEEP    = [10, 50, 100, 200, 500]
# ─────────────────────────────────────────────────────────────────────────────


def load_top_concept_idx(csv_path):
    """Return concept_idx of the rank-1 concept from all_concepts_ranked.csv."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return int(row["concept_idx"])  # rank 1 = first row


def load_probe_weights(probe_path):
    ckpt = torch.load(probe_path, map_location="cpu")
    # probe checkpoint has 'weights' key: shape [n_concepts]
    return ckpt["weights"]


def main():
    print("=" * 70)
    print("RETAIN DATASET SIZE ANALYSIS")
    print("=" * 70)

    # 1. Load top concept
    top_concept_idx = load_top_concept_idx(ALL_CONCEPTS_CSV)
    print(f"✓ Top concept index: {top_concept_idx}")

    # 2. Load probe weights
    weights = load_probe_weights(PROBE_PATH)
    top_weight = weights[top_concept_idx].item()
    print(f"✓ Top concept probe weight: {top_weight:+.6f}")

    # 3. Load SAE activations for train split [N_train, n_concepts]
    print(f"\n→ Loading SAE activations from: {TRAIN_ACTS_PATH}")
    acts = torch.load(TRAIN_ACTS_PATH, map_location="cpu")
    if acts.ndim == 3 and acts.shape[1] == 1:
        acts = acts.squeeze(1)
    n_train, n_concepts = acts.shape
    print(f"✓ Activations shape: {acts.shape}")

    # 4. Score each image on the top concept
    concept_scores = acts[:, top_concept_idx]  # [N_train]
    sorted_indices = torch.argsort(concept_scores, descending=True)

    print(f"\n{'forget_k':>10} {'retain_n':>10} {'f/r ratio':>12} "
          f"{'forget_mean_act':>18} {'retain_mean_act':>18} "
          f"{'forget_min_act':>16} {'act_gap':>10}")
    print("-" * 100)

    results = []
    for k in FORGET_K_SWEEP:
        forget_idx  = sorted_indices[:k]
        retain_idx  = sorted_indices[k:]

        forget_acts = concept_scores[forget_idx]
        retain_acts = concept_scores[retain_idx]

        forget_mean = forget_acts.mean().item()
        retain_mean = retain_acts.mean().item()
        forget_min  = forget_acts.min().item()   # weakest in forget set
        act_gap     = forget_min - retain_acts.max().item()  # separation margin
        ratio       = k / len(retain_idx)

        results.append({
            "forget_k": k,
            "retain_n": len(retain_idx),
            "ratio": ratio,
            "forget_mean": forget_mean,
            "retain_mean": retain_mean,
            "forget_min": forget_min,
            "act_gap": act_gap,
        })

        print(f"{k:>10} {len(retain_idx):>10} {ratio:>12.4f} "
              f"{forget_mean:>18.4f} {retain_mean:>18.4f} "
              f"{forget_min:>16.4f} {act_gap:>10.4f}")

    # 5. Recommendations
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  act_gap > 0   → clean separation: forget images activate concept
                  more than ALL retain images. Ideal for unlearning.
  act_gap < 0   → overlap: some retain images are as active as
                  forget images. Unlearning may damage retain set.

  forget_min    → the 'weakest' image in your forget set.
                  If it's near 0, you may be including images where
                  the concept barely fires → noisy forget signal.

  ratio (f/r)   → keep this small (< 0.01 ideally).
                  Certified unlearning assumes forget << retain.
    """)

    # Find recommended k: largest k where act_gap >= 0
    clean_ks = [r for r in results if r["act_gap"] >= 0]
    if clean_ks:
        best = clean_ks[-1]
        print(f"✓ Recommended forget_k = {best['forget_k']}  "
              f"(retain_n = {best['retain_n']}, ratio = {best['ratio']:.4f})")
    else:
        print("⚠  No clean separation found. Consider using forget_k=10 "
              "and accepting some activation overlap.")

    print("\n→ Activation distribution of top concept across ALL train images:")
    pcts = [0, 25, 50, 75, 90, 95, 99, 100]
    percentiles = np.percentile(concept_scores.numpy(), pcts)
    for p, v in zip(pcts, percentiles):
        print(f"   p{p:3d}: {v:.4f}")


if __name__ == "__main__":
    main()