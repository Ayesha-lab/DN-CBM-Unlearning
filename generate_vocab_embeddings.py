"""
Generate CLIP embeddings for vocabulary words.
Process vocabulary in batches to avoid OOM.
"""

import torch
import clip
import os
import argparse
from pathlib import Path


def generate_vocab_embeddings(vocab_path, output_path, model_name='RN50', batch_size=1000, device='cuda'):
    """
    Generate CLIP embeddings for vocabulary words.
    
    Args:
        vocab_path: Path to vocabulary text file (one word per line)
        output_path: Where to save embeddings (.pth file)
        model_name: CLIP model to use (RN50, ViT-B/16, ViT-L/14)
        batch_size: Number of words to process at once
        device: Device to use (cuda or cpu)
    """
    
    print("\n" + "="*70)
    print("GENERATE CLIP VOCABULARY EMBEDDINGS")
    print("="*70)
    print(f"Vocabulary file:     {vocab_path}")
    print(f"Output file:         {output_path}")
    print(f"CLIP Model:          {model_name}")
    print(f"Batch size:          {batch_size}")
    print(f"Device:              {device}")
    print("="*70 + "\n")
    
    # Check vocabulary file exists
    if not os.path.exists(vocab_path):
        print(f"✗ Vocabulary file not found: {vocab_path}")
        return False
    
    # Load vocabulary
    print(f"→ Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]
    
    print(f"✓ Loaded {len(words)} vocabulary words\n")
    
    # Load CLIP model
    print(f"→ Loading CLIP model: {model_name}")
    try:
        model, preprocess = clip.load(model_name, device=device)
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading CLIP model: {e}")
        return False
    
    # Process in batches
    print(f"→ Processing vocabulary in batches of {batch_size}...")
    all_embeddings = []
    num_batches = (len(words) + batch_size - 1) // batch_size
    
    try:
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(words))
                batch_words = words[start_idx:end_idx]
                
                print(f"  Batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})...", end=' ')
                
                # Tokenize and encode
                text_tokens = clip.tokenize(batch_words).to(device)
                text_embeddings = model.encode_text(text_tokens)
                
                # Normalize embeddings
                text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
                all_embeddings.append(text_embeddings.cpu())
                
                print(f"✓ (shape: {text_embeddings.shape})")
        
        print()
    except Exception as e:
        print(f"\n✗ Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Concatenate all batches
    print(f"→ Concatenating embeddings...")
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"✓ Final embeddings shape: {all_embeddings.shape}\n")
    
    # Save embeddings
    print(f"→ Saving embeddings to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(all_embeddings, output_path)
    print(f"✓ Successfully saved embeddings\n")
    
    print("="*70)
    print("COMPLETE!")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for vocabulary")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./vocab/clipdissect_20k.txt",
        help="Path to vocabulary file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./vocab/embeddings_clipRN50_clipdissect_20k.pth",
        help="Path to save embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        choices=["RN50", "ViT-B/16", "ViT-L/14"],
        help="CLIP model to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    success = generate_vocab_embeddings(
        vocab_path=args.vocab_path,
        output_path=args.output_path,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()