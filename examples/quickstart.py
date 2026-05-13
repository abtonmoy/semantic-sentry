"""Quickstart example for SemanticSentry."""

import numpy as np
import torch
import torch.nn as nn

from semantic_sentry import DriftMonitor, AnchorSet
from semantic_sentry.adapters.custom import CustomAdapter


def create_simple_encoder(input_dim: int = 64, output_dim: int = 32) -> nn.Module:
    """Create a simple encoder for demonstration."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )


def main():
    """Run quickstart example."""
    print("=" * 60)
    print("SemanticSentry Quickstart Example")
    print("=" * 60)
    
    # 1. Create anchor set
    print("\n1. Creating anchor set...")
    anchor_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming how we build software",
        "Python is a versatile programming language",
        "Natural language processing enables machines to understand text",
        "Deep learning models require large amounts of data",
        "Transfer learning can improve model performance",
        "Neural networks are inspired by biological brains",
        "Attention mechanisms have revolutionized NLP",
        "Embeddings capture semantic meaning in vector space",
        "Fine-tuning adapts pre-trained models to specific tasks",
    ] * 10 # 100 samples
    
    anchor_set = AnchorSet(
        inputs=anchor_texts,
        labels=tuple([f"topic_{i % 5}" for i in range(len(anchor_texts))]),
        modality="text"
    )
    print(f" Created anchor set with {anchor_set.n_samples} samples")
    print(f" Version hash: {anchor_set.version_hash}")
    
    # 2. Create models
    print("\n2. Creating encoder models...")
    torch.manual_seed(42)
    model_v0 = create_simple_encoder(input_dim=64, output_dim=32)
    
    # Simulate drift by perturbing weights
    model_v1 = create_simple_encoder(input_dim=64, output_dim=32)
    with torch.no_grad():
        for param in model_v1.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    print(" Created model v0 (base)")
    print(" Created model v1 (drifted)")
    
    # 3. Create adapter
    print("\n3. Creating custom adapter...")
    def encode_fn(texts):
        # Simple tokenization (just use hash for demo)
        embeddings = []
        for text in texts:
            # Create a simple feature vector
            tokens = [ord(c) % 64 for c in text[:64]]
            tokens += [0] * (64 - len(tokens))
            embeddings.append(tokens)
        
        x = torch.tensor(embeddings, dtype=torch.float32)
        with torch.no_grad():
            emb = model_v0(x)
        return emb
    
    adapter = CustomAdapter(encode_fn=encode_fn, tower_count=1)
    print(" Adapter created")
    
    # 4. Capture snapshots
    print("\n4. Capturing snapshots...")
    monitor = DriftMonitor()
    
    # Snapshot v0
    snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=adapter)
    print(f" Captured snapshot v0: {snapshot_v0.checkpoint_hash[:8]}")
    
    # Snapshot v1 (with drifted model)
    def encode_fn_v1(texts):
        embeddings = []
        for text in texts:
            tokens = [ord(c) % 64 for c in text[:64]]
            tokens += [0] * (64 - len(tokens))
            embeddings.append(tokens)
        
        x = torch.tensor(embeddings, dtype=torch.float32)
        with torch.no_grad():
            emb = model_v1(x)
        return emb
    
    adapter_v1 = CustomAdapter(encode_fn=encode_fn_v1, tower_count=1)
    snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=adapter_v1)
    print(f" Captured snapshot v1: {snapshot_v1.checkpoint_hash[:8]}")
    
    # 5. Compare snapshots
    print("\n5. Comparing snapshots...")
    comparison = monitor.compare(snapshot_v0, snapshot_v1)
    
    print(f"\n Drift Severity: {comparison.severity.value.upper()}")
    print(f" Global Metrics:")
    for metric_name, value in comparison.global_metrics.items():
        print(f" {metric_name}: {value:.4f}")
    
    # 6. Interpret results
    print("\n6. Interpretation:")
    if comparison.severity.value == "low":
        print(" Minimal drift detected. Model is stable.")
    elif comparison.severity.value == "medium":
        print(" Moderate drift detected. Monitor closely.")
    elif comparison.severity.value == "high":
        print(" Significant drift detected. Consider re-evaluation.")
    else:
        print(" Critical drift! Model may need retraining.")
    
    # 7. Save and load
    print("\n7. Serialization test...")
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "snapshot"
        snapshot_v0.save(save_path)
        print(f" Saved snapshot to {save_path}")
        
        from semantic_sentry import Snapshot
        loaded = Snapshot.load(save_path)
        print(f" Loaded snapshot: {loaded.checkpoint_hash[:8]}")
        print(f" Match: {loaded.checkpoint_hash == snapshot_v0.checkpoint_hash}")
    
    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
