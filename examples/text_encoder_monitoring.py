"""Example: Monitoring text encoder drift."""

import numpy as np
import torch
import torch.nn as nn

from semantic_sentry import DriftMonitor, AnchorSet
from semantic_sentry.adapters.custom import CustomAdapter


def create_text_encoder(vocab_size: int = 1000, embed_dim: int = 64, output_dim: int = 128) -> nn.Module:
    """Create a simple text encoder."""
    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.encoder = nn.LSTM(embed_dim, output_dim, batch_first=True, bidirectional=True)
            self.projection = nn.Linear(output_dim * 2, output_dim)
        
        def forward(self, x):
            emb = self.embedding(x)
            _, (h_n, _) = self.encoder(emb)
            # Concatenate forward and backward
            h = torch.cat([h_n[0], h_n[1]], dim=-1)
            return self.projection(h)
    
    return TextEncoder()


def tokenize_simple(texts, vocab_size=1000):
    """Simple tokenization (hash-based for demo)."""
    tokenized = []
    for text in texts:
        tokens = [hash(word) % vocab_size for word in text.lower().split()[:10]]
        tokens += [0] * (10 - len(tokens))  # Pad
        tokenized.append(tokens)
    return torch.tensor(tokenized, dtype=torch.long)


def main():
    """Run text encoder monitoring example."""
    print("=" * 60)
    print("Text Encoder Drift Monitoring")
    print("=" * 60)
    
    # Create anchor set with semantic clusters
    print("\n1. Creating semantic anchor set...")
    anchor_texts = [
        # Technology cluster
        "Artificial intelligence is revolutionizing industries",
        "Machine learning algorithms improve with data",
        "Neural networks can learn complex patterns",
        "Deep learning requires powerful computing resources",
        "Computer vision enables machines to see",
        
        # Nature cluster
        "The forest ecosystem supports diverse wildlife",
        "Ocean currents regulate global climate patterns",
        "Mountain ranges influence local weather systems",
        "Coral reefs are biodiversity hotspots",
        "Wetlands filter water and prevent flooding",
        
        # Business cluster
        "Companies compete in global markets",
        "Investors seek returns on capital",
        "Supply chains connect producers and consumers",
        "Marketing strategies drive sales growth",
        "Innovation creates competitive advantages",
    ] * 6  # 90 samples
    
    labels = ["tech"] * 30 + ["nature"] * 30 + ["business"] * 30
    
    anchor_set = AnchorSet(
        inputs=anchor_texts,
        labels=tuple(labels),
        modality="text"
    )
    print(f"   Created {anchor_set.n_samples} anchor samples across 3 topics")
    
    # Create models
    print("\n2. Creating text encoders...")
    torch.manual_seed(42)
    model_v0 = create_text_encoder()
    
    # Simulate fine-tuning drift
    model_v1 = create_text_encoder()
    model_v1.load_state_dict(model_v0.state_dict())
    with torch.no_grad():
        # Simulate fine-tuning on nature domain
        for param in model_v1.embedding.parameters():
            param.add_(torch.randn_like(param) * 0.3)
    
    print("   Created base model v0")
    print("   Created fine-tuned model v1 (nature-biased)")
    
    # Create adapters
    def make_adapter(model):
        def encode(texts):
            tokens = tokenize_simple(texts)
            with torch.no_grad():
                return model(tokens)
        return CustomAdapter(encode_fn=encode, tower_count=1)
    
    adapter_v0 = make_adapter(model_v0)
    adapter_v1 = make_adapter(model_v1)
    
    # Monitor drift
    print("\n3. Monitoring drift...")
    monitor = DriftMonitor()
    
    snapshot_v0 = monitor.snapshot(model_v0, anchor_set, adapter=adapter_v0)
    snapshot_v1 = monitor.snapshot(model_v1, anchor_set, adapter=adapter_v1)
    
    comparison = monitor.compare(snapshot_v0, snapshot_v1)
    
    print(f"\n   Drift Analysis:")
    print(f"   - Severity: {comparison.severity.value.upper()}")
    print(f"   - CKA (global): {comparison.global_metrics.get('cka', 0):.4f}")
    print(f"   - NPS (local): {comparison.global_metrics.get('nps', 0):.4f}")
    print(f"   - Isotropy Δ: {comparison.global_metrics.get('isotropy_delta', 0):.4f}")
    
    # Analyze per-cluster drift
    print("\n4. Per-cluster analysis...")
    print("   (Would analyze per-cluster NPS here)")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
