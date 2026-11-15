
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from voiceforge.batching import collate_batch
from voiceforge.dataset import LJSpeechDataset
from voiceforge.model import AcousticModel
from voiceforge.text import CHAR_TO_ID
from voiceforge.training import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the VoiceForge acoustic model."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="data/LJSpeech-1.1/metadata.csv",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/voiceforge.pt",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    print("Loading LJSpeech dataset...")

    dataset = LJSpeechDataset(metadata_path)

    print(f"Dataset samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        n_mels=80,
        embedding_dim=128,
        hidden_dim=128,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")
    print("Starting VoiceForge training...")

    history = train_model(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    print("\nTraining completed.")

    for index, loss in enumerate(history, start=1):
        print(f"Epoch {index}: loss = {loss:.4f}")

    print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
