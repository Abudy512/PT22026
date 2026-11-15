
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def train_model(
    model,
    dataloader,
    epochs: int = 1,
    learning_rate: float = 0.001,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
):
    if epochs < 1:
        raise ValueError("Epochs must be at least 1")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.L1Loss(reduction="none")
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            texts = batch["text"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            targets = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(
                texts,
                text_lengths,
                mel_lengths,
            )

            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes do not match: "
                    f"{predictions.shape} vs {targets.shape}"
                )

            frame_positions = torch.arange(
                targets.shape[2],
                device=device,
            )

            mask = (
                frame_positions.unsqueeze(0)
                < mel_lengths.unsqueeze(1)
            ).unsqueeze(1)

            errors = loss_function(predictions, targets)

            masked_errors = errors * mask

            valid_elements = mask.sum() * targets.shape[1]

            loss = masked_errors.sum() / valid_elements

            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")

            loss.backward()

            clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("The training dataloader is empty")

        average_loss = total_loss / batch_count
        history.append(average_loss)

        if checkpoint_path is not None:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
            }

            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(checkpoint, checkpoint_file)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    return history
