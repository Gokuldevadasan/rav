"""Small transparent training loop used for COCO VLM experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def configure_lora(model: Any, settings: Any, *, qlora: bool = False) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("LoRA requires peft. Install: pip install -e '.[vlm]'") from exc
    if qlora:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def trainable_parameter_count(model: Any) -> tuple[int, int]:
    parameters = list(model.parameters())
    return sum(p.numel() for p in parameters if p.requires_grad), sum(p.numel() for p in parameters)


def smoke_train(
    model: Any,
    dataloader: Any,
    *,
    device: str,
    learning_rate: float = 2e-4,
    gradient_accumulation: int = 1,
    epochs: int = 1,
) -> float:
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    final_loss = 0.0
    for _ in range(epochs):
        step = 0
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / gradient_accumulation
            loss.backward()
            final_loss = float(loss.detach().cpu()) * gradient_accumulation
            if step % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step % gradient_accumulation:
            optimizer.step()
    return final_loss


def evaluate_loss(model: Any, dataloader: Any, *, device: Any) -> float:
    """Return mean validation loss without retaining gradients."""
    import torch
    from tqdm.auto import tqdm

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    if not losses:
        raise ValueError("Validation dataloader is empty.")
    return sum(losses) / len(losses)


def train_with_validation(
    model: Any,
    train_dataloader: Any,
    validation_dataloader: Any,
    *,
    device: Any,
    learning_rate: float,
    gradient_accumulation: int,
    epochs: int,
    save_steps: int,
    on_checkpoint: Callable[[int], None],
    on_validation: Callable[[int, float], None],
) -> tuple[float, float, int]:
    """Train by epoch, validate after each epoch, and emit periodic checkpoints."""
    import torch
    from tqdm.auto import tqdm

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    final_training_loss = 0.0
    final_validation_loss = 0.0
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_steps = 0
        progress = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            raw_loss = model(**batch).loss
            (raw_loss / gradient_accumulation).backward()
            final_training_loss = float(raw_loss.detach().cpu())
            global_step += 1
            pending_steps += 1
            progress.set_postfix(
                loss=f"{final_training_loss:.4f}",
                optimizer_steps=global_step // gradient_accumulation,
            )
            if pending_steps == gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
            if global_step % save_steps == 0:
                on_checkpoint(global_step)
        if pending_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        final_validation_loss = evaluate_loss(model, validation_dataloader, device=device)
        on_validation(epoch, final_validation_loss)
    return final_training_loss, final_validation_loss, global_step
