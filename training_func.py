from typing import Tuple
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import neptune
from tqdm import tqdm


def log_batch_neptune(
    stage: str,
    run: neptune.Run,
    loss: float,
    correct: int,
    total: int,
    forward_pass_time: float,
    grad_time: float,
    calc_perplexity: bool = False,
) -> None:
    metrics = {
        f"{stage}_loss": loss,
        f"{stage}_batch_accuracy": correct / total,
        "forward_pass_time": forward_pass_time,
        "grad_time": grad_time,
    }
    if calc_perplexity:
        metrics[f"{stage}_batch_perplexity"] = torch.exp(torch.tensor(loss)).item()

    for key, value in metrics.items():
        run[f"metrics/{key}"].append(value)


def prepare_inputs_and_targets(
    data: torch.Tensor, task: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if task == "classification":
        return data[0].to(device), data[1].to(device)
    elif task == "sequence_modelling":
        inputs = data.detach().clone().to(device)
        targets = data.contiguous().view(-1).to(device)
        return inputs, targets
    else:
        raise ValueError(f"Unsupported task: {task}")


def train_one_batch(
    data: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    task: str,
) -> Tuple[float, int, int, float, float]:
    optimizer.zero_grad()
    inputs, targets = prepare_inputs_and_targets(data, task, model.device)

    start_fwd = time.time()
    outputs = model(inputs).view(targets.shape[0], -1)
    end_fwd = time.time()

    loss = loss_fn(outputs, targets)

    start_grad = time.time()
    loss.backward()
    optimizer.step()
    end_grad = time.time()

    correct = (outputs.argmax(-1) == targets).sum().item()
    total = outputs.shape[0]

    return loss.item(), correct, total, end_fwd - start_fwd, end_grad - start_grad


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    run: neptune.Run,
    task: str,
):
    running_loss = 0
    correct = 0
    total = 0
    for inputs in tqdm(train_loader, "Training:"):
        loss, cor, tot, forward_pass_time, grad_time = train_one_batch(
            inputs, model, optimizer, loss_fn, task
        )
        running_loss += loss
        correct += cor
        total += tot
        log_batch_neptune(
            stage="train",
            run=run,
            loss=loss,
            correct=cor,
            total=tot,
            forward_pass_time=forward_pass_time,
            grad_time=grad_time,
        )

    run["metrics/train_epoch_avg_loss"].append(running_loss / len(train_loader))
    run["metrics/train_epoch_accuracy"].append(correct / total)
    return True


def evaluate_one_batch(
    data: torch.Tensor, model: nn.Module, loss_fn: nn.Module, task: str
) -> Tuple[float, int, int]:
    inputs, targets = prepare_inputs_and_targets(data, task, model.device)
    outputs = model(inputs).view(targets.shape[0], -1)
    loss = loss_fn(outputs, targets)
    correct = (outputs.argmax(-1) == targets).sum().item()
    return loss.item(), correct, outputs.shape[0]


def evaluate_one_epoch(
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    run: neptune.Run,
    task: str,
    stage: str = "val",
) -> float:
    running_vloss = 0
    correct = total = 0

    with torch.no_grad():
        for vdata in val_loader:
            vloss, batch_correct, batch_total = evaluate_one_batch(
                vdata, model, loss_fn, task
            )
            running_vloss += vloss
            correct += batch_correct
            total += batch_total

    avg_loss = running_vloss / len(val_loader)
    accuracy = correct / total
    run[f"metrics/{stage}_avg_loss"].append(avg_loss)
    run[f"metrics/{stage}_acc"].append(accuracy)
    return running_vloss


def train(
    model: nn.Module,
    args: Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    run: neptune.Run,
    task: str,
    epochs: int = 10,
) -> None:
    log_hyperparameters(model, args, run)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_one_epoch(train_loader, model, optimizer, loss_fn, run, task)
        if scheduler:
            scheduler.step()

        evaluate_one_epoch(val_loader, model, loss_fn, run, task, "val")

        save_model(model, args, run, epoch)

    evaluate_one_epoch(test_loader, model, loss_fn, run, task, "test")


def log_hyperparameters(model: nn.Module, args: Namespace, run: neptune.Run) -> None:
    run["model_params"] = model.get_hyperparams()
    run["global_params"] = {"seed": args.seed, "device": args.device}
    run["training_params"] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "number_of_params": sum(
            p.nelement() for p in model.parameters()
        ),  # T O D O check if this is correct
        "task": args.task,
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
    }


def save_model(model: nn.Module, args: Namespace, run: neptune.Run, epoch: int) -> None:
    path = Path(
        f"models_checkpoints/{run['sys/id'].fetch()}_{args.model}_{args.mha_type}"
    )
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"epoch-{epoch}.pth")
