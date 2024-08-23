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
    loss: torch.Tensor,
    correct: int,
    total: int,
    running_loss_one_epoch: float,
    running_loss_start: float,
    forward_pass_time: float,
    grad_time: float,
    calc_perplexity: bool = False,
):
    run[f"metrics/{stage}_loss"].append(loss)
    if calc_perplexity:
        run[f"metrics/{stage}_batch_perplexity"].append(torch.exp(loss))
    run[f"metrics/{stage}_batch_accuracy"].append(correct / total)
    run[f"metrics/{stage}_running_loss"].append(
        running_loss_one_epoch + running_loss_start
    )
    run["metrics/forward_pass_time"].append(forward_pass_time)
    run["metrics/grad_time"].append(grad_time)


def train_one_batch(
    data: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    task: str,
):
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    if task == "classification":
        inputs, targets = data[0].to(model.device), data[1].to(model.device)
    if task == "sequence_modelling":
        inputs = data[:, :-1].to(model.device).detach().clone()
        targets = data[:, 1:].contiguous().view(-1).to(model.device).detach().clone()

    # Make predictions for this batch
    start_fwd = time.time()
    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])
    end_fwd = time.time()

    # Compute the loss function value
    loss = loss_fn(outputs, targets)

    # Calculate the gradient and update weights
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
    running_loss_start: float = 0.0,
):
    running_loss_one_epoch = 0.0
    correct = 0
    total = 0
    for inputs in tqdm(train_loader, "Epoch:"):
        loss, cor, tot, forward_pass_time, grad_time = train_one_batch(
            inputs, model, optimizer, loss_fn, task
        )
        running_loss_one_epoch += loss
        log_batch_neptune(
            stage="train",
            run=run,
            loss=loss,
            correct=cor,
            total=tot,
            running_loss_one_epoch=running_loss_one_epoch,
            running_loss_start=running_loss_start,
            forward_pass_time=forward_pass_time,
            grad_time=grad_time,
        )
        correct += cor
        total += tot

    run["metrics/train_epoch_avg_loss"].append(running_loss_one_epoch / len(train_loader))
    run["metrics/train_epoch_accuracy"].append(correct / total)
    return running_loss_one_epoch + running_loss_start


def validation_one_batch(
    data: torch.Tensor, model: nn.Module, loss_fn: nn.Module, task: str
):
    if task == "classification":
        inputs, targets = data[0].to(model.device), data[1].to(model.device)
    if task == "sequence_modelling":
        inputs = data[:, :-1].to(model.device)
        targets = data[:, 1:].contiguous().view(-1).to(model.device)

    # Make predictions for this batch
    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])

    # Compute the loss function value
    loss = loss_fn(outputs, targets)

    correct = (outputs.argmax(-1) == targets).sum().item()
    total = outputs.shape[0]
    return loss.item(), correct, total


def validation_one_epoch(
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    run: neptune.Run,
    task: str,
    running_vloss_start: float = 0.0,
):
    running_vloss_one_epoch = 0
    correct = 0
    total = 0

    for vdata in val_loader:
        vloss, cor, tot = validation_one_batch(vdata, model, loss_fn, task)
        correct += cor
        total += tot
        running_vloss_one_epoch += vloss

    run["metrics/val_avg_loss"].append(running_vloss_one_epoch / len(val_loader))
    run["metrics/val_acc"].append(correct / total)
    return running_vloss_one_epoch + running_vloss_start


def train(
    model: nn.Module,
    args: Namespace,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run: neptune.Run,
    task: str,
    epochs: int = 10,
):
    running_loss = 0.0
    running_vloss = 0.0

    hyperparams = model.get_hyperparams()

    run["model_params"] = hyperparams
    run["global_params"] = {"seed": args.seed, "device": args.device}
    run["training_params"] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "number_of_params": sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ),
        "task": task,
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
    }

    for epoch_number in range(epochs):
        model.train(True)
        running_loss = train_one_epoch(
            train_loader, model, optimizer, loss_fn, run, task, running_loss
        )

        # Evaluate model on validation set
        model.train(False)
        with torch.no_grad():
            running_vloss = validation_one_epoch(
                val_loader, model, loss_fn, run, task, running_vloss
            )

        # Save model
        path = Path(
            f"models_checkpoints/{run['sys/id'].fetch()}_{args.model}_{args.mha_type}"
        )
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"epoch-{epoch_number}.pth"
        torch.save(model.state_dict(), path)


def evaluate_one_batch(
    data: torch.Tensor, model: nn.Module, loss_fn: nn.Module, task: str
):
    if task == "classification":
        inputs, targets = data[0].to(model.device), data[1].to(model.device)
    if task == "sequence_modelling":
        inputs = data.to(model.device)
        targets = data.contiguous().view(-1).to(model.device)

    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])

    loss = loss_fn(outputs, targets)

    correct = (outputs.argmax(-1) == targets).sum().item()
    total = outputs.shape[0]
    return loss.item(), correct, total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    run: neptune.Run,
    task: str,
):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    for data in loader:
        loss, cor, tot = evaluate_one_batch(data, model, loss_fn, task)
        running_loss += loss
        correct += cor
        total += tot

    accuracy = correct / total
    avg_loss = running_loss / len(loader)
    run["metrics/val_avg_loss"].append(avg_loss)
    run["metrics/val_acc"].append(accuracy)
    return avg_loss, accuracy
