import logging

from training_func import train
from training_setup.attention_params import get_attention_params

from training_setup import (
    parse_arguments,
    setup_logging,
    setup_tokenizer,
    create_dataloaders,
    setup_neptune,
    initialize_model,
    setup_training,
)


def main():
    args = parse_arguments()
    setup_logging()
    logging.info("Starting training script.")

    tokenizer = setup_tokenizer(args)
    logging.info("Tokenizer set up.")

    logging.info("Loading and tokenizing dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(args, tokenizer)
    logging.info("Data loaders created.")

    run = setup_neptune(args)
    logging.info("Neptune run initialized.")

    method_params = get_attention_params(args)
    model = initialize_model(args, tokenizer, method_params)
    logging.info("Model %s initialized.", args.model)

    training_setup = setup_training(args, model)
    logging.info("Training setup completed.")

    logging.info("Starting training...")
    train(
        model,
        args,
        training_setup["optimizer"],
        training_setup["scheduler"],
        training_setup["loss_fn"],
        train_loader,
        val_loader,
        test_loader,
        run,
        task=args.task,
        epochs=args.epochs,
    )
    logging.info("Training finished.")

    logging.info("Average loss on test set: %s", run["metrics/test_avg_loss"])
    logging.info("Accuracy on test set: %s", "metrics/test_acc")
    logging.info("Evaluation finished.")

    run.stop()


main()
