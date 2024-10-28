import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_scheduler_with_warmup(
    optimizer, num_warmup_steps, final_lr_fraction, num_all_steps
):
    """
    Function that returns a scheduler that warms up the learning rate for lr_warmup_steps steps,
    then decays it using cosine schedule to final_lr_fraction * lr and then stays constant.
    """

    def get_fraction(step: int):
        if step < num_warmup_steps:
            return (step + 1) / num_warmup_steps
        # cosine schedule that ends at final_lr_fraction * lr, then constant
        elif step < num_all_steps:
            return final_lr_fraction + 0.5 * (1 - final_lr_fraction) * (
                1
                + math.cos(
                    math.pi
                    * (step - num_warmup_steps)
                    / (num_all_steps - num_warmup_steps)
                )
            )
        else:
            return final_lr_fraction

    return LambdaLR(optimizer, get_fraction)
