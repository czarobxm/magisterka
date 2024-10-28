import argparse

from training_setup.config import DEFAULT_ARGS


def t_or_f(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(f"Invalid value for boolean: {arg}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training parameters")

    # Add arguments based on DEFAULT_ARGS
    for arg, value in DEFAULT_ARGS.items():
        arg_type = type(value) if value is not None else str
        if isinstance(value, bool):
            parser.add_argument(f"--{arg}", type=t_or_f, default=value, help="")
        elif isinstance(value, list):
            parser.add_argument(f"--{arg}", nargs="+", type=int, default=value, help="")
        else:
            parser.add_argument(f"--{arg}", type=arg_type, default=value, help="")

    return parser.parse_args()
