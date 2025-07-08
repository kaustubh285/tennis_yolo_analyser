import functools
import datetime
from typing import Any, Callable


def log_function_call(func: Callable) -> Callable:
    """
    Decorator that logs function calls with timestamp, function name, and arguments.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        func_name = func.__name__

        # Format arguments
        args_str = ", ".join(repr(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())

        all_args = []
        if args_str:
            all_args.append(args_str)
        if kwargs_str:
            all_args.append(kwargs_str)

        args_display = ", ".join(all_args) if all_args else "no arguments"

        # print(f"{timestamp} : '{func_name}' : args  {args_display}")
        print(f"{timestamp} : '{func_name}'")

        result = func(*args, **kwargs)
        return result

    return wrapper
