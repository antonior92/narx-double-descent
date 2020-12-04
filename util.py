import argparse
import inspect


def parse_funct_arguments(fn, args=None, free_arguments=None):
    if free_arguments is None:
        free_arguments = []
    fn_parser = argparse.ArgumentParser()
    sign = inspect.signature(fn)
    for pname, pval in sign.parameters.items():
        if pname not in free_arguments:
            fn_parser.add_argument('--'+pname, default=pval.default, type=pval.annotation)
    fn_args, unk = fn_parser.parse_known_args(args)

    def new_fn(*args, **kwargs):
        return fn(*args, **kwargs, **vars(fn_args))

    return new_fn, vars(fn_args), unk


