import os
import random
import datetime
import numpy as np
import torch as th

#-------used in my utils------
from omegaconf import DictConfig, OmegaConf
import fcntl


def format_metrics(metrics):
    """Format metric in metric dict for logging."""
    return f"loss: {metrics['loss']:.2f}, roc: {metrics['roc']*100:.2f}, ap: {metrics['ap']*100:.2f}"
    # return " ".join(
            # ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

#----------- My utils -------
def set_seed(seed):
    # th.set_num_threads(threadnum)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def get_model_name(args):
    if args.description:
        model = f'{args.model}-{args.description}'
    else:
        model = f'{args.model}'

    if args.is_bn:
        model=f'{model}-{args.bn_type}'

    optim = f'{args.lr}-{args.optimizer}-{args.weight_decay}'
    name = f'{args.seed}-{args.dataset}-{optim}-{model}-dim_{args.dim}-L_{args.num_layers}-{datetime.datetime.now().strftime("%H_%M")}'
    return name

def parse_cfg(args, cfg: DictConfig):
    # Function to recursively set attributes, keeping only the final key name
    def set_attributes_from_dict(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                # If the value is a dict, continue to extract its values
                set_attributes_from_dict(target, value)
            else:
                # Directly set the attribute on the target
                setattr(target, key, value)

    # Convert Hydra config to a nested dictionary and then flatten it
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_attributes_from_dict(args, cfg_dict)

    # get model name
    args.modelname = get_model_name(args)

    # Set args for HNN
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    return args

def write_final_results(dataset_name,message):
    file_path = os.path.join(os.getcwd(), 'final_results_' + dataset_name)
    # Create a file lock
    with open(file_path, "a",encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock

        # Write the message to the file
        file.write(message + "\n")

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


