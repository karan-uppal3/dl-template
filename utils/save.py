import torch.nn.functional as F
import shutil
import numpy as np
import torch


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """Saves the model state after chosen number of epochs

    Args:
    =========
        state: checkpoint we want to save

        is_best: is this the best checkpoint; min validation loss

        checkpoint_path: path to save checkpoint

        best_model_path: path to save best model
    """
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(checkpoint_path, best_model_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """Load a checkpoint state for the model

    Args:
    =========
        checkpoint_path: path to save checkpoint

        model: model that we want to load checkpoint parameters into

        optimizer: optimizer we defined in previous training

    Returns:
    =========
        model: model that we want to load checkpoint parameters into

        optimizer: optimization method used

        checkpoint: model state parameters loaded

        valid_loss_min: minimum validation loss at the checkpoint 

    """

    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min
