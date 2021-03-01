import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
from data.cityscapes_dataset import cityscapesLoader
from model import CreateModel
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from torch.autograd import Variable
import scipy.io as sio
from evaluate import evaulate
from torch.utils import data
from utils.save import save_ckp, load_ckp
import wandb

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1, 3, 1, 1))
CS_weights = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32)
CS_weights = torch.from_numpy(CS_weights)


def main():

    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time': Timer()}

    opt.print_options(args)

    train_data = cityscapesLoader(
        args.data_dir, split='train', is_transform=True)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size)

    val_data = cityscapesLoader(
        args.data_dir, split='val', is_transform=True)
    val_loader = data.DataLoader(val_data, batch_size=1)

    model, optim = CreateModel(args)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    wandb.login(key=args.wandb_key)

    start_iter = 0
    valid_loss_min = np.Inf

    if args.restore is True:
        model, optim, start_iter, valid_loss_min = load_ckp(
            args.checkpoint_path, model, optim)
        wandb.init(project="DL-SegSem", resume='allow', id=args.wandb_id)

    else:
        wandb.init(project="DL-SegSem")
        
    wandb.config.update(args)

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    _t['iter time'].tic()

    for epoch in range(start_iter+1, args.epochs):

        epoch_loss = 0.0

        for step, batch_data in enumerate(train_loader):

            # Get the inputs and labels
            inputs = batch_data[0].to('cuda')
            labels = batch_data[1].to('cuda').long()

            # Forward propagation
            outputs = model(inputs)

            # Loss computation
            loss = criterion(outputs, labels)

            print("Epoch:", epoch, "| Step:", step, "| Loss:", loss.item())

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

        val_data = evaulate(model, val_loader)
        val_loss = val_data['Loss']

        if epoch % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print("Epoch number:", epoch, "| Training Loss=", epoch_loss /
                  len(train_loader), "| Valiadtion Loss= ", val_loss, "| Mean IOU (val)=", val_data['Mean IOU'], "(", _t['iter time'].diff, "seconds )")
            _t['iter time'].tic()

        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': valid_loss_min,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
        }

        save_ckp(checkpoint, False, args.checkpoint_path, args.best_model_path)

        wandb.log({
            "Training Loss": epoch_loss/len(train_loader),
            "Validation Loss": val_loss,
            "Mean IoU": val_data['Mean IOU'],
            "Predictions": val_data['Predictions']
        })

        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, val_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, args.checkpoint_path,
                     args.best_model_path)
            valid_loss_min = val_loss


if __name__ == '__main__':
    main()
