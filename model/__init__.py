from model.enet import ENet
import torch.optim as optim


def CreateModel(args):

    if args.model == 'enet':
        model = ENet(num_classes=args.num_classes)
    else:
        raise ValueError('The model must be enet')

    return model
