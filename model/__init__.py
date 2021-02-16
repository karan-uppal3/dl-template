from model.enet import ENet
from model.deeplab import Deeplab
from model.FRRNet import FRRNet
import torch.optim as optim


def CreateModel(args):

    if args.model == 'enet':
        model = ENet(num_classes=args.num_classes).cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.model == 'deeplab':
        model = Deeplab(num_classes=args.num_classes).cuda()
        optimizer = None

    elif args.model == 'frrnet':
        model = FRRNet(out_channels=args.num_classes).cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate)

    else:
        raise ValueError('The model must be enet/deeplab/frrnet')

    return model, optimizer
