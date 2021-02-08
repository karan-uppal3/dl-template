from model.enet import ENet
from model.deeplab import Deeplab
import torch.optim as optim


def CreateModel(args):

    if args.model == 'enet':
        model = ENet(num_classes=args.num_classes).cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.model == 'deeplab':
        model = Deeplab(num_classes=args.num_classes).cuda()
        optimizer = None

    else:
        raise ValueError('The model must be enet')

    return model, optimizer
