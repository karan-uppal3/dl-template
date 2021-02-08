import argparse
import os.path as osp


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="training script for FDA")
        parser.add_argument("--model", type=str, default='enet',
                            help="available options : enet")
        parser.add_argument("--GPU", type=str, default='0',
                            help="which GPU to use")
        parser.add_argument("--data-dir", type=str, default='../data_semseg',
                            help="Path to the directory containing the dataset.")
        parser.add_argument("--set", type=str, default='train',
                            help="choose adaptation set.")
        parser.add_argument("--batch-size", type=int,
                            default=10, help="input batch size.")
        parser.add_argument("--epochs", type=int,
                            default=300, help="Number of epochs.")
        parser.add_argument("--learning-rate", type=float, default=5e-4,
                            help="initial learning rate for the segmentation network.")
        parser.add_argument("--weight-decay", type=float, default=2e-4,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--num-classes", type=int, default=19,
                            help="Number of classes for cityscapes.")
        parser.add_argument("--print-freq", type=int,
                            default=1, help="print loss and time fequency.")
        parser.add_argument("--restore", type=bool,
                            default=False, help="Whether to restore from previous checkpoint.")
        parser.add_argument("--checkpoint_path", type=str,
                            default='./checkpoint/current_checkpoint.pt', help="Path to previous checkpoint.")
        parser.add_argument("--best_model_path", type=str,
                            default='./best_model/best_model.pt', help="Path to best checkpoint.")
        parser.add_argument("--wandb_key", type=str,
                            default='8836a9cd165e3f15e80fb49e2bc9362a6bb63bb7', help="Wandb api key")
        parser.add_argument("--wandb_id", type=str,
                            default=None, help="Wandb run resume id")

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
