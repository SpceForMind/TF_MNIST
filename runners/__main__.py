from argparse import ArgumentParser

from runners.train import main as train_main
from runners.test import main as test_main


def parse_args():
    parser = ArgumentParser(description='Runners package')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='Run train loop')
    train_parser.add_argument('--epochs',
                              action='store',
                              dest='epochs',
                              type=int,
                              required=True)
    train_parser.add_argument('--model_path',
                              action='store',
                              dest='model_path',
                              type=str,
                              required=False)
    train_parser.set_defaults(func=train_main)

    test_parser = subparsers.add_parser('test', help='Run test loop')
    test_parser.add_argument('--model_path',
                              action='store',
                              dest='model_path',
                              type=str,
                              required=False)
    test_parser.set_defaults(func=test_main)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
