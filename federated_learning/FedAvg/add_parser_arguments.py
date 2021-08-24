from argparse import ArgumentParser

def new_arguements(parser: ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--T', type=int, default=100, help='Communication rounds')
    parser.add_argument('--K', type=int, default=50, help='Local iterations')
