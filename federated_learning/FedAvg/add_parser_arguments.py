from argparse import ArgumentParser

def new_arguements(parser: ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.01)
    pass
