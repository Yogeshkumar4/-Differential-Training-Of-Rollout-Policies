
import argparse

def str2bool(v):
    # https://stackoverflow.com/a/43357954/2570622
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ACTIONS = ['up', 'down', 'left', 'right']