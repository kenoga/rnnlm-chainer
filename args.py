import argparse

parser = argparse.ArgumentParser(description="test")

parser.add_argument("model_file", \
                    action="store", \
                    nargs=None, \
                    default="", \
                    help=""
                    )
parser.add_argument("-e", "--epoch-num", \
                    action="store", \
                    nargs=None, \
                    default="10", \
                    type=int, \
                    help="a number of epoch for NN training."
                    
                    )

parser.add_argument("-s", "--hidden-size", \
                    action="store", \
                    nargs=None, \
                    default=1000, \
                    type=int, \
                    help="a hidden size of RNN."
                    )
parser.add_argument("-b", "--batch-size", \
                    action="store", \
                    nargs=None, \
                    default=100, \
                    type=int, \
                    help="a batch size for NN training." \
                    )
args = parser.parse_args()

print args
