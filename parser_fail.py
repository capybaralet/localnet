
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--test", type=int, dest='test', help="WHY NO DEFAULT???", default=1)
parser.parse_args()
args_dict = vars(parser.parse_args())
locals().update(args_dict)

print test

