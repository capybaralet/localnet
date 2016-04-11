
import numpy
np = numpy
import os

from speechgeneration.wordembeddings.launchers.job_launchers import launch

filename = os.path.basename(__file__)[:-3]
script_path = "./dk_localnet.py"

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp", type=str, dest='exp', default='MNIST')
args_dict = vars(parser.parse_args())
locals().update(args_dict)

dataset = exp
if exp == "MNIST":
    nets = ["LeNet"]
elif exp == "CIFAR10":
    nets = ["AlexNet"]

print "launching", nets, exp

jobs = []
for lr in [.01]:
    for init_scale in [.01]:
        for tie_every_n_batches in [1,2,5,10,25,50,100,500]:
            for net in nets:
                cmd_line_args = []
                cmd_line_args.append(['tie_every_n_batches', tie_every_n_batches])
                cmd_line_args.append(['lr', lr])
                cmd_line_args.append(['init_scale', init_scale])
                cmd_line_args.append(['net', net])
                cmd_line_args.append(['dataset', dataset])
                jobs.append((script_path, cmd_line_args))

gpus = range(8)
ngpu = gpus[0]
print "njobs =", len(jobs)
print "ngpus =", len(gpus)
print jobs

launch(jobs, gpus, filename)

# log this launch
os.system("touch " + os.path.join(os.environ["SAVE_PATH"], "logging", "launched___" + filename))



