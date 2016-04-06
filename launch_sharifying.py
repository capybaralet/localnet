
import numpy
np = numpy
import os

from speechgeneration.wordembeddings.launchers.job_launchers import launch

filename = os.path.basename(__file__)[:-3]
script_path = "./dk_localnet.py"

jobs = []
for lr in [.01]:
    for init_scale in [.01]:
        for sharify_every_n_batches in [1,5,25,125]:
            for architecture in 'lenet':
                cmd_line_args = []
                cmd_line_args.append(['sharify_every_n_batches', sharify_every_n_batches])
                cmd_line_args.append(['lr', lr])
                cmd_line_args.append(['init_scale', init_scale])
                cmd_line_args.append(['architecture', architecture])
                jobs.append((script_path, cmd_line_args))

gpus = range(8)
ngpu = gpus[0]
print "njobs =", len(jobs)
print "ngpus =", len(gpus)
print jobs

launch(jobs, gpus, filename)

# log this launch
os.system("touch " + os.path.join(os.environ["SAVE_PATH"], "logging", "launched___" + filename))



