import os
while 1:
    with open('experiment_list.txt') as nf:
        all_experiments_left = nf.readlines()
        if not all_experiments_left:
            break
        first_experiment = all_experiments_left[0]
        with open('experiment_list.txt','w') as f:
            f.writelines(all_experiments_left[1:])
        os.system("nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9")
        os.system(first_experiment)

