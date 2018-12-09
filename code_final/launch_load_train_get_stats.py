"""
Launcher program for launch_load_train_get_stats.py
"""
f = open('/homes/murph213/DeepLearning/code_final/launch_load_train_get_stats.sh', 'w')
f.write("cd /homes/murph213/DeepLearning/code_final")
f.write("\n")

for conf in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
    for net in ['pruned']:
        for fold in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            command = 'CUDA_VISIBLE_DEVICES=0 python load_train_get_stats.py '
            command += '--config_integer={} --net_type={} --fold={}'.format(conf, net, fold)
            f.write(command)
            f.write("\n")

f.close()
