batch_size = 8
n_gpus = 8
num_workers_per_gpu = 2 
exp_prefix = "exp"

fp_loss = True 
frustum_size = 8 
CE_ssc_loss = True
sem_scal_loss = True
geo_scal_loss = True

lr = 4e-4
weight_decay = 0.01
max_epochs = 30
num_samples_per_gpu = 1
print_freq = 50
grad_max_norm = 30
enable_log = True