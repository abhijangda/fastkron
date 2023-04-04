# Kronecker GPU 


### Runnning SGKIGP 

* Set up the data using get_data.sh and set UCI_PATH experiments/data.py at line 16.
* Set up the environment or install required packages. For reference, sgkigp_requirement is the pip list from the environment where the code has been tested. I've also added requirements_sgkigp.tx which may not be complete.
* Understanding the run command: 
```
/home/myadav_umass_edu/anaconda3/envs/sgkigp/bin/python -m experiments.runner --dataset houseelectric --epochs 50 --method 0 --grid_size_dim 3 --boundary_slack 0.005 --interp_type 2 --log_dir /home/myadav_umass_edu/sgkigp/logs/sweep-dense-simplex-v2/dataset_houseelectric_gsd_5_bs_0.005
# Explaination of command:
/home/myadav_umass_edu/anaconda3/envs/sgkigp/bin/python -m experiments.runner # run only in module mode 
--dataset houseelectric 
--epochs 50 
--method 0 # this shall be fixed
--grid_size_dim 3 # this you can vary, this tells how many points per dimension 
--boundary_slack 0.005 
--interp_type 2 # this makes 
--log_dir /home/myadav_umass_edu/sgkigp/logs/sweep-dense-simplex-v2/dataset_houseelectric_gsd_5_bs_0.005 # any directory path 
```
* 
* Running the code for a simple case and test if you can run it for both algorithms of SKI that we care about. 
```
 /home/myadav_umass_edu/anaconda3/envs/sgkigp/bin/python -m experiments.runner --dataset servo --epochs 50 --method 0 --grid_size_dim 5 --boundary_slack 0.005 --interp_type 2 --log_dir /home/myadav_umass_edu/sgkigp/logs/
```

#### Expected result that wandb should spit. As long as results are in same ballpark we are good. 
```
wandb: Run summary:
wandb:      grid_size 0
wandb:  test/best_nll 0.0
wandb: test/best_rmse 0.28366
wandb:       test/mae 0.18709
wandb:       test/nll 0.0
wandb:   test/pred_ts 0.008
wandb:      test/rmse 0.28043
wandb:    train/bw_ts 0.01197
wandb:  train/loss_ts 0.06621
wandb:      train/mll -0.51677
wandb: train/total_ts 0.07818
wandb:  val/best_step 49
wandb:        val/mae 0.13593
wandb:        val/nll 0.0
wandb:    val/pred_ts 0.35376
wandb:       val/rmse 0.19558
```


### Experimental protocol
* Goal: record inf-time, train-time, rmse, maximum memory usage for big datasets. 
* Choice of datasets: houseelectric 3droad protein. For all these, three commands can be obtained using skisimplex.sh script.
  * Details about these datasets can be found in uci_datasets.ipynb, e.g., number of points and dimensions. 
* Obtaining results: 
  * Try to obtain a reasonably large value of grid_size_dim for houseelectric and fix it for all three datasets. 
  * Run for all three datasets. If they are taking too much time, feel free to reduce the number of epochs. 

