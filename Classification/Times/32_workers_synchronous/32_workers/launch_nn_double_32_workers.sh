module purge
module load dislib/FVN
module load intel mkl python
module load COMPSs/3.3.2
module load intel/2023.2.0 mkl/2023.2.0 oneapi/2023.2.0 impi/2021.10.0 ucx/1.15.0
module load sqlite3/3.45.2
# Add python packages

export ComputingUnits=1
export ComputingUnitsGPUs=1

enqueue_compss --log_level=off \
       --project_name=bsc19 --qos=acc_bsccs \
       --master_working_dir=${PWD} \
       --worker_working_dir=${PWD} \
       --log_dir=${PWD} \
       --network=ethernet \
       --job_dependency=24546060 \
       --gen_coredump \
       --exec_time=70 \
       --pythonpath=$PWD:/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/NFNet-F0_measure_time/synchronous_times \
       --worker_in_master_cpus=0 --max_tasks_per_node=80 --num_nodes=9 \
       distributed_training_double_writing_csv_32_workers_double_size_tensors.py

