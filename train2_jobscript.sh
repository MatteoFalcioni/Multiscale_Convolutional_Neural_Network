#!/bin/bash
#SBATCH --job-name=2train      # Nome del job
#SBATCH --output=job_%j.out               # File di output (%j inserisce il JobID)
#SBATCH --error=job_%j.err                # File di errore (%j inserisce il JobID)
#SBATCH --time=15:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=400GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

echo "Running python script..."

srun python -u main.py --dataset_filepath 'data/training_data/21/train_21.csv' --training_data_filepath 'data/training_data/21/train_21.csv' --batch_size 32 --num_workers 32 --epochs 10 --patience 2 --window_sizes '[10, 20 ,30]' --features_to_use intensity red green blue nir l1 l2 l3

echo "training model w/ intensity, r, g, b, nir, l1 l2 l3 and ws 10,20,30"