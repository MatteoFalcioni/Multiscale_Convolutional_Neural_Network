#!/bin/bash
#SBATCH --job-name=2train      # Nome del job
#SBATCH --output=job_%j.out               # File di output (%j inserisce il JobID)
#SBATCH --error=job_%j.err                # File di errore (%j inserisce il JobID)
#SBATCH --time=17:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=400GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

echo "Running python script..."

srun python -u main.py --dataset_filepath 'data/training_data/21/train_21.csv' --training_data_filepath 'data/training_data/21/train_21.csv' --batch_size 32 --num_workers 32 --window_sizes '[5, 10, 20]' --features_to_use intensity red green blue nir delta_z l1 l2 l3 theta theta_variance --epochs 10 --patience 2

echo "training model on **train 21** with window sizes [5, 10, 20] and features: [intensity red green blue nir delta_z l1 l2 l3 theta theta_variance], to see if the best model predicts bretter with smaller windows"