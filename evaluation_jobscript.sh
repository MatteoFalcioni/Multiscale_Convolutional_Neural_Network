#!/bin/bash
#SBATCH --job-name=eval     # Nome del job
#SBATCH --output=job_%j.out               # File di output (%j inserisce il JobID)
#SBATCH --error=job_%j.err                # File di errore (%j inserisce il JobID)
#SBATCH --time=01:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=200GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

echo "Running python script..."

srun python -u main.py --perform_evaluation --load_model_filepath 'models/saved/mcnn_model_20241213_002308/model.pth' --dataset_filepath 'data/training_data/21/test_21.csv' --evaluation_data_filepath 'data/training_data/21/test_21.csv' --batch_size 32 --num_workers 32 

echo "Evaluate 'old' model on test_21"