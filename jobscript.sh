#!/bin/bash
#SBATCH --job-name=train      # Nome del job
#SBATCH --output=job_%j.out               # File di output (%j inserisce il JobID)
#SBATCH --error=job_%j.err                # File di errore (%j inserisce il JobID)
#SBATCH --time=12:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=400GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

echo "Running python script..."

srun python -u main.py --dataset_filepath 'data/datasets/sampled_data_5251681.csv' --training_data_filepath 'data/datasets/train_dataset.csv' --evaluate_model_after_training --evaluation_data_filepath 'data/datasets/eval_dataset.csv' --batch_size 32 --num_workers 32 --window_sizes '[2.5, 5, 10]' --features_to_use intensity --epochs 10 --patience 2
