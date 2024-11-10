#!/bin/bash
#SBATCH --job-name=CNN_inference_try      # Nome del job
#SBATCH --output=job.out                  # File di output (%j inserisce il JobID)
#SBATCH --error=job.err                   # File di errore (%j inserisce il JobID)
#SBATCH --time=06:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=250GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

# Carica i moduli necessari (eventualmente usa cineca-ai in futuro, c'è già cuda)
module load cuda/12.2

srun python main.py --load_model --load_model_filepath 'models/saved/mcnn_model_20241030_051517/model.pth' --batch_size 32 --num_workers 16
