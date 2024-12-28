#!/bin/bash
#SBATCH --job-name=pred     # Nome del job
#SBATCH --output=job_%j.out               # File di output (%j inserisce il JobID)
#SBATCH --error=job_%j.err                # File di errore (%j inserisce il JobID)
#SBATCH --time=20:00:00                   # Tempo massimo di esecuzione (hh:mm:ss)
#SBATCH --partition=boost_usr_prod        # Partizione su cui lanciare il lavoro (boost_usr_prod = normale, con GPU)
#SBATCH --nodes=1                         # Numero di compute nodes
#SBATCH --ntasks-per-node=1               # Numero di task
#SBATCH --gpus-per-node=1                 # Numero di gpu per nodo
#SBATCH --cpus-per-task=32                # Numero di CPU per task
#SBATCH --mem=400GB                       # Memoria per nodo (0 = richedi tutta la memoria?)
#SBATCH --account=DTbo_DTBO-HPC           # Account da cui prendere le ore di CPU

echo "Running python script..."

srun python -u main.py --predict_labels --file_to_predict 'data/chosen_tiles/32_687000_4930000_FP21.las'  --load_model_filepath 'models/saved/mcnn_model_20241226_223628/model.pth' --batch_size 32 --num_workers 32 

echo "Predicted labels on file: data/chosen_tiles/32_687000_4930000_FP21.las w/ last model (probably overfitting)"