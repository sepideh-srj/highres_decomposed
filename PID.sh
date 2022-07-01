#!/bin/bash
#SBATCH -J FID_compute     # Name that will show up in squeue
#SBATCH --time=6-22:00       # Max job time is 7 days
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --cpus-per-task=16
#SBATCH --qos=overcap

source /home/ssarajia/miniconda3/etc/profile.d/conda.sh
conda activate pix2pix
hostname
echo ------------compute FID---------
echo $CUDA_AVAILABLE_DEVICES

srun --qos=overcap python prepare_fid.py --dir results_val/highRes_AtoB_wgangp_noratio_third_bad_all_blurfullSize
echo highFake highRes_AtoB_full_highres_wgangp_noratio_resnet9
srun --qos=overcap python -m  pytorch_fid results_val/highRes_AtoB_wgangp_noratio_third_bad_all_blurfullSize/real results_val/highRes_AtoB_wgangp_noratio_third_bad_all_blurfullSize/highFake/ --device cuda:0 

echo lowFake highRes_AtoB_full_highres_wgangp_noratio_resnet9
srun --qos=overcap python -m  pytorch_fid results_val/highRes_AtoB_wgangp_noratio_third_bad_all_blurfullSize/real results_val/highRes_AtoB_wgangp_noratio_third_bad_all_blurfullSize/lowFake

