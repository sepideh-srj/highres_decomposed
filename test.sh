#!/bin/bash
#SBATCH -J test_highres     # Name that will show up in squeue
#SBATCH --gres=gpu:1    # Request 1 GPU "generic resource"
#SBATCH --time=6-22:00       # Max job time is 7 days
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --qos=overcap
#SBATCH -w cs-venus-04

# The SBATCH directives above set options similarly to command line arguments to srun
# Run this script with: sbatch my_experiment.sh
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue
	
# Your experiment setup logic here
source /home/ssarajia/miniconda3/etc/profile.d/conda.sh
conda activate pix2pix_new
hostname
echo ------------Starting Training---------
echo $CUDA_AVAILABLE_DEVICES
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  highRes_AtoB_full_fake_third_bad --model pix2pix --direction AtoB --display_id -1 --dataset_mode highres --ratio 1   --all_fake --third_bad
srun --qos=overcap python test.py --dataroot /project/aksoy-lab/Sepideh/data_big --name highRes_BtoA_wgangp_noratio_third_bad --model pix2pix --midas 0 --direction BtoA --dataset_mode highres --random 1 --ratio 0 --all_fake --netG unet_256 --results_dir results_BtoA --epoch 170
#srun --qos=overcap python train.py --dataroot /project/aksoy-lab/Sepideh/data_big --name  highRes_AtoB_full_highres_noratio_wggp_resnet6 --model pix2pix --direction AtoB --display_id -1 --dataset_mode highres --ratio 0 --all_fake --gan_mode wgangp --netG resnet_6blocks 
