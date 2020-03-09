#$ -S /bin/bash
#$ -N thimble_Nt_1
#$ -l h_rt=10:00:00
#$ -l h_vmem=800m
#$ -l h_fsize=500m
#$ -q run64new

export OMP_NUM_THREADS=8

cp /lpt/jquarroz/python/phi4mu/phi4mu.py .

mkdir ./data

python3 phi4mu.py $1 $2

cp ./data/config_mu_${1}_T_${2}.txt /lpt/jquarroz/python/phi4mu/data/
