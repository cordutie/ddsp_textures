Run the code in the cluster using this code:

sbatch submit.sh ocean_long_gru_stats

options:

ocean; fire; water; rain

long; short

gru; mlp; stemsgru; stemsmlp

multispec; stats; stems

water + fire long mlp stat

for element in fire water; do for length in long; do for model in mlp; do for type in stats; do sbatch submit.sh ${element}_${length}_${model}_${loss}; done; done; done; done

water + fire short mlp gru stat

for element in fire water; do for length in short; do for model in mlp gru; do for type in stats; do sbatch submit.sh ${element}_${length}_${model}_${loss}; done; done; done; done

for element in fire water; do for length in long short; do for model in gru mlp; do for type in stats multispec; do sbatch submit_trainer_${element}_${length}_${model}_${type}.sh; done; done; done; done

