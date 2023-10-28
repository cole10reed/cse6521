/* Login */

ssh [username]@[cluster].osc.edu	// I have been using the Owens cluster, but the different clusters have different GPUs.

/* Once you've logged in */

sinteractive	-A pas2622
		-N [number of nodes to request]
		-n [number of tasks to request]
		-c [number of CPU cores to request]
		-m [memory per CPU]
		-M [memory per node]
		-g [number of GPUs to request]
		-G [GRES to request]
		-L [Licenses to request]
		-t [time limit (minutes)]
		-J [job name (default: interactive)]
		-w [node name]

sinteractive -A pas2622 -c 4 -g 1 -t 30		// this is what I have been using

/* After your node(s) has been allocated */

module load python/3.9-2022.05 cuda/11.8.0
module list

conda create -n sam	// only need to do this command on your first time
source activate sam
conda install pip	// only once
pip install [all requirements]	//only once

