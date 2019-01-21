#!/bin/bash

#SBATCH --time=00:60:00   # walltime
#SBATCH --ntasks=10 # number of processor cores total (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=5   # set 5 processors per node
#SBATCH --mem-per-cpu=4096M   # 4GB memory per CPU core
#SBATCH -J "Spark"   # job name

# Compatibility variables for PBS. spark-beta.py uses this variable to determine which nodes should appear in the cluster.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export HADOOP_GROUP=/fslgroup/fslg_hadoop/
export SPARK_PATH=/fslgroup/fslg_hadoop/spark-2.0.0/
export JAVA_HOME=/usr/java/latest/

module load python/2/7

# If you have 2 Nodes with 3 cores and 2048M per Node (so total of 6 cores), the configuration would be:
#
# ...../spark.py 	3 2048 6 "" "/path/to/my/program.py arg0 arg1"
#
$SPARK_PATH/bin/spark.py 5 4096 10 "" "${HOME}/lab4.py /fslgroup/fslg_hadoop/netflix_data.txt"

exit 0
