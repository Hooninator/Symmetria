# Build Instructions #
This section contains instructions on how to build this project.

## Software Dependencies ##
Several external pieces of software are required in order to build this project.

- CombBLAS (https://github.com/tf-mac/CombBLAS)
- GALATIC (included in CombBLAS)
- NVSHMEM 3.0.6 (https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html) 
- json (https://github.com/nlohmann/json/tree/master) 

Build instructions for each of these libraries can be found in their respective repositores/installation guides, which I have linked.

Make sure each of the previously listed libraries are installed in a directory called `SpSYRK/deps`.

Aside from these, ensure your system has the following pieces of software installed before attempting to build Symmetria.

- cudatoolkit 12.4
- cmake 3.30.2
- GCC 12.3
- CrayMPICH 8.1.28

Additionally, in order to run Symmetria across multiple GPUs/nodes, it is necessary to have a job manager like SLURM installed.
The scripts and instructions in this repo require a system with SLURM installed, so please make sure you have SLURM on your system.

## Building Symmetria ##
In order to build Symmetria, run these commands.

`export COMBBLAS\_HOME=/path/to/combblas/install`

`mkdir build && cd build`

`cmake ..`

`make -j`

# Running Symmetria #

## Reproducing Results ##

First, `cd experiments`.

The raw FASTA file used to produce the plots in the report can be obtained using `wget https://portal.nersc.gov/project/m1982/bella/celegans20x_0001.fasta`.

To process this file using ELBA, `cd` into the `deps/ELBA/build` directory and run the following

`srun -n 16 ./elba -i /path/to/celegans20x_0001.fasta -k 17 --idxmap elba-celegans -c 670567 --alph dna --of overlap-celegans --af alignment-celegans -s 1 -O 100000 --afreq 100000 --ca 5`

This will create a matrix in matrix market format called `celegans20x_0001.fasta.mtx`. Note that this repo already contains this matrix in `experiments/celegans/celegans.mtx`, in compliance with the instructions for the project. 

If you would like to replace this provided matrix with the equivalent one you just generated, please `mkdir celegans && mv celegans20x_0001.fasta.mtx celegans/celegans.mtx`.

Finally, run `mkdir timings` from the `experiments` directory.

Now, all is in place to run the experiments.

From `experiments`, run `sbatch run.sbatch`. This will enqueue a job that will use both SpSYRK algorithms and CombBLAS's SpGEMM routine in order to perform similarity detection on the `celegans` dataset.

Once the job has finished running, timing data will be written to `experiments/timings` in the form of CSV files.
If you want to remake the plots in the paper, run `mkdir plots`, then open and run the cells in the provided `plotting.ipynb` Jupyter Notebook.

