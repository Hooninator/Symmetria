#!/usr/bin/bash

# Args
action="null"
matname="null"
nnz="null"
rows="null"
cols="null"
ntrials=1
nodes=$SLURM_NNODES


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            action="$2"
            shift 2
            ;;
        --matname)
            matname="$2"
            shift 2
            ;;
        --ntrials)
            ntrials="$2"
            shift 2
            ;;
        --nodes)
            nodes="$2"
            shift 2
            ;;
        --nnz)
            nnz="$2"
            shift 2
            ;;
        --rows)
            rows="$2"
            shift 2
            ;;
        --cols)
            cols="$2"
            shift 2
            ;;
        --type)
            type="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--action VALUE] [--matname VALUE] [--ntrials VALUE] [--nodes VALUE]"
            exit 0
            ;;
        *)
            echo "Unknown type: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

if [ "$action" = "null" ]; then
	echo "Error, need to pass --action arg"
    exit;
fi

if [ "$matname" = "null" ]; then
	echo "Error, need to pass --matname arg"
    exit;
fi

if [ "$nnz" = "null" ]; then
	echo "Error, need to pass --nnz arg"
    exit;
fi

if [ "$rows" = "null" ]; then
	echo "Error, need to pass --rows arg"
    exit;
fi

if [ "$cols" = "null" ]; then
	echo "Error, need to pass --cols arg"
    exit;
fi

# make the command
cmd="srun -N $nodes --gpus-per-node 4 --tasks-per-node 4 ./$action --name $matname --ntrials $ntrials --type $type --nnz $nnz --rows $rows --cols $cols"

# Make logfile
touch log.out

echo "============= $(date) =============" >> log.out
echo "    COMMAND: $cmd" >> log.out
echo "    ACTION: $action" >> log.out
echo "    TYPE: $type" >> log.out
echo "    MATRIX: $matname" >> log.out
echo "    NNZ: $nnz" >> log.out
echo "    ROWS: $rows" >> log.out
echo "    COLS: $cols" >> log.out
echo "    NODES: $nodes" >> log.out
echo "    NTRIALS: $ntrials" >> log.out

cat log.out


# Run the thing
cd ../build
echo "Running ${cmd}"
$cmd

if [[ $? -ne 0 ]]; then
    echo "Error: Returned nonzero exit code $?"
    rm log.out
    exit
fi


# Move timing csvs to experiments directory
mv timings_* ../experiments/timings

# Make timing directory for this experiment run
cd ../experiments/timings
dirname="timings_${action}_${type}_${matname}_${nodes}Nodes"

if [[ -e $dirname ]]; then
    rm -rf $dirname
fi

mkdir $dirname

# Move timing csvs into timing directory
mv *.csv $dirname

# Move logfile
mv ../log.out $dirname
cd $dirname

echo "Done!"

