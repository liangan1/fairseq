###
### script for Transformer-LT trianing for cpu
###
### 1. install:
###   pip install --editatable .
###  
###
### 2. use ipex and dnnl for FP32:
###   ./run_training_cpu_performance_multi_instance.sh ipex dnnl 


NUM_CORES=`lscpu | grep Core | awk '{print $4}'`
NUM_THREAD=$NUM_CORES
NUM_NUMA=$((`lscpu | grep 'NUMA node(s)'|awk '{print $3}' ` - 1))
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export LD_PRELOAD=~/.local/lib/libtcmalloc.so

ARGS=""
if [[ "$1" == "ipex" ]]
then
   ARGS="$ARGS --ipex"
   echo "### Enable Intel Pytorch Extension" 
fi

if [[ "$2" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

data_type=$3

echo "$data_type"

if [[ "$2" == "bf16" ]]
then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
fi


DATASET=/lustre/dataset/wmt17_en_de/
MAX_TOKEN=4096

if [ -e $DATASET ]; then
  echo "### $DATASET found..."
else
  echo "### $DATASET doesn't exist, prepare to download..."
fi

echo -e "### using OMP_NUM_THREADS=$NUM_THREAD"
echo -e "### using $KMP_AFFINITY"
echo -e "### using ARGS=$ARGS\n"

INST_PER_NODE=$(($NUM_CORES / $NUM_THREAD - 1 ))
startid=0
endid=$NUM_THREAD
OMP_NUM_THREADS=$NUM_THREAD numactl --physcpubind=$startid-$endid --membind=0 \
fairseq-train $DATASET \
   --arch transformer_wmt_en_de_big --share-decoder-input-output-embed --optimizer adam \
   --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4     --lr-scheduler inverse_sqrt \
   --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
   --label-smoothing 0.1 --max-tokens $MAX_TOKEN $ARGS \
   2>&1 |tee training_cpu_max_token_${MAX_TOKEN}_accuracy.log 

