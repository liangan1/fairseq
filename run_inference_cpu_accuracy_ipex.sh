#!/bin/sh

#!/bin/sh

###############################################################################
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export LD_PRELOAD=~/.local/lib/libtcmalloc.so

ARGS=""
if [[ "$1" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

data_type=$2

echo "$data_type"

if [[ "$2" == "bf16" ]]
then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

MAX_TOKEN=4096
DATASET=../wmt17_en_de

if [ -e $MODEL ]; then
  echo "### $DATASET found..."
else
  echo "### $DATASET doesn't exist, prepare to download..."
fi

fairseq-train $DATASET --arch transformer_wmt_en_de_big --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4     --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $MAX_TOKEN --ipex     --dnnl --validate-training-performance --performance-begin-its 30 --performance-its-count 50 2>&1|tee training_max_token_$MAX_TOKEN_ipex_mkldnn.log
