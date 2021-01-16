# train
DIR_TRAIN="data/dimacs/train/sr3to10"
DIR_LOG="log/gen_sr3to10_train.log"
rm -rf ${DIR_TRAIN}
mkdir -p ${DIR_TRAIN}

python src/gen_sr_dimacs.py ${DIR_TRAIN} ${DIR_LOG} 300000 --min_n 3 --max_n 10

# validation
DIR_VAL="data/dimacs/validation/sr10"
DIR_LOG="log/gen_sr10_validation.log"
rm -rf ${DIR_VAL}
mkdir -p ${DIR_VAL}

python src/gen_sr_dimacs.py ${DIR_VAL} ${DIR_LOG} 10000 --min_n 10 --max_n 10


