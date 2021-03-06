# Combine all steps of data generation into one single script
N_TRAIN_PAIRS=300000
N_TEST_PAIRS=10000
# train
DIR_TRAIN="data/dimacs/train/sr3to10"
DIR_LOG="log/gen_sr3to10_train.log"
rm -rf ${DIR_TRAIN}
mkdir -p ${DIR_TRAIN}

python src/gen_sr_dimacs.py ${DIR_TRAIN} ${DIR_LOG} ${N_TRAIN_PAIRS} --min_n 3 --max_n 10

# validation
DIR_VAL="data/dimacs/validation/sr10"
DIR_LOG="log/gen_sr10_validation.log"
rm -rf ${DIR_VAL}
mkdir -p ${DIR_VAL}

python src/gen_sr_dimacs.py ${DIR_VAL} ${DIR_LOG} ${N_TEST_PAIRS} --min_n 10 --max_n 10


# train
DIR_DIMACS="data/dimacs/train/sr3to10"
DIR_AIG="data/aig_raw/train/sr3to10"
DIR_LOG="log/dimacs2aig_sr3to10_train.log"
rm -rf ${DIR_AIG}
mkdir -p ${DIR_AIG}

python src/dimacs2aig.py ${DIR_DIMACS} ${DIR_AIG} ${DIR_LOG}

# validation
DIR_DIMACS="data/dimacs/validation/sr10"
DIR_AIG="data/aig_raw/validation/sr10"
DIR_LOG="log/dimacs2aig_sr10_validation.log"
rm -rf ${DIR_AIG}
mkdir -p ${DIR_AIG}

python src/dimacs2aig.py ${DIR_DIMACS} ${DIR_AIG} ${DIR_LOG}



# train
DIR_AIG="data/aig_raw/train/sr3to10"
DIR_AIG_ABC="data/aig_abc/train/sr3to10"
DIR_LOG="log/aig2aig_sr3to10_train.log"
rm -rf ${DIR_AIG_ABC}
mkdir -p ${DIR_AIG_ABC}

python src/aig2abcaig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}

# validation
DIR_AIG="data/aig_raw/validation/sr10"
DIR_AIG_ABC="data/aig_abc/validation/sr10"
DIR_LOG="log/aig2aig_sr10_validation.log"
rm -rf ${DIR_AIG_ABC}
mkdir -p ${DIR_AIG_ABC}

python src/aig2abcaig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}




# train
DIR_AIG_ABC="data/aig_abc/train/sr3to10"
DIR_IGRAPG="data"
DATASET_NAME="sr3to10_train"


python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME}

# validation
DIR_AIG_ABC="data/aig_abc/validation/sr10"
DIR_IGRAPG="data"
DATASET_NAME="sr10_validation"

python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME}




