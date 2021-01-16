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



