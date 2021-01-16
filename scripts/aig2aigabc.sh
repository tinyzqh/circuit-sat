# train
DIR_AIG="data/aig_raw/train/sr3to10"
DIR_AIG_ABC="data/aig_abc/train/sr3to10"
DIR_LOG="log/aig2aig_sr3to10_train.log"
rm -rf ${DIR_AIG_ABC}
mkdir -p ${DIR_AIG_ABC}

python src/dimacs2aig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}

# validation\
DIR_AIG="data/aig_raw/validation/sr10"
DIR_AIG_ABC="data/aig_abc/validation/sr10"
DIR_LOG="log/aig2aig_sr10_validation.log"
rm -rf ${DIR_AIG_ABC}
mkdir -p ${DIR_AIG_ABC}

python src/dimacs2aig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}


