# train
# DIR_AIG_ABC="data/aig_abc/train/sr3to10"
# DIR_IGRAPG="data"
# DATASET_NAME="sr3to10_train"


# python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME}

# validation
DIR_AIG_ABC="data/aig_abc/validation/sr10"
DIR_IGRAPG="data"
DATASET_NAME="sr10_validation"


python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME}


