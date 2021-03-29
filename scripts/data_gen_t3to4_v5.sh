# # Combine all steps of data generation into one single script
# # train
# DIR_TRAIN="data/dimacs/train/sr3to4"
# DIR_LOG="log/gen_sr3to4_train.log"
# rm -rf ${DIR_TRAIN}
# mkdir -p ${DIR_TRAIN}

# python src/gen_sr_dimacs.py ${DIR_TRAIN} ${DIR_LOG} 10000 --min_n 3 --max_n 4

# # validation
# DIR_VAL="data/dimacs/validation/sr5"
# DIR_LOG="log/gen_sr5_validation.log"
# rm -rf ${DIR_VAL}
# mkdir -p ${DIR_VAL}

# python src/gen_sr_dimacs.py ${DIR_VAL} ${DIR_LOG} 1000 --min_n 5 --max_n 5


# # train
# DIR_DIMACS="data/dimacs/train/sr3to4"
# DIR_AIG="data/aig_raw/train/sr3to4"
# DIR_LOG="log/dimacs2aig_sr3to4_train.log"
# rm -rf ${DIR_AIG}
# mkdir -p ${DIR_AIG}

# python src/dimacs2aig.py ${DIR_DIMACS} ${DIR_AIG} ${DIR_LOG}

# # validation
# DIR_DIMACS="data/dimacs/validation/sr5"
# DIR_AIG="data/aig_raw/validation/sr5"
# DIR_LOG="log/dimacs2aig_sr5_validation.log"
# rm -rf ${DIR_AIG}
# mkdir -p ${DIR_AIG}

# python src/dimacs2aig.py ${DIR_DIMACS} ${DIR_AIG} ${DIR_LOG}



# # train
# DIR_AIG="data/aig_raw/train/sr3to4"
# DIR_AIG_ABC="data/aig_abc/train/sr3to4"
# DIR_LOG="log/aig2aig_sr3to4_train.log"
# rm -rf ${DIR_AIG_ABC}
# mkdir -p ${DIR_AIG_ABC}

# python src/aig2abcaig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}

# # validation
# DIR_AIG="data/aig_raw/validation/sr5"
# DIR_AIG_ABC="data/aig_abc/validation/sr5"
# DIR_LOG="log/aig2aig_sr5_validation.log"
# rm -rf ${DIR_AIG_ABC}
# mkdir -p ${DIR_AIG_ABC}

# python src/aig2abcaig.py ${DIR_AIG} ${DIR_AIG_ABC} ${DIR_LOG}




# train
DIR_AIG_ABC="data/aig_abc/train/sr3to4"
DIR_IGRAPG="data"
DATASET_NAME="sr3to4_train_NE"
DIR_SOLUTION="data/dimacs/train/sr3to4"


python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME} --aig_solution_dir ${DIR_SOLUTION}

# validation
DIR_AIG_ABC="data/aig_abc/validation/sr5"
DIR_IGRAPG="data"
DATASET_NAME="sr5_validation_NE"
DIR_SOLUTION="data/dimacs/validation/sr5"


python src/aig_parser.py ${DIR_AIG_ABC} ${DIR_IGRAPG} ${DATASET_NAME} --aig_solution_dir ${DIR_SOLUTION}




