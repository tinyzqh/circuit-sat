# The training script for tiny dataset: SR3 actually.
python src/train.py --data-type  AIG  \
                --model DVAEdgeEncoder \
                --train-data sr3to10 \
                --test-data sr10 \
                --nvt 4 \
                --save-interval 100 \
                --hs 64 \
                --gs 64 \
                --bidirectional \
                --lr 1e-4 \
                --epochs 50 \
                --n-rounds 20 \
                --batch-size 128 \
                --no-invert

