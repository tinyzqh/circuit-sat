# The training script for tiny dataset: SR10/SR3 actually.
python src/train.py --data-name  AIG  \
                --igraph-dir data \
                --data-name sr10 \
                --nvt 4 \
                --save-interval 100 \
                --no-test \
                --small-train \
                --model DVAEncoder \
                --hs 128 \
                --nz 64 \
                --bidirectional \
                --lr 1e-4 \
                --epochs 300 \
                --batch-size 32

