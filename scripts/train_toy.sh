# The training script for tiny dataset: SR3 actually.
python src/train.py --data-name  AIG  \
                --model DVAEncoder \
                --data-name sr3 \
                --nvt 4 \
                --save-interval 100 \
                --hs 512 \
                --nz 64 \
                --bidirectional \
                --lr 1e-4 \
                --epochs 300 \
                --batch-size 1024

