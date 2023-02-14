#for subset in 0 1 2 3 4 5 6 7 8 9; do
for subset in 9; do
    echo "subset: $subset"

    echo "preprocessing..."
    python3.10 preprocessing.py $subset
    echo ""

    echo "augment gts..."
    python3.10 gt-augment.py
    echo ""

    cd ..

    echo "training..."
    python2.7 train_caltech.py
    echo ""

    echo "testing..."
    python2.7 test_caltech.py
    echo ""

    mv output "output-ss$subset"
    cd work
done