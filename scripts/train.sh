# shellcheck disable=SC2164
cd csp
python train_caltech.py
# shellcheck disable=SC2164
cd scripts/
python upload_train_result.py