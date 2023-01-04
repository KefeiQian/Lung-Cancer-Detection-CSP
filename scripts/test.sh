# shellcheck disable=SC2164
cd csp
python test_caltech.py
# shellcheck disable=SC2164
cd scripts/
python upload_test_result.py