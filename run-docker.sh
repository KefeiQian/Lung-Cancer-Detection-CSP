 docker run -it --gpus=all -v /mnt/c/Programming/Projects/Lung-Cancer:/root/csp lung-csp:v1 /bin/bash

# cloud service docker build & test
# docker build -t lung-csp:v2.1 .
# docker run -it --gpus=all lung-csp:v2.1 /bin/bash