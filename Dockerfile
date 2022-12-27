FROM lung-csp:base

WORKDIR /root


COPY scripts /root/csp/scripts/

COPY train_caltech.py /root/csp/

COPY test_caltech.py /root/csp/

COPY run.sh /root