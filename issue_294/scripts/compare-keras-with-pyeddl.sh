#!/bin/bash

k1='logs/report-keras-without-bn.log'
k2='logs/report-keras-with-bn.log'
p1='logs/report-pyeddl-without-bn.log'
p2='logs/report-pyeddl-with-bn.log'


function comparison()
{
    subset="$1"
    bn="$2"
    f1="$3"
    f2="$4"
    echo "# "
    echo "# "
    echo "# Comparing using the training subset ${bn} BatchNormalisation"
    echo "# "
    echo "#     Tensorflow+Keras                        pyeddl "
    echo "# "
    echo "# "
    grep -A3 "Confusion matrix for subset ${subset}" ${f1} | awk '{ printf("%-35s \n", $0) }' >/tmp/f1.txt
    grep -A3 "Confusion matrix for subset ${subset}" ${f2} | awk '{ printf("%-35s \n", $0) }' >/tmp/f2.txt
    paste /tmp/f1.txt /tmp/f2.txt
}

comparison "train" "without"  ${k1} ${p1}
comparison "test"  "without"  ${k1} ${p1}
comparison "train" "with"     ${k2} ${p2}
comparison "test"  "with"     ${k2} ${p2}
