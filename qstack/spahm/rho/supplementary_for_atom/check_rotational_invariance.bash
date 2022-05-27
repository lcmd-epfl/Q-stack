#!/usr/bin/env bash

for model in pure sad-diff occup lowdin-short lowdin-long lowdin-short-x lowdin-long-x ; do

  echo $model
  echo

  for mol in mol/*.xyz ; do
    ../1_DMbRep.py --mol $mol ;
  done > /dev/null

  for r0 in X_mol/X_rot0_mol_????.npy; do
    r1=${r0/rot0/rot1}
    echo $r0
    echo $r1
    python3 ./repdiff.py $r0 $r1
    echo
  done

  echo
  echo

done
