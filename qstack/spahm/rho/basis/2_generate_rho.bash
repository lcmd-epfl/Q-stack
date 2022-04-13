#!/usr/bin/env bash

cat xyz.dat | while read mol charge spin a1 a2 ; do
  ./2_generate_rho.py xyz/${mol} dm/${mol}.dm_bond.npy minao ${mol}.rho_bond.npz ${a1} ${a2}
done
