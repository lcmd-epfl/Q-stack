#!/usr/bin/env bash

cat xyz.dat | while read mol charge spin a1 a2 ; do
  ./1_extract_dm_bond.py xyz/${mol} ${charge} ${spin} ${a1} ${a2}
done
