#!/bin/bash

xyz='../H2O_dist_rot.xyz'
x2t ${xyz} > coords

for basis in 'cc-pVDZ' 'cc-pVTZ' 'cc-pVQZ'; do

rm control basis -f
define << EOF


a coords
*
no
b all ${basis}
*
eht
y
0
y
q
EOF

basis_out=$(echo ${basis} | tr [:upper:] [:lower:] | tr -d '-')
dscf > ${basis_out}.out
mv mos mos-${basis_out}

done



basis='cc-pVDZ'

rm control basis -f
define << EOF


a coords
*
no
b all ${basis}
*
eht
y
1
y
y
q
EOF

basis_out=$(echo ${basis} | tr [:upper:] [:lower:] | tr -d '-')
dscf > ${basis_out}-cation.out
