# sample rotations around the z-axis of 1.xyz (36 points):
./1_generate_rot.py 1.xyz 36
mkdir 36/
mv 1.xyz.* 36

# compute all the DF coefficients:
for i in 36/*.xyz.?? ; do ./2_fitting.py $i ; done

# compute the overlap of the density
# attributed to the 1st atom of 1.xyz and the 1st atom of 2.xyz
# averaged over rotation around the z-axis
./3_overlap.py 36/ 2.xyz

