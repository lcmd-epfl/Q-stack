#!/usr/bin/bash

echo "def2_orca_l_order = {"
for BASIS in 'def2-mSVP' 'def2-mTZVP' 'def2-mTZVPP' 'def2-SV(P)' 'def2-SVP' 'def2-TZVP(-f)' 'def2-TZVP' 'def2-TZVPP' 'def2-QZVP' 'def2-QZVPP' 'def2-SVPD' 'def2-TZVPD' 'def2-TZVPPD' 'def2-QZVPD' 'def2-QZVPPD'; do

  echo "'${BASIS}':{"

  if echo 'def2-SVPD' 'def2-TZVPD' 'def2-TZVPPD' 'def2-QZVPD' 'def2-QZVPPD' | grep -w -q ${BASIS}; then
    TEMPLATE=periodic.template2.inp
  else
    TEMPLATE=periodic.template.inp
  fi
  TMP1=$(mktemp)
  sed s/BASIS/${BASIS}/ ${TEMPLATE} > ${TMP1}.inp

  timeout 1 orca ${TMP1}.inp > ${TMP1}.out 2> /dev/null

  TMP2=$(mktemp)
  sed -n '/BASIS SET IN INPUT FORMAT/,/ECP PARAMETER INFORMATION/p' ${TMP1}.out | head -n -3 | tail -n +4 > ${TMP2}
  csplit ${TMP2} --silent --prefix ${TMP2} --suppress-matched '/^$/' {*}

  for FILE in ${TMP2}??; do
    ELEMENT=$(head ${FILE} -n 1 | cut -f8 -d' ')
    ANGULAR_MOMENTA=$(cat ${FILE} | head -n -1 | tail -n +3 | grep "[A-Z]" | cut -f2 -d' ' | tr -s 'SPDFGH' '012345')
    if ! echo "${ANGULAR_MOMENTA}" | sort --check=silent ; then
      echo "    '${ELEMENT}'" : "($(echo "${ANGULAR_MOMENTA}" | tr -s '\n' ',')),"
    fi
  done

  echo '},'
  echo
done
echo '}'
