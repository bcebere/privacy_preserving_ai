#!/bin/bash

function echo_file {
  str=$1
  echo "$str"
  python3 -m yapf -i "$str"
}


if [ -n "$ONLY_MODIFIED" ]; then
    files=$(git status --porcelain | awk '{print $2}' | grep -E "\.(py)$")
else
    files=`find . -name "*\.py" | sort`
fi

echo $files
for f in $files; do
  echo_file $f
done

