#!/bin/bash
for filename in "."/*.csv
do
	echo "$filename"
	awk -F "," '{ print $1 "," $2 }' "$filename" > "$filename.val.csv"
	rm -f "$filename"
done
