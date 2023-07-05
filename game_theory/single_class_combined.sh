#!/bin/bash
echo "run,n_jobs,tput" > single_class_combined_2.csv
#find single_class_output_1/* -type f -exec cat {} + >> single_class_combined_1.csv
n_files=100000
for file_idx_start in {1..15000}
do
    cat "single_class_output_2/output_$file_idx_start.csv" >> single_class_combined_2.csv
done
#cat single_class_output_1/* >> single_class_combined_1.csv

