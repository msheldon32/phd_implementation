#!/bin/bash
echo "run,n_jobs,tput" > single_class_combined_1.csv
cat single_class_output_1/* >> single_class_combined_1.csv
