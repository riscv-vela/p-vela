#!/bin/bash

#setup
CPU_ID=2,3,4,5
INTERVAL=1 # seconds
OUTPUT_FILE="turbostat.csv"

echo "--- turbostat with epoch logging ---"

# Outputfile header
echo "epoch_ns,Timestamp,Core,CPU,Avg_MHz,Busy%,Bzy_MHz,TSC_MHz,IPC,IRQ,POLL,C1,C2,C3,POLL%,C1%,C2%,C3%,CorWatt,PkgWatt" > $OUTPUT_FILE

turbostat --cpu $CPU_ID --interval $INTERVAL | \
while read -r line; do
    csv_line=$(echo "$line" | tr -s '\t' ',')

    if [[ "$csv_line" =~ ^[-0-9] ]]; then
        epoch_ns=$(date +%s%N)
        ts=$(date +"%T.%3N")
        echo "$epoch_ns,$ts,$csv_line" >> $OUTPUT_FILE
    fi
done
