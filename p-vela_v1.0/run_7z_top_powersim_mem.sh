#!/bin/bash

OUT_DIR="/mnt/host/4_core/7z"
OUT_PREFIX="${OUT_DIR}/7z"

mkdir -p "$OUT_DIR"

# -------------------------------
# 0) system power simulator
# -------------------------------
python3 /mnt/host/4_core/7z/power_sim_mem.py &
POWER_SIM_PID=$!

# -------------------------------
# 1) START timestamp 기록
# -------------------------------
START_TS=$(date +%s.%N)
echo "[START] $START_TS" > ${OUT_PREFIX}_meta.log

# -------------------------------
# 2) mpstat (per-core) 백그라운드 실행
# -------------------------------
mpstat -P ALL 1 > ${OUT_PREFIX}_mpstat.log &
MPSTAT_PID=$!

# -------------------------------
# 3) top 백그라운드 실행 (전체 MEM/CPU 기록)
# -------------------------------
top -b -d 1 > ${OUT_PREFIX}_top.log &
TOP_PID=$!

sleep 1

# -------------------------------
# 4) Running Benchmark
# -------------------------------
echo "Running 7z benchmark..."

taskset -c 0 7z b & PID0=$!
taskset -c 1 7z b & PID1=$!
taskset -c 2 7z b & PID2=$!
taskset -c 3 7z b & PID3=$!

# 4개 7z 프로세스 모두 대기
wait $PID0 $PID1 $PID2 $PID3

# -------------------------------
# 5) END timestamp 기록
# -------------------------------
END_TS=$(date +%s.%N)
echo "[END] $END_TS" >> ${OUT_PREFIX}_meta.log

# -------------------------------
# 6) background 프로세스 종료
# -------------------------------
kill $MPSTAT_PID
kill $TOP_PID
kill $POWER_SIM_PID

echo "Running complete."
echo "Saved logs to: $OUT_DIR"
