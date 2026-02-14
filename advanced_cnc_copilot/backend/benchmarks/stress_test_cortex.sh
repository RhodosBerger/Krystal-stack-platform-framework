#!/bin/bash
echo "Starting Cortex Stress Test..."

# 1. Start Sysbench (CPU Load) in background
echo "Generating Synthetic Load (Sysbench)..."
sysbench cpu --cpu-max-prime=10000 --threads=4 run > sysbench_output.log &
SYSBENCH_PID=$!

# 2. Query Cortex Status while under load
echo "Querying Cortex API under load..."
for i in {1..5}; do
    curl -s http://localhost:5000/cortex/status
    sleep 1
done

# 3. Cleanup
echo "Stopping Load..."
kill $SYSBENCH_PID
echo "Test Complete. Check sysbench_output.log for raw metrics."
