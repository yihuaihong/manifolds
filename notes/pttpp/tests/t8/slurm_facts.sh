#!/bin/bash
# T8: scheduling facts (no results): associations, QOS limits, partitions with their GPU types, and this batch's jobs by state.
set -u
echo "=== date $(date -Is)"
echo "=== associations of yh6210"
timeout 120 sacctmgr -nP show assoc user=yh6210 format=account,partition,qos,defaultqos,grptres,maxtres,maxjobs,maxsubmit 2>&1
echo "=== QOS"
timeout 120 sacctmgr -nP show qos format=name,priority,maxtrespu,maxjobspu,maxsubmitpu,grptres,maxwall,preempt,flags 2>&1 | grep -iE "gpu|b200|a100|l40|normal|prem|tandon|625|520" | head -40
echo "=== partitions (name, avail, nodes, gres, features, allowaccounts, allowqos)"
timeout 120 scontrol show partition 2>&1 | grep -E "PartitionName|AllowAccounts|AllowQos|QoS=|DenyAccounts|TRES=" | paste - - - - 2>/dev/null | grep -iE "a100|b200|l40|h100|h200|prem|tandon|625|520" | cut -c1-600 | head -40
echo "=== GPU types per partition (sinfo)"
timeout 120 sinfo -h -o "%P|%G|%f|%D|%a" 2>&1 | grep -iE "a100|b200|l40" | sort -u | head -40
echo "=== this batch's jobs (llmadd_tpp-*): id name account partition state nice priority reason"
timeout 300 squeue -u yh6210 -h -o "%i|%j|%a|%P|%T|%y|%Q|%r|%M|%L" 2>&1 | grep "llmadd_tpp" > /scratch/yh6210/llmadd_torch/scale_work/tpp/t8/squeue_tpp.txt
wc -l < /scratch/yh6210/llmadd_torch/scale_work/tpp/t8/squeue_tpp.txt
awk -F'|' '{split($2,a,"-"); k=$3"|"$4"|"$5; c[k]++} END {for (k in c) print c[k], k}' /scratch/yh6210/llmadd_torch/scale_work/tpp/t8/squeue_tpp.txt | sort -k2
echo "=== B200 slot jobs"
grep "b200-slot" /scratch/yh6210/llmadd_torch/scale_work/tpp/t8/squeue_tpp.txt
echo "=== other own jobs (ptshare / dim / others): name account partition state nice"
timeout 300 squeue -u yh6210 -h -o "%j|%a|%P|%T|%y" 2>&1 | grep -v llmadd_tpp | sort | uniq -c | head -30
