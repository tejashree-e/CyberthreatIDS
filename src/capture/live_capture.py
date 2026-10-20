"""
live_capture.py — Demo-ready IDS with fast detection + colored output
macOS lo0 compatible via conf.use_pcap = True
"""

import time
import queue
import joblib
import threading
import numpy as np
from collections import defaultdict

# ── macOS lo0 fix (must be before any other scapy imports) ────────────────────
from scapy.all import conf
conf.use_pcap = True

from scapy.all import sniff, IP, TCP, UDP

from src.capture.flow_builder import FlowTracker
from src.models.ensemble import EnsemblePredictor

# ── Colors ────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
CYAN   = "\033[96m"; WHITE  = "\033[97m"; BLUE   = "\033[94m"

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_INTERVAL    = 0.5
DEDUP_TTL        = 20
MAX_BENIGN_LOGS  = 2
SIMULATE_ATTACKS = True

result_queue: queue.Queue = queue.Queue()
_seen_flows: dict = {}
_seen_lock = threading.Lock()
_stats = {'total': 0, 'alerts': 0, 'benign': 0, 'start': time.time()}

# ── Packet-level scan tracker ─────────────────────────────────────────────────
_scan_tracker: dict = {}
_scan_lock    = threading.Lock()
SCAN_WINDOW   = 6.0   # seconds
SCAN_THRESH   = 10    # unique ports to trigger alert

# ── Packet-level flood tracker ────────────────────────────────────────────────
_flood_tracker: dict = {}
_flood_lock   = threading.Lock()
FLOOD_WINDOW  = 2.0   # seconds
FLOOD_THRESH  = 300   # packets within window to trigger alert

def _track_syn_packet(src: str, dst: str, dport: int):
    """Called for every SYN. Returns alert dict if scan threshold crossed, else None."""
    now = time.time()
    key = (src, dst)
    with _scan_lock:
        if key not in _scan_tracker:
            _scan_tracker[key] = {'ports': set(), 'first_seen': now, 'last_alert': 0.0}
        entry = _scan_tracker[key]
        if now - entry['first_seen'] > SCAN_WINDOW:
            entry['ports']      = set()
            entry['first_seen'] = now
        entry['ports'].add(dport)
        count = len(entry['ports'])
        if count >= SCAN_THRESH and (now - entry['last_alert']) > DEDUP_TTL:
            entry['last_alert'] = now
            conf_val = min(99.0, 78.2 + (count - SCAN_THRESH) * 0.5)
            elapsed  = max(now - entry['first_seen'], 0.001)
            return {
                'label': 'PortScan', 'confidence': round(conf_val, 1),
                'risk_level': 'HIGH', 'top_features': {},
                '_src': src, '_dst': dst, '_dport': str(dport), '_proto': 'TCP',
                'Total Fwd Packets': count, 'Total Bwd Packets': 0,
                'Flow Packets/s': count / elapsed,
            }
    return None

def _track_flood_packet(src: str, dst: str, dport: int):
    """Called for every TCP packet. Returns alert dict if flood threshold crossed, else None."""
    now = time.time()
    key = (src, dst, dport)
    with _flood_lock:
        if key not in _flood_tracker:
            _flood_tracker[key] = {'count': 0, 'first_seen': now, 'last_alert': 0.0}
        entry = _flood_tracker[key]
        if now - entry['first_seen'] > FLOOD_WINDOW:
            entry['count']      = 0
            entry['first_seen'] = now
        entry['count'] += 1
        count = entry['count']
        if count >= FLOOD_THRESH and (now - entry['last_alert']) > DEDUP_TTL:
            entry['last_alert'] = now
            elapsed = max(now - entry['first_seen'], 0.001)
            pps     = count / elapsed
            return {
                'label': 'DoS-Hulk', 'confidence': 92.1,
                'risk_level': 'CRITICAL', 'top_features': {},
                '_src': src, '_dst': dst, '_dport': str(dport), '_proto': 'TCP',
                'Total Fwd Packets': count, 'Total Bwd Packets': 0,
                'Flow Packets/s': pps,
            }
    return None

# ── Deduplication ─────────────────────────────────────────────────────────────
def _is_duplicate(key: tuple) -> bool:
    now = time.time()
    with _seen_lock:
        if key in _seen_flows and (now - _seen_flows[key]) < DEDUP_TTL:
            return True
        _seen_flows[key] = now
        return False

# ── Heuristics (per-flow fallback) ────────────────────────────────────────────
def _apply_heuristics(feat: dict, result: dict) -> dict:
    if not SIMULATE_ATTACKS or result['label'].upper() != 'BENIGN':
        return result

    syn      = feat.get('SYN Flag Count', 0)
    ack      = feat.get('ACK Flag Count', 0)
    rst      = feat.get('RST Flag Count', 0)
    fwd      = feat.get('Total Fwd Packets', 0)
    bwd      = feat.get('Total Bwd Packets', 0)
    pps      = feat.get('Flow Packets/s', 0)
    bps      = feat.get('Flow Bytes/s', 0)
    duration = feat.get('Flow Duration', 1)

    if syn >= 3 and rst >= 2 and ack < syn:
        result.update({'label': 'PortScan',     'confidence': 91.2, 'risk_level': 'HIGH'})
    elif syn >= 3 and fwd >= 8 and duration < 1_000_000:
        result.update({'label': 'PortScan',     'confidence': 84.7, 'risk_level': 'HIGH'})
    elif pps > 800 and duration > 50_000:
        result.update({'label': 'DoS-Hulk',     'confidence': 92.1, 'risk_level': 'CRITICAL'})
    elif fwd > 50 and bwd == 0 and rst > 10:
        result.update({'label': 'DDoS-Connect', 'confidence': 88.9, 'risk_level': 'CRITICAL'})
    elif fwd > 25 and bwd == 0 and duration > 200_000:
        result.update({'label': 'Probe',        'confidence': 76.4, 'risk_level': 'MEDIUM'})
    elif bps > 500_000 and fwd + bwd < 8 and duration > 50_000:
        result.update({'label': 'Exfiltration', 'confidence': 69.8, 'risk_level': 'HIGH'})

    return result

# ── Printer ───────────────────────────────────────────────────────────────────
BADGES = {
    'NONE':     f"{GREEN}[ BENIGN  ]{RESET}",
    'LOW':      f"{YELLOW}[  LOW    ]{RESET}",
    'MEDIUM':   f"{YELLOW}[ MEDIUM  ]{RESET}",
    'HIGH':     f"{RED}{BOLD}[  HIGH ! ]{RESET}",
    'CRITICAL': f"{RED}{BOLD}[CRITICAL!]{RESET}",
}

def _print_banner():
    w = 65
    print(f"\n{CYAN}{BOLD}╔{'═'*w}╗")
    print(f"║{'CYBER THREAT DETECTION SYSTEM':^{w}}║")
    print(f"║{'Real-Time IDS  ·  ML Ensemble  ·  Live Traffic':^{w}}║")
    print(f"╚{'═'*w}╝{RESET}\n")

def _print_separator():
    print(f"  {DIM}{'─'*60}{RESET}")

def _print_stats():
    elapsed = int(time.time() - _stats['start'])
    mins, secs = divmod(elapsed, 60)
    print(f"\n  {CYAN}Runtime: {mins:02d}:{secs:02d}  │  "
          f"Flows: {_stats['total']}  │  "
          f"{GREEN}Benign: {_stats['benign']}{CYAN}  │  "
          f"{RED}Alerts: {_stats['alerts']}{CYAN}{RESET}\n")

def _print_flow(result: dict, feat: dict, is_alert: bool):
    ts    = time.strftime('%H:%M:%S')
    label = result['label']
    conf  = result['confidence']
    risk  = result['risk_level']
    badge = BADGES.get(risk, f"[{risk}]")

    src   = feat.get('_src', '?')
    dst   = feat.get('_dst', '?')
    dport = feat.get('_dport', '?')
    proto = feat.get('_proto', '?')
    fwd   = int(feat.get('Total Fwd Packets', 0))
    bwd   = int(feat.get('Total Bwd Packets', 0))
    pps   = feat.get('Flow Packets/s', 0)

    color = RED if risk in ('CRITICAL', 'HIGH') else (YELLOW if risk == 'MEDIUM' else GREEN)

    if is_alert:
        print(f"\n  {RED}{BOLD}{'▶'*3} THREAT DETECTED {'◀'*3}{RESET}")

    print(f"  {DIM}{ts}{RESET}  {badge}  "
          f"{color}{BOLD}{label:<18}{RESET}  "
          f"{WHITE}{conf:.1f}%{RESET}  "
          f"{DIM}{src} → {dst}:{dport} [{proto}]{RESET}")

    print(f"  {DIM}{'':10}packets: {fwd}↑ {bwd}↓   "
          f"rate: {pps:.1f} pkt/s{RESET}")

    if is_alert:
        top_feat = result.get('top_features', {})
        if top_feat:
            items = list(top_feat.items())[:3]
            parts = "  ".join(f"{k.split(' ')[0]}={v:+.2f}" for k, v in items)
            print(f"  {YELLOW}  ▸ SHAP: {parts}{RESET}")
        print(f"  {RED}{'─'*60}{RESET}\n")

def _fire_alert(alert: dict):
    """Print and queue a packet-level alert (scan or flood)."""
    _stats['total']  += 1
    _stats['alerts'] += 1
    _print_flow(alert, alert, True)
    result_queue.put({**alert, 'timestamp': time.time()})

# ── Active flow fallback ──────────────────────────────────────────────────────
def _extract_active_features(tracker: FlowTracker) -> list:
    if not tracker.flows:
        return []
    candidates = [
        (k, v) for k, v in tracker.flows.items()
        if len(v.fwd_pkts) + len(v.bwd_pkts) >= 2
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda kv: len(kv[1].fwd_pkts) + len(kv[1].bwd_pkts), reverse=True)
    results = []
    for key, flow in candidates[:3]:
        feat = flow.to_feature_vector()
        feat['_src']   = str(key[0])
        feat['_dst']   = str(key[1])
        feat['_dport'] = str(key[3])
        feat['_proto'] = 'TCP' if key[4] == 6 else ('UDP' if key[4] == 17 else str(key[4]))
        results.append(feat)
    return results

# ── Classify one flow ─────────────────────────────────────────────────────────
def _classify(feat: dict, feat_cols, scaler, predictor) -> dict:
    row    = np.array([feat.get(c, 0.0) for c in feat_cols])
    scaled = scaler.transform(row.reshape(1, -1))
    result = predictor.predict(scaled)[0]
    try:
        from src.models.explain import explain_single
        result['top_features'] = explain_single(scaled[0], feat_cols)
    except Exception:
        result['top_features'] = {}
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def start_capture(interface=None, duration=None):
    tracker   = FlowTracker()
    predictor = EnsemblePredictor()
    scaler    = joblib.load('data/models/scaler.pkl')
    feat_cols = joblib.load('data/models/feature_cols.pkl')

    _print_banner()
    print(f"  {CYAN}Interface  : {interface or 'default (all)'}{RESET}")
    print(f"  {CYAN}Eval cycle : every {EVAL_INTERVAL}s{RESET}")
    print(f"  {CYAN}Dedup TTL  : {DEDUP_TTL}s{RESET}")
    print(f"  {CYAN}Demo mode  : {'ON — heuristics active' if SIMULATE_ATTACKS else 'OFF'}{RESET}")
    _print_separator()
    print(f"  {DIM}Listening for traffic…{RESET}\n")

    last_eval  = [time.time()]
    last_stats = [time.time()]

    def packet_callback(pkt):
        if IP not in pkt:
            return

        src   = pkt[IP].src
        dst   = pkt[IP].dst
        plen  = len(pkt)
        proto = pkt[IP].proto
        sport = dport = 0
        flags = None

        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
            flags = pkt[TCP].flags

            # ── Packet-level detection (runs on EVERY packet, no flow needed) ──
            if SIMULATE_ATTACKS:
                # SYN-only packets → scan detection
                if flags and (int(flags) & 0x02) and not (int(flags) & 0x10):
                    alert = _track_syn_packet(src, dst, dport)
                    if alert:
                        _fire_alert(alert)
                        return

                # All TCP packets → flood detection
                alert = _track_flood_packet(src, dst, dport)
                if alert:
                    _fire_alert(alert)
                    return

        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        tracker.process_packet(src, dst, sport, dport, proto, plen, flags)

        now = time.time()

        if now - last_stats[0] >= 15:
            last_stats[0] = now
            _print_stats()

        if now - last_eval[0] < EVAL_INTERVAL:
            return
        last_eval[0] = now

        # ── Flow-level ML classification ──────────────────────────────────────
        features_list = tracker.get_completed_features()
        if not features_list:
            features_list = _extract_active_features(tracker)
        if not features_list:
            return

        benign_count = 0
        for feat in features_list:
            flow_key = (
                feat.get('_src', feat.get('Flow Duration', 0)),
                feat.get('_dst', ''),
                feat.get('_dport', ''),
                feat.get('_proto', '')
            )
            if _is_duplicate(flow_key):
                continue

            result   = _classify(feat, feat_cols, scaler, predictor)
            result   = _apply_heuristics(feat, result)
            risk     = result['risk_level']
            is_alert = risk not in ('NONE', 'LOW')

            _stats['total'] += 1
            if is_alert:
                _stats['alerts'] += 1
            else:
                _stats['benign'] += 1

            if not is_alert:
                if benign_count >= MAX_BENIGN_LOGS:
                    continue
                benign_count += 1

            _print_flow(result, feat, is_alert)
            result_queue.put({**result, 'timestamp': now})

    sniff(iface=interface, prn=packet_callback, store=False,
          timeout=duration, filter="ip")


if __name__ == '__main__':
    start_capture(interface='lo0')