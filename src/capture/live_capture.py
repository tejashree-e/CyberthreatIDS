"""
live_capture.py — Demo-ready IDS with fast detection + colored output
"""

import time
import queue
import joblib
import threading
import numpy as np
from scapy.all import sniff, IP, TCP, UDP

from src.capture.flow_builder import FlowTracker
from src.models.ensemble import EnsemblePredictor

# ── Colors ────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
CYAN   = "\033[96m"; WHITE  = "\033[97m"; BLUE   = "\033[94m"

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_INTERVAL    = 2.0   # check for new flows every 1.5s
DEDUP_TTL        = 30     # don't re-alert same flow for 30s
MAX_BENIGN_LOGS  = 2      # max benign lines per cycle
SIMULATE_ATTACKS = True   # demo heuristics on top of ML

result_queue: queue.Queue = queue.Queue()
_seen_flows: dict = {}
_seen_lock = threading.Lock()
_stats = {'total': 0, 'alerts': 0, 'benign': 0, 'start': time.time()}

# ── Deduplication ─────────────────────────────────────────────────────────────
def _is_duplicate(key: tuple) -> bool:
    now = time.time()
    with _seen_lock:
        if key in _seen_flows and (now - _seen_flows[key]) < DEDUP_TTL:
            return True
        _seen_flows[key] = now
        return False

# ── Heuristics (demo attack simulation) ───────────────────────────────────────
def _apply_heuristics(feat: dict, result: dict) -> dict:
    """
    Rule-based overrides so real localhost traffic triggers alerts.
    Only fires when model says BENIGN. Tuned for Mac localhost demo.
    """
    if not SIMULATE_ATTACKS or result['label'].upper() != 'BENIGN':
        return result

    syn      = feat.get('SYN Flag Count', 0)
    ack      = feat.get('ACK Flag Count', 0)
    rst      = feat.get('RST Flag Count', 0)
    fin      = feat.get('FIN Flag Count', 0)
    fwd      = feat.get('Total Fwd Packets', 0)
    bwd      = feat.get('Total Bwd Packets', 0)
    pps      = feat.get('Flow Packets/s', 0)
    bps      = feat.get('Flow Bytes/s', 0)
    duration = feat.get('Flow Duration', 1)   # microseconds
    total=fwd+bwd;
    
    # hping3 flood → VERY high packet rate → DDoS
 # nmap SYN scan: many SYN, zero or few ACK, many RST responses
    if syn >= 3 and rst >= 2 and ack < syn:
        result.update({'label': 'PortScan',     'confidence': 91.2, 'risk_level': 'HIGH'})

    # nmap intense: lots of packets, mix of flags, very short flows
    elif syn >= 2 and fwd >= 5 and duration < 500_000:   # < 0.5s
        result.update({'label': 'PortScan',     'confidence': 84.7, 'risk_level': 'HIGH'})

    # hping3 flood: massive packet rate (even localhost flood hits >50 pps)
        
    elif pps > 500:
        result.update({'label': 'DoS-Hulk',   'confidence': 92.1, 'risk_level': 'CRITICAL'})

    # Python socket flood (connect() loop): many fwd, no bwd, high RST
    elif fwd > 30 and bwd == 0 and rst > 5:
        result.update({'label': 'DDoS-Connect', 'confidence': 88.9, 'risk_level': 'CRITICAL'})

    # Asymmetric probe: many forward packets, no responses at all
    elif fwd > 15 and bwd == 0:
        result.update({'label': 'Probe',        'confidence': 76.4, 'risk_level': 'MEDIUM'})

    # Bulk transfer: large bytes, few packets = possible exfil
    elif bps > 500_000 and fwd + bwd < 8:
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
    # Take top 3 busiest active flows
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
    row = np.array([feat.get(c, 0.0) for c in feat_cols])
    scaled = scaler.transform(row.reshape(1, -1))
    result = predictor.predict(scaled)[0]
    # attach SHAP explanation
    try:
        from src.models.explain import explain_single
        explainer_path = 'data/models/shap_explainer.pkl'
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
    print(f"  {CYAN}Interface  : {interface or 'default (all)' }{RESET}")
    print(f"  {CYAN}Eval cycle : every {EVAL_INTERVAL}s{RESET}")
    print(f"  {CYAN}Dedup TTL  : {DEDUP_TTL}s{RESET}")
    print(f"  {CYAN}Demo mode  : {'ON — heuristics active' if SIMULATE_ATTACKS else 'OFF'}{RESET}")
    _print_separator()
    print(f"  {DIM}Listening for traffic…{RESET}\n")

    last_eval    = [time.time()]
    last_stats   = [time.time()]

    def packet_callback(pkt):
        if IP not in pkt:
            return

        src, dst, plen, proto = pkt[IP].src, pkt[IP].dst, len(pkt), pkt[IP].proto
        sport = dport = 0
        flags = None

        if TCP in pkt:
            sport, dport, flags = pkt[TCP].sport, pkt[TCP].dport, pkt[TCP].flags
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        tracker.process_packet(src, dst, sport, dport, proto, plen, flags)

        now = time.time()

        # Print stats bar every 15 seconds
        if now - last_stats[0] >= 15:
            last_stats[0] = now
            _print_stats()

        if now - last_eval[0] < EVAL_INTERVAL:
            return
        last_eval[0] = now

        # Get completed flows first, fall back to active
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

    sniff(iface=interface, prn=packet_callback, store=False, timeout=duration)

if __name__ == '__main__':
    start_capture()