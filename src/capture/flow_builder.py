import time
from collections import defaultdict
import numpy as np


class FlowRecord:
    """Tracks a single TCP/UDP network flow."""

    def __init__(self, key):
        self.key = key
        self.start_time = time.time()
        self.last_time = time.time()
        self.fwd_pkts, self.bwd_pkts = [], []
        self.fwd_bytes, self.bwd_bytes = 0, 0
        self.fwd_iat, self.bwd_iat = [], []
        self.flags = defaultdict(int)  # SYN, ACK, FIN, RST, PSH, URG

    def add_packet(self, pkt_len, direction, tcp_flags=None, timestamp=None):
        now = timestamp or time.time()
        iat = now - self.last_time
        self.last_time = now

        if direction == 'fwd':
            self.fwd_pkts.append(pkt_len)
            self.fwd_bytes += pkt_len
            self.fwd_iat.append(iat)
        else:
            self.bwd_pkts.append(pkt_len)
            self.bwd_bytes += pkt_len
            self.bwd_iat.append(iat)

        if tcp_flags:
            for f in ['S', 'A', 'F', 'R', 'P', 'U']:
                if f in str(tcp_flags):
                    self.flags[f] += 1

    def to_feature_vector(self) -> dict:
        duration = max(self.last_time - self.start_time, 1e-6)
        total_pkts = len(self.fwd_pkts) + len(self.bwd_pkts)
        all_iat = self.fwd_iat + self.bwd_iat
        feat={}
        feat['_src']   = self.key[0]
        feat['_dst']   = self.key[1]
        feat['_dport'] = str(self.key[3])
        feat['_proto'] = 'TCP' if self.key[4] == 6 else ('UDP' if self.key[4] == 17 else str(self.key[4]))
        

        def stats(lst):
            if not lst:
                return 0, 0, 0, 0
            a = np.array(lst)
            return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

        fwd_m, fwd_s, fwd_mn, fwd_mx = stats(self.fwd_pkts)
        bwd_m, bwd_s, bwd_mn, bwd_mx = stats(self.bwd_pkts)
        iat_m, iat_s, iat_mn, iat_mx = stats(all_iat)
        

        return {
            'Flow Duration': duration * 1e6,
            'Total Fwd Packets': len(self.fwd_pkts),
            'Total Bwd Packets': len(self.bwd_pkts),
            'Total Length Fwd': self.fwd_bytes,
            'Total Length Bwd': self.bwd_bytes,
            'Fwd Packet Length Mean': fwd_m,
            'Fwd Packet Length Std': fwd_s,
            'Fwd Packet Length Min': fwd_mn,
            'Fwd Packet Length Max': fwd_mx,
            'Bwd Packet Length Mean': bwd_m,
            'Bwd Packet Length Std': bwd_s,
            'Bwd Packet Length Min': bwd_mn,
            'Bwd Packet Length Max': bwd_mx,
            'Flow Bytes/s': (self.fwd_bytes + self.bwd_bytes) / duration,
            'Flow Packets/s': total_pkts / duration,
            'Flow IAT Mean': iat_m,
            'Flow IAT Std': iat_s,
            'Flow IAT Min': iat_mn,
            'Flow IAT Max': iat_mx,
            'Fwd IAT Mean': stats(self.fwd_iat)[0],
            'Bwd IAT Mean': stats(self.bwd_iat)[0],
            'SYN Flag Count': self.flags['S'],
            'ACK Flag Count': self.flags['A'],
            'FIN Flag Count': self.flags['F'],
            'RST Flag Count': self.flags['R'],
            'PSH Flag Count': self.flags['P'],
            'URG Flag Count': self.flags['U'],
            'Avg Fwd Segment Size': fwd_m,
            'Avg Bwd Segment Size': bwd_m,
            'Subflow Fwd Packets': len(self.fwd_pkts),
            'Subflow Bwd Packets': len(self.bwd_pkts),
        }
        


class FlowTracker:
    """Maintains all active flows, expires old ones."""
    FLOW_TIMEOUT = 120  # seconds

    def __init__(self):
        self.flows = {}
        self.completed = []

    def _key(self, src_ip, dst_ip, sport, dport, proto):
        return (
            min(src_ip, dst_ip),
            max(src_ip, dst_ip),
            min(sport, dport),
            max(sport, dport),
            proto
        )

    def process_packet(self, src_ip, dst_ip, sport, dport, proto, pkt_len, tcp_flags=None):
        key = self._key(src_ip, dst_ip, sport, dport, proto)
        direction = 'fwd' if (src_ip, sport) <= (dst_ip, dport) else 'bwd'

        if key not in self.flows:
            self.flows[key] = FlowRecord(key)

        self.flows[key].add_packet(pkt_len, direction, tcp_flags)

        # Expire old flows
        now = time.time()
        expired = [
            k for k, v in self.flows.items()
            if now - v.last_time > self.FLOW_TIMEOUT
        ]

        for k in expired:
            self.completed.append(self.flows.pop(k))

    def get_completed_features(self):
        out = [f.to_feature_vector() for f in self.completed]
        self.completed.clear()
        return out