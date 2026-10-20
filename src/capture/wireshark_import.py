
import argparse
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

from src.capture.flow_builder import FlowTracker


def import_pcap(path: str) -> pd.DataFrame:
    print(f"Reading {path} ...")
    packets = rdpcap(path)
    print(f"Loaded {len(packets)} packets")

    tracker = FlowTracker()
    for pkt in packets:
        if IP not in pkt:
            continue
        src, dst = pkt[IP].src, pkt[IP].dst
        plen = len(pkt)
        proto = pkt[IP].proto
        sport, dport, flags = 0, 0, None

        if TCP in pkt:
            sport, dport, flags = pkt[TCP].sport, pkt[TCP].dport, pkt[TCP].flags
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        ts = float(pkt.time)
        tracker.process_packet(src, dst, sport, dport, proto, plen, flags)
        # process_packet already timestamps internally with time.time();
        # if you need exact pcap timestamps for offline analysis, extend
        # FlowRecord.add_packet to accept `timestamp=ts` (it already supports it).

    # Flush any flows still "open" at end of capture
    for flow in list(tracker.flows.values()):
        tracker.completed.append(flow)
    tracker.flows.clear()

    rows = tracker.get_completed_features()
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} flow records from the capture")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_path", help="Path to .pcap/.pcapng file exported from Wireshark")
    parser.add_argument("--out", default="data/processed/wireshark_features.csv")
    args = parser.parse_args()

    df = import_pcap(args.pcap_path)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")