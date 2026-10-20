# src/rag/knowledge_base.py
"""
Retrieval-Augmented Knowledge component for the Explanation Agent.

Small curated knowledge base (attack description, MITRE ATT&CK technique,
mitigation) per CICIDS label. TF-IDF + cosine similarity does the retrieval
(no external API/embedding service needed — keeps this fully offline and
reproducible for your demo). The retrieved entry is combined with the SHAP
top-feature dict to produce a grounded, human-readable explanation string.

If you want nicer prose later, swap `_template_explanation` for a call to
an LLM, passing the retrieved KB entry + SHAP features as context — that's
the natural place to plug one in without changing anything else.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = [
    {
        "label": "DDoS",
        "mitre_id": "T1498",
        "description": "Distributed Denial of Service: high-volume flood of "
                        "packets (often SYN or UDP) from many sources aimed at "
                        "exhausting server or network resources.",
        "mitigation": "Rate limiting, SYN cookies, upstream traffic scrubbing.",
    },
    {
        "label": "PortScan",
        "mitre_id": "T1046",
        "description": "Network service scanning: sequential probing of ports "
                        "to discover open services, usually a reconnaissance "
                        "precursor to a targeted attack.",
        "mitigation": "Firewall port restrictions, IDS alerting on scan patterns.",
    },
    {
        "label": "Bot",
        "mitre_id": "T1071",
        "description": "Botnet command-and-control traffic: periodic beaconing "
                        "from an infected host to an external controller.",
        "mitigation": "Block known C2 domains/IPs, isolate and reimage host.",
    },
    {
        "label": "FTP-Patator",
        "mitre_id": "T1110",
        "description": "Brute-force credential guessing against an FTP service "
                        "via repeated rapid login attempts.",
        "mitigation": "Account lockout policies, disable FTP in favor of SFTP.",
    },
    {
        "label": "SSH-Patator",
        "mitre_id": "T1110",
        "description": "Brute-force credential guessing against SSH via "
                        "repeated rapid login attempts.",
        "mitigation": "Key-based auth only, fail2ban, rate limiting.",
    },
    {
        "label": "Web Attack - Brute Force",
        "mitre_id": "T1110",
        "description": "Repeated login attempts against a web application's "
                        "authentication endpoint.",
        "mitigation": "CAPTCHA, account lockout, WAF rate limiting.",
    },
    {
        "label": "Web Attack - Sql Injection",
        "mitre_id": "T1190",
        "description": "Malicious SQL fragments injected into web input fields "
                        "to manipulate or exfiltrate database contents.",
        "mitigation": "Parameterized queries, input validation, WAF.",
    },
    {
        "label": "Web Attack - XSS",
        "mitre_id": "T1059.007",
        "description": "Cross-site scripting: malicious script injected into "
                        "web pages, executed in victims' browsers.",
        "mitigation": "Output encoding, Content-Security-Policy headers.",
    },
    {
        "label": "Infiltration",
        "mitre_id": "T1190",
        "description": "Initial-access exploitation traffic, often preceding "
                        "lateral movement inside the network.",
        "mitigation": "Patch exposed services, network segmentation.",
    },
    {
        "label": "Heartbleed",
        "mitre_id": "CVE-2014-0160",
        "description": "Exploitation of the OpenSSL Heartbleed vulnerability "
                        "to leak process memory (keys, credentials) from a server.",
        "mitigation": "Patch OpenSSL, rotate certificates/keys.",
    },
]


class KnowledgeRetriever:
    def __init__(self, kb: list = None):
        self.kb = kb or KNOWLEDGE_BASE
        self.labels = [e["label"] for e in self.kb]
        corpus = [f"{e['label']} {e['description']}" for e in self.kb]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, predicted_label: str, top_k: int = 1) -> list:
        """Exact label match first (fast path); falls back to TF-IDF
        similarity search for labels not in the KB verbatim."""
        for entry in self.kb:
            if entry["label"].lower() == predicted_label.lower():
                return [entry]

        query_vec = self.vectorizer.transform([predicted_label])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.kb[i] for i in top_idx]


def generate_explanation(predicted_label: str, confidence: float,
                          shap_top_features: dict, retriever: KnowledgeRetriever) -> str:
    """Combines the retrieved knowledge entry with SHAP feature attributions
    into one grounded explanation string — this is the RAG output shown to
    the analyst in the dashboard, and the thing to screenshot for your report."""
    if predicted_label.upper() == "BENIGN":
        return f"Classified as benign traffic (confidence {confidence:.1f}%). No known attack pattern matched."

    hits = retriever.retrieve(predicted_label)
    if not hits:
        return f"Flagged as {predicted_label} (confidence {confidence:.1f}%). No matching knowledge base entry found."

    entry = hits[0]
    top_feats = ", ".join(f"{k} ({v:+.3f})" for k, v in list(shap_top_features.items())[:5])

    return (
        f"Flagged as {entry['label']} (confidence {confidence:.1f}%), "
        f"matching {entry['mitre_id']}: {entry['description']} "
        f"Model attributed the decision mainly to: {top_feats}. "
        f"Suggested mitigation: {entry['mitigation']}"
    )


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    demo_shap = {"Flow Duration": 0.42, "Bwd Packet Length Mean": -0.31, "Flow IAT Mean": 0.18}
    print(generate_explanation("DDoS", 97.8, demo_shap, retriever))