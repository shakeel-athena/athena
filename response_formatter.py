"""
Response Formatter - Context-Aware SOC Reporting
Implements SOC-grade formatting with strict filtering, suspicious detection,
and structured output for all query types.

RULES:
  1. Filter BEFORE format — always.
  2. No raw JSON in dashboard output (except safe fallback).
  3. Every response must show structured fields, not just counts.
  4. Suspicious detection for ports and processes.
  5. Vulnerability deduplication by CVE ID.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# SUSPICIOUS DETECTION CONSTANTS
# ============================================================================

SUSPICIOUS_PORTS = {
    4444: "Known Metasploit/reverse shell default port",
    5555: "Known Android debug / reverse shell port",
    6666: "Known IRC backdoor port",
    1337: "Known leet/hacker culture port — often used by malware",
    31337: "Known Back Orifice trojan port",
    12345: "Known NetBus trojan port",
    6667: "IRC — commonly used for C2 communication",
    6697: "IRC over TLS — commonly used for C2",
    9001: "Common Tor relay port",
    9050: "Common Tor SOCKS proxy port",
}

SUSPICIOUS_HIGH_PORTS_LISTENING = 49152  # Dynamic ports shouldn't be LISTEN

DATABASE_PORTS = {
    3306: "MySQL",
    5432: "PostgreSQL",
    27017: "MongoDB",
    6379: "Redis",
    9200: "Elasticsearch",
    5984: "CouchDB",
    1433: "MSSQL",
    1521: "Oracle DB",
}

INSECURE_PORTS = {
    21: "FTP — insecure, migrate to SFTP/SCP",
    23: "Telnet — insecure, migrate to SSH",
    25: "SMTP — verify if mail server, otherwise unexpected",
    69: "TFTP — insecure, no authentication",
    161: "SNMP v1/v2 — use SNMPv3 with encryption",
}

NON_STANDARD_HTTP = {
    8080: "Non-standard HTTP — verify application",
    8443: "Non-standard HTTPS — verify application",
    8888: "Non-standard HTTP — verify application",
    3000: "Dev server port — should not be in production",
    8000: "Dev server port — verify if intentional",
}

SUSPICIOUS_PROCESSES = [
    "xmrig", "minerd", "cryptonight", "stratum", "cpuminer",  # Crypto miners
    "nc", "ncat", "netcat", "socat",                          # Reverse shells
    "mimikatz", "procdump", "lazagne", "secretsdump",          # Credential theft
    "empire", "meterpreter", "beacon", "cobalt",               # C2 frameworks
    "certutil", "bitsadmin",                                    # Living-off-the-land
    "masscan", "nmap", "zmap",                                  # Scanners (unusual on endpoints)
    "hydra", "medusa", "hashcat", "john",                       # Brute force tools
]

SYSTEM_ROOT_PROCESSES = {
    # Core kernel and system
    "systemd", "init", "kthreadd", "ksoftirqd", "kworker", "rcu_sched",
    "rcu_preempt", "rcu_gp", "rcu_par_gp", "rcu_bh", "migration", "watchdog",
    "cpuhp", "netns", "kdevtmpfs", "kauditd", "khungtaskd", "oom_kill",
    "writeback", "kcompactd", "crypto", "kblockd", "ata_sff", "md",
    "edac-poller", "devfreq_wq", "kswapd", "cifsiod", "kthrotld",
    "irq", "scsi", "dm_bufio",
    # Power/hardware management
    "acpid", "thermald", "irqbalance", "fwupd", "udisksd", "upower",
    # System services
    "sshd", "rsyslogd", "cron", "crond", "atd", "auditd", "polkitd",
    "systemd-journald", "systemd-journal", "systemd-logind", "systemd-udevd",
    "systemd-resolved", "systemd-networkd", "systemd-timesyncd",
    "dbus-daemon", "agetty", "multipathd", "iscsid", "lvmetad", "lvm2-lvmpolld",
    # Ubuntu/Debian specific
    "accounts-daemon", "networkd-dispatcher", "unattended-upgr",
    "packagekitd", "snapd",
    # Wazuh agents
    "wazuh-agentd", "wazuh-execd", "wazuh-modulesd", "wazuh-syscheckd",
    "wazuh-logcollec", "wazuh-logcollector", "wazuh-analysisd", "wazuh-remoted",
    "wazuh-db",
    # Web servers
    "apache2", "nginx", "httpd",
    # Databases
    "mysqld", "postgres", "mongod", "redis-server",
}

# Kernel thread prefixes — these are normal system threads, not suspicious
KERNEL_THREAD_PREFIXES = (
    "kworker/", "irq/", "rcu_", "scsi_", "xfs_", "ext4-",
    "jbd2/", "loop", "blkcg_punt", "devfreq_", "watchdog/",
    "migration/", "ksoftirqd/", "cpuhp/", "idle_inject/",
    "kcompactd", "writeback", "md_", "dm-",
)


class ResponseFormatter:
    """
    SOC-grade response formatter with context awareness.

    Key Features:
    - Excludes manager (ID 000) from all agent counts
    - Shows only relevant sections based on query type
    - Proper tabular formatting with full agent details
    - Suspicious port and process detection
    - Vulnerability deduplication by CVE ID
    - Safe fallback mode on parse errors
    """

    def _exclude_manager(self, agents: List[Dict]) -> List[Dict]:
        """Remove Wazuh manager (ID 000) from agent lists"""
        if not isinstance(agents, list):
            return []
        return [a for a in agents if a.get('id') != '000']

    @staticmethod
    def _os_aliases(os_filter: str) -> List[str]:
        """Return all keywords that should match a given canonical OS value.

        Used by defensive OS filters in the formatter / correlation layers so
        that "Microsoft Windows Server 2022", "Win32", "Linux", "Ubuntu",
        "macOS Sonoma", "Red Hat Enterprise Linux 8.5", "Amazon Linux 2"
        etc. all match the canonical query.

        IMPORTANT — Linux is special. Wazuh stores the *distribution* in
        host.os.platform (e.g. "ubuntu", "rhel"), not the family. So the
        Linux alias list MUST enumerate every distro name; otherwise the
        defensive filter throws away legitimate Ubuntu / RHEL / CentOS rows
        that the indexer correctly returned. This list is kept in sync with
        wazuh_indexer.get_vulnerabilities()'s linux_distros — do not let
        them drift apart.
        """
        s = (os_filter or "").lower().strip()
        if s == "windows":
            return ["windows", "win32", "microsoft windows", " win "]
        if s == "linux":
            return [
                "linux", "ubuntu", "debian", "centos", "redhat", "red hat",
                "rhel", "fedora", "alpine", "kali", "amazon linux", "amzn",
                "amazon", "suse", "opensuse", "oracle", "oracle linux",
                "rocky", "rocky linux", "almalinux", "alma", "arch",
                "raspbian", "gentoo",
            ]
        if s == "darwin":
            return ["darwin", "macos", "mac os", "osx", "os x", "mac"]
        return [s]

    def _extract_data(self, correlated_data: Any) -> List[Dict]:
        """
        Extract data from various possible structures.
        Handles both direct lists and nested Wazuh API responses.
        Normalizes all MCP tool payload formats:
          - {"content":[{"text":"...json..."}]}
          - {"data": {...}}
          - {"affected_items":[...]}
          - direct list
        """
        if isinstance(correlated_data, list):
            return correlated_data

        if isinstance(correlated_data, dict):
            # MCP wrapper: {"content":[{"text":"...json..."}]}
            if "content" in correlated_data and isinstance(correlated_data.get("content"), list):
                content = correlated_data["content"]
                if content:
                    text = content[0].get("text", "")
                    if isinstance(text, str) and text.strip():
                        try:
                            parsed = json.loads(text)
                            return self._extract_data(parsed)
                        except json.JSONDecodeError:
                            logger.warning("[EXTRACT] Could not parse JSON from content text")
                            return []

            if 'data' in correlated_data:
                data = correlated_data['data']
                if isinstance(data, dict) and 'affected_items' in data:
                    items = data['affected_items']
                    return items if isinstance(items, list) else []
                elif isinstance(data, list):
                    return data

            if 'affected_items' in correlated_data:
                items = correlated_data['affected_items']
                return items if isinstance(items, list) else []

            if 'items' in correlated_data:
                items = correlated_data['items']
                return items if isinstance(items, list) else []

        return []

    # ================================================================
    # AGENT FORMATTERS
    # ================================================================

    # Status aliases: user-friendly terms → Wazuh canonical status
    STATUS_ALIASES = {
        "online": "active",
        "running": "active",
        "connected": "active",
        "up": "active",
        "offline": "disconnected",
        "down": "disconnected",
        "inactive": "disconnected",
        "lost": "disconnected",
    }

    def format_agent_status_response(self, correlated_data, summary, status_filter=None, **kwargs):
        """
        Main agent formatter — routes based on status_filter.
        RULE: Filter BEFORE format. Exact match only. No cross-status leakage.
        """
        agents = self._extract_data(correlated_data)
        agents = self._exclude_manager(agents)

        if status_filter:
            canonical = self.STATUS_ALIASES.get(status_filter.lower(), status_filter.lower())
            # STRICT exact-match filter
            agents = [a for a in agents if a.get('status', '').lower() == canonical]
            return self._format_agent_details_table(agents, canonical)
        else:
            return self._format_agent_overview(agents)

    def _format_agent_details_table(self, agents: List[Dict], status_label: str) -> str:
        """
        Full details table for a specific status filter.
        Shows: Agent ID, Name, IP, OS, Version, Status, Last Keep Alive.
        Works for active, disconnected, never_connected, or any status.
        """
        status_icons = {
            "active": "🟢", "disconnected": "🔴",
            "never_connected": "🟡", "pending": "🟡"
        }
        icon = status_icons.get(status_label, "⚪")

        if not agents:
            return f"{icon} No {status_label} agents found (excluding manager)"

        lines = [
            f"## {icon} {status_label.replace('_', ' ').title()} Agents: {len(agents)}",
            "",
            "| Agent ID | Name | IP Address | OS | Version | Status | Last Keep Alive |",
            "|----------|------|------------|----|---------:|--------|-----------------|"
        ]

        for agent in agents:
            agent_id = agent.get('id', 'N/A')
            name = agent.get('name', 'N/A')
            ip = agent.get('ip', 'N/A')
            os_info = agent.get('os', {})
            os_name = f"{os_info.get('name', 'N/A')} {os_info.get('version', '')}".strip() if isinstance(os_info, dict) else 'N/A'
            version = str(agent.get('version', 'N/A')).replace('Wazuh v', '')
            status = agent.get('status', 'N/A')
            last_ka = str(agent.get('lastKeepAlive', 'N/A'))
            if last_ka and last_ka != 'N/A':
                last_ka = last_ka[:19].replace('T', ' ')

            lines.append(
                f"| {agent_id} | {name} | {ip} | {os_name} | {version} | {status} | {last_ka} |"
            )

        lines.append("")
        lines.append(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show all agents` | `check agent <ID> health` | `show vulnerabilities` | `show ports for agent <ID>`")

        return "\n".join(lines)

    def _format_agent_overview(self, agents: List[Dict]) -> str:
        """
        Overview when no status filter: shows status summary counts
        AND first 10 agents with full details so user always sees actual data.
        """
        status_counts = {
            'active': 0, 'disconnected': 0, 'never_connected': 0, 'pending': 0
        }
        for agent in agents:
            status = agent.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1

        total = sum(status_counts.values())

        if total == 0:
            return "## Agent Overview\n\nNo agents found (excluding manager)"

        health_pct = (status_counts['active'] / total * 100) if total > 0 else 0

        lines = [
            "## Agent Status Overview",
            "",
            f"**Total Agents:** {total} (excluding manager) | **Infrastructure Health:** {health_pct:.0f}% online",
            "",
            "| Status | Count | Percentage |",
            "|--------|------:|-----------:|"
        ]

        for status, count in status_counts.items():
            if count > 0:
                pct = (count / total * 100) if total > 0 else 0
                label = status.replace('_', ' ').title()
                lines.append(f"| {label} | {count} | {pct:.1f}% |")

        if status_counts['disconnected'] > 0:
            lines.append("")
            lines.append("### Action Required")
            lines.append(f"- **{status_counts['disconnected']} disconnected agents** need investigation")
            lines.append("- Check: Network connectivity, firewall rules, agent service status")

            # Show longest-disconnected agents
            disconnected = [a for a in agents if a.get('status', '').lower() == 'disconnected']
            disconnected.sort(key=lambda a: a.get('lastKeepAlive', ''), reverse=False)
            if disconnected:
                top_disc = disconnected[:3]
                lines.append("")
                lines.append("**Longest Disconnected:**")
                for a in top_disc:
                    last_ka = str(a.get('lastKeepAlive', 'N/A'))[:19].replace('T', ' ')
                    lines.append(f"- Agent {a.get('id', '?')} ({a.get('name', '?')}) — last seen: {last_ka}")

        # Show first 10 agents with details
        show_agents = agents[:10]
        if show_agents:
            lines.append("")
            lines.append(f"### Agent Details (showing {len(show_agents)} of {total})")
            lines.append("")
            lines.append("| Agent ID | Name | IP Address | OS | Version | Status | Last Keep Alive |")
            lines.append("|----------|------|------------|----|---------:|--------|-----------------|")

            for agent in show_agents:
                agent_id = agent.get('id', 'N/A')
                name = agent.get('name', 'N/A')
                ip = agent.get('ip', 'N/A')
                os_info = agent.get('os', {})
                os_name = f"{os_info.get('name', 'N/A')} {os_info.get('version', '')}".strip() if isinstance(os_info, dict) else 'N/A'
                version = str(agent.get('version', 'N/A')).replace('Wazuh v', '')
                status = agent.get('status', 'N/A')
                last_ka = str(agent.get('lastKeepAlive', 'N/A'))
                if last_ka and last_ka != 'N/A':
                    last_ka = last_ka[:19].replace('T', ' ')

                lines.append(
                    f"| {agent_id} | {name} | {ip} | {os_name} | {version} | {status} | {last_ka} |"
                )

            if total > 10:
                lines.append("")
                lines.append(f"*...and {total - 10} more agents. Use `show active agents` or `show disconnected agents` for filtered views.*")

        lines.append("")
        lines.append(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show active agents` | `show disconnected agents` | `check agent 001 health` | `show vulnerabilities`")

        return "\n".join(lines)

    # ================================================================
    # PALLAS 3.5: ADVANCED SOC CORRELATION FORMATTERS
    # ================================================================

    def format_temporal_attack_sequence_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Temporal attack sequence analysis.
        Shows multi-phase attack sequences detected across Wazuh + Suricata.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            sequences = data.get("correlated_data", [])
        else:
            sequences = data if isinstance(data, list) else []
            summary = summary or {}

        total_seqs = summary.get("total_sequences", len(sequences))
        critical = summary.get("critical_sequences", 0)
        high = summary.get("high_sequences", 0)
        total_events = summary.get("total_events_analyzed") or 0
        wazuh_ev = summary.get("wazuh_events") or 0
        suricata_ev = summary.get("suricata_events") or 0
        cross_src = summary.get("cross_source_sequences") or 0
        progressive = summary.get("progressive_sequences") or 0

        lines = [
            "## Temporal Attack Sequence Analysis",
            "",
            f"**Sequences Detected:** {total_seqs} | "
            f"**Critical:** {critical} | **High:** {high} | "
            f"**Events Analyzed:** {total_events:,}",
            f"**Sources:** Wazuh ({wazuh_ev:,}) + Suricata ({suricata_ev:,}) | "
            f"**Cross-Source:** {cross_src} | **Progressive Kill-Chain:** {progressive}",
            "",
        ]

        if not sequences:
            lines.append("No multi-phase attack sequences detected in the analyzed time window.")
            lines.append("")
            lines.append("*This is a positive indicator — no entities show correlated multi-stage attack behavior.*")
        else:
            for idx, seq in enumerate(sequences[:10], 1):
                risk_level = seq.get("risk_level", "MEDIUM")
                risk_score = seq.get("risk_score", 0)
                entity_name = seq.get("entity_name", seq.get("entity", "Unknown"))
                entity_raw = seq.get("entity", "")
                phases = seq.get("phases", [])
                event_count = seq.get("event_count", 0)
                first_ts = str(seq.get("first_timestamp", "N/A"))[:19].replace("T", " ")
                last_ts = str(seq.get("last_timestamp", "N/A"))[:19].replace("T", " ")
                cross_src_flag = " | Cross-Source" if seq.get("cross_source") else ""
                progressive_flag = " | Kill-Chain Progressive" if seq.get("is_progressive") else ""

                lines += [
                    f"### Sequence {idx} (Risk: {risk_level} — Score: {risk_score}/100)",
                    f"**Entity:** {entity_name} ({entity_raw})",
                    f"**Duration:** {first_ts} → {last_ts} | **Events:** {event_count} | **Phases:** {len(phases)}/8{cross_src_flag}{progressive_flag}",
                    f"**Attack Phases:** {' → '.join(str(p) for p in phases)}",
                    "",
                    "| Time | Phase | Event | Severity | Source |",
                    "|------|-------|-------|:--------:|--------|",
                ]

                events = seq.get("events", [])
                for ev in events[:15]:
                    ts = str(ev.get("timestamp", "N/A"))[:19].replace("T", " ")
                    phase = ev.get("phase", "?")
                    desc = str(ev.get("description") or "N/A")[:60]
                    sev = ev.get("severity", 0)
                    source = ev.get("source", "?")
                    sev_label = "Critical" if sev >= 12 else "High" if sev >= 8 else "Medium" if sev >= 5 else "Low"
                    lines.append(f"| {ts} | {phase} | {desc} | {sev_label} ({sev}) | {source} |")

                if len(events) > 15:
                    lines.append(f"\n*...and {len(events) - 15} more events in this sequence*")
                lines.append("")

        lines += [
            "---",
            "**Related queries:** `show attack chain analysis` | `calculate risk score` | `show recent alerts` | `show MITRE coverage`",
        ]

        return "\n".join(lines)

    def format_multi_asset_campaign_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Multi-asset campaign detection formatter.
        Shows campaigns, attack graph, shadow IT, and encryption anomalies.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            campaigns = data.get("correlated_data", [])
        else:
            campaigns = data if isinstance(data, list) else []
            summary = summary or {}

        total_campaigns = summary.get("total_campaigns", len(campaigns))
        wazuh_analyzed = summary.get("wazuh_alerts_analyzed") or 0
        suricata_analyzed = summary.get("suricata_alerts_analyzed") or 0
        agents_in_campaigns = summary.get("agents_in_campaigns", 0)
        graph_edges = summary.get("attack_graph_edges", [])
        graph_nodes = summary.get("attack_graph_node_count", 0)
        shadow_it = summary.get("shadow_it", [])
        shadow_count = summary.get("shadow_it_count", 0)
        enc_anomalies = summary.get("encryption_anomalies", [])
        enc_count = summary.get("encryption_anomaly_count", 0)

        lines = [
            "## Multi-Asset Campaign Detection",
            "",
            f"**Alerts Analyzed:** Wazuh ({wazuh_analyzed:,}) + Suricata ({suricata_analyzed:,}) | "
            f"**Campaigns Detected:** {total_campaigns} | "
            f"**Agents in Campaigns:** {agents_in_campaigns}",
            "",
        ]

        # Campaign Detection
        if campaigns:
            lines += [
                "### Coordinated Campaigns",
                "",
                "| Indicator Type | Indicator | Agents Affected | Agent Names |",
                "|---------------|-----------|:---------------:|-------------|",
            ]
            for c in campaigns[:15]:
                itype = c.get("indicator_type", "?")
                ival = str(c.get("indicator_value") or "?")[:40]
                count = c.get("agents_affected", 0)
                names = ", ".join(str(n) for n in c.get("agent_names", [])[:5])
                lines.append(f"| {itype} | {ival} | {count} | {names} |")
            lines.append("")

            # Highlight top campaigns
            top = campaigns[:3]
            if top:
                lines.append("### Campaign Details")
                lines.append("")
                for idx, c in enumerate(top, 1):
                    lines.append(f"**Campaign {idx}:** {c.get('indicator_type', '?')}:{c.get('indicator_value', '?')}")
                    lines.append(f"  - Agents affected: {c.get('agents_affected', 0)}")
                    names = c.get("agent_names", [])
                    if names:
                        lines.append(f"  - Agents: {', '.join(str(n) for n in names[:8])}")
                    lines.append("")
        else:
            lines.append("No coordinated multi-asset campaigns detected.")
            lines.append("")

        # Attack Graph
        if graph_edges:
            lines += [
                f"### Attack Graph ({graph_nodes} nodes, {len(graph_edges)} edges)",
                "",
                "| Source | → | Target | Alert Type |",
                "|--------|---|--------|-----------|",
            ]
            for e in graph_edges[:15]:
                lines.append(f"| {e.get('source', '?')} | → | {e.get('target', '?')} | {e.get('alert_type', '?')} |")
            lines.append("")

        # Shadow IT
        if shadow_it:
            lines += [
                f"### Shadow IT / Unmanaged Devices ({shadow_count} detected)",
                "",
                "*Internal IPs not matching any Wazuh agent — possible unmanaged devices.*",
                "",
                "| Internal IP | Connections | External Targets | Risk |",
                "|-------------|:----------:|:----------------:|:----:|",
            ]
            for s in shadow_it[:10]:
                lines.append(f"| {s.get('ip', '?')} | {s.get('connections', 0)} | {s.get('external_targets', 0)} | {s.get('risk', '?')} |")
            lines.append("")

        # Encryption Anomalies
        if enc_anomalies:
            lines += [
                f"### Anomalous Encryption ({enc_count} detected)",
                "",
                "| Src IP | Dest IP | TLS Version | JA3 | Anomaly |",
                "|--------|---------|-------------|-----|---------|",
            ]
            for ea in enc_anomalies[:10]:
                reasons = "; ".join(str(r) for r in (ea.get("anomaly_reasons") or []))
                lines.append(f"| {ea.get('src_ip', '?')} | {ea.get('dest_ip', '?')} | {ea.get('tls_version', '?')} | {ea.get('ja3_hash', '?')} | {reasons} |")
            lines.append("")

        lines += [
            "---",
            "**Related queries:** `show attack sequence` | `calculate risk score` | `show MITRE coverage` | `show suricata alerts`",
        ]

        return "\n".join(lines)

    def format_alert_noise_analysis_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Alert noise analysis — FP candidates, tuning recommendations, stealth chains.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            fp_candidates = data.get("correlated_data", [])
        else:
            fp_candidates = data if isinstance(data, list) else []
            summary = summary or {}

        total_rules = summary.get("total_rules_analyzed", 0)
        total_fps = summary.get("fp_candidates", len(fp_candidates))
        high_fps = summary.get("high_confidence_fps", 0)
        tuning_recs = summary.get("tuning_recommendations", [])
        stealth_chains = summary.get("stealth_chains", [])
        stealth_count = summary.get("stealth_chain_count", 0)

        lines = [
            "## Alert Noise Analysis",
            "",
            f"**Rules Analyzed:** {total_rules} | "
            f"**False Positive Candidates:** {total_fps} | "
            f"**High-Confidence FPs:** {high_fps}",
            f"**Low-Severity Stealth Chains:** {stealth_count} agents",
            "",
        ]

        # False Positive Candidates
        if fp_candidates:
            lines += [
                "### False Positive Candidates",
                "",
                "| Rule ID | Description | Level | Fires | FP Score | Agents | Top Reason |",
                "|---------|------------|:-----:|------:|:--------:|:------:|-----------|",
            ]
            for fp in fp_candidates[:15]:
                rid = fp.get("rule_id", "?")
                desc = str(fp.get("description") or "?")[:40]
                level = fp.get("level", "?")
                count = fp.get("count") or 0
                score = fp.get("fp_score") or 0
                agents = fp.get("agents_affected", 0)
                reasons = fp.get("fp_reasons") or ["?"]
                lines.append(f"| {rid} | {desc} | {level} | {count:,} | {score} | {agents} | {reasons[0] if reasons else '?'} |")
            lines.append("")

            # Detail for top FPs
            lines.append("### FP Analysis Detail")
            lines.append("")
            for fp in fp_candidates[:5]:
                lines.append(f"**Rule {fp.get('rule_id', '?')}:** {fp.get('description', '?')}")
                for reason in fp.get("fp_reasons", []):
                    lines.append(f"  - {reason}")
                groups = fp.get("groups", [])
                if groups:
                    lines.append(f"  - Groups: {', '.join(str(g) for g in groups[:5])}")
                lines.append("")
        else:
            lines.append("No significant false positive candidates identified.")
            lines.append("")

        # Tuning Recommendations
        if tuning_recs:
            lines += [
                f"### Tuning Recommendations ({len(tuning_recs)})",
                "",
            ]
            for idx, rec in enumerate(tuning_recs[:10], 1):
                action = rec.get("action", "review")
                icon = "EXCLUDE" if action == "exclude_or_tune" else "REVIEW"
                lines.append(f"{idx}. **[{icon}]** {rec.get('recommendation', '?')}")
            lines.append("")

        # Stealth Chain Detection
        if stealth_chains:
            lines += [
                f"### Low-Severity Stealth Chain Detection ({stealth_count} agents)",
                "",
                "*Agents with 5+ different low-severity rules may indicate stealthy attack techniques that avoid high-severity triggers.*",
                "",
                "| Agent | Distinct Low-Sev Rules | Risk | Rule IDs (sample) |",
                "|-------|:---------------------:|:----:|-------------------|",
            ]
            for sc in stealth_chains[:10]:
                name = sc.get("agent_name", "?")
                count = sc.get("distinct_low_sev_rules", 0)
                risk = sc.get("risk_level", "?")
                rules = ", ".join(str(r) for r in (sc.get("rule_ids") or [])[:5])
                lines.append(f"| {name} | {count} | {risk} | {rules} |")
            lines.append("")

        lines += [
            "---",
            "**Related queries:** `show recent alerts` | `show rule trigger analysis` | `show attack sequence` | `show alert timeline`",
        ]

        return "\n".join(lines)

    def format_off_hours_anomaly_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Off-hours anomaly detection formatter.
        Shows off-hours activity breakdown by agent with anomaly flags.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            agents = data.get("correlated_data", [])
        else:
            agents = data if isinstance(data, list) else []
            summary = summary or {}

        total_events = summary.get("total_events_analyzed") or 0
        total_off = summary.get("total_off_hours_events") or 0
        agents_analyzed = summary.get("agents_analyzed") or 0
        anomalous_count = summary.get("anomalous_agents") or 0
        biz_hours = summary.get("business_hours", "08:00-18:00")
        suricata_off = summary.get("suricata_off_hours", [])
        suricata_count = summary.get("suricata_off_hours_count", 0)

        off_ratio = round(total_off / max(total_events, 1) * 100, 1)

        lines = [
            "## Off-Hours Anomaly Detection",
            "",
            f"**Business Hours:** {biz_hours} | "
            f"**Events Analyzed:** {total_events:,} | "
            f"**Off-Hours Events:** {total_off:,} ({off_ratio}%)",
            f"**Agents Analyzed:** {agents_analyzed} | "
            f"**Anomalous Agents:** {anomalous_count}",
            "",
        ]

        if agents:
            # Summary table
            lines += [
                "### Agent Off-Hours Activity",
                "",
                "| Agent | Total Events | Off-Hours | Ratio | Suspicious | Anomalous |",
                "|-------|:-----------:|:---------:|:-----:|:----------:|:---------:|",
            ]
            for a in agents[:15]:
                name = a.get("agent_name", "?")
                total = a.get("total_events") or 0
                off = a.get("off_hours_events") or 0
                ratio = f"{(a.get('off_hours_ratio') or 0) * 100:.1f}%"
                susp = a.get("suspicious_count", 0)
                anom = "YES" if a.get("is_anomalous") else "No"
                lines.append(f"| {name} | {total:,} | {off:,} | {ratio} | {susp} | {anom} |")
            lines.append("")

            # Detailed anomalous agents
            anomalous = [a for a in agents if a.get("is_anomalous")][:5]
            if anomalous:
                lines.append("### Anomalous Off-Hours Activity")
                lines.append("")
                for a in anomalous:
                    name = a.get("agent_name", "?")
                    ratio = f"{(a.get('off_hours_ratio') or 0) * 100:.1f}%"
                    lines.append(f"**{name}** ({a.get('agent_ip', '?')}) — {ratio} off-hours activity")

                    suspicious = a.get("suspicious_patterns", [])
                    if suspicious:
                        lines.append("  Suspicious patterns:")
                        for sp in suspicious[:3]:
                            lines.append(f"  - [{sp.get('hour', '?')}:00] {sp.get('description', '?')} (Level {sp.get('level', '?')})")

                    events = a.get("sample_events", [])
                    if events and not suspicious:
                        lines.append("  Sample off-hours events:")
                        for ev in events[:3]:
                            lines.append(f"  - [{ev.get('hour', '?')}:00] {ev.get('description', '?')} (Level {ev.get('level', '?')})")
                    lines.append("")
        else:
            lines.append("No significant off-hours activity detected.")
            lines.append("")

        # Suricata off-hours network activity
        if suricata_off:
            lines += [
                f"### Off-Hours Network Activity ({suricata_count} Suricata events)",
                "",
                "| Time | Src IP | Dest IP | Signature | Severity |",
                "|------|--------|---------|-----------|:--------:|",
            ]
            for s in suricata_off[:10]:
                lines.append(f"| {s.get('timestamp', '?')} | {s.get('src_ip', '?')} | {s.get('dest_ip', '?')} | {s.get('signature', '?')} | {s.get('severity', '?')} |")
            lines.append("")

        lines += [
            "---",
            "**Related queries:** `show attack sequence` | `show recent alerts` | `detect data exfiltration` | `show agent health overview`",
        ]

        return "\n".join(lines)

    def format_data_exfiltration_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Data exfiltration detection formatter.
        Shows file access patterns, compression activity, and outbound transfer indicators.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            indicators = data.get("correlated_data", [])
        else:
            indicators = data if isinstance(data, list) else []
            summary = summary or {}

        total_fim = summary.get("total_fim_events") or 0
        agents_fim = summary.get("agents_with_fim") or 0
        agents_comp = summary.get("agents_with_compression") or 0
        outbound = summary.get("outbound_indicators", [])
        outbound_count = summary.get("outbound_count", 0)
        crit = summary.get("critical_risk", 0)
        high = summary.get("high_risk", 0)

        lines = [
            "## Data Exfiltration Detection",
            "",
            f"**FIM Events Analyzed:** {total_fim:,} | **Agents with File Activity:** {agents_fim} | "
            f"**Compression Detected:** {agents_comp} agents",
            f"**Risk Assessment:** {crit} CRITICAL | {high} HIGH | "
            f"**Outbound Indicators:** {outbound_count}",
            "",
        ]

        if indicators:
            lines += [
                "### Exfiltration Risk by Agent",
                "",
                "| Agent | Risk Score | Risk Level | Files | Sensitive | Compression | Indicators |",
                "|-------|:----------:|:----------:|------:|----------:|:-----------:|-----------|",
            ]
            for ind in indicators[:15]:
                name = ind.get("agent_name", "?")
                score = ind.get("risk_score", 0)
                level = ind.get("risk_level", "?")
                files = ind.get("file_count", 0)
                sens = ind.get("sensitive_files", 0)
                comp = ", ".join(str(t) for t in (ind.get("compression_tools") or [])) or "None"
                inds = "; ".join(str(x) for x in (ind.get("indicators") or []))[:60]
                lines.append(f"| {name} | {score} | {level} | {files} | {sens} | {comp} | {inds} |")
            lines.append("")

            # Detailed view for critical/high risk
            critical_high = [i for i in indicators if i.get("risk_level", "UNKNOWN") in ("CRITICAL", "HIGH")][:5]
            if critical_high:
                lines.append("### Detailed Evidence")
                lines.append("")
                for ind in critical_high:
                    name = ind.get("agent_name", "?")
                    lines.append(f"**{name}** (Risk: {ind.get('risk_level', '?')} — Score: {ind.get('risk_score', 0)})")
                    for indicator in ind.get("indicators", []):
                        lines.append(f"  - {indicator}")
                    events = ind.get("sample_events", [])
                    if events:
                        lines.append("  - Sample file events:")
                        for ev in events[:3]:
                            lines.append(f"    - `{ev.get('path', '?')}` — {ev.get('event', '?')} ({ev.get('timestamp', '?')})")
                    dirs = ind.get("directories", [])
                    if dirs:
                        lines.append(f"  - Directories: {', '.join(dirs[:5])}")
                    lines.append("")
        else:
            lines.append("No significant data exfiltration indicators detected.")
            lines.append("")

        # Outbound transfer section
        if outbound:
            lines += [
                f"### Outbound Transfer Indicators ({len(outbound)} detected)",
                "",
                "| Signature | Src IP | Dest IP | Severity | Time |",
                "|-----------|--------|---------|:--------:|------|",
            ]
            for o in outbound[:10]:
                lines.append(f"| {o.get('signature', '?')} | {o.get('src_ip', '?')} | {o.get('dest_ip', '?')} | {o.get('severity', '?')} | {o.get('timestamp', '?')} |")
            lines.append("")

        lines += [
            "---",
            "**Related queries:** `show FIM events` | `show processes` | `show attack sequence` | `calculate risk score`",
        ]

        return "\n".join(lines)

    def format_cross_platform_threat_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Cross-platform threat correlation (Suricata + Wazuh endpoint).
        Shows correlated network + endpoint threats and C2 beaconing suspects.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            threats = data.get("correlated_data", [])
        else:
            threats = data if isinstance(data, list) else []
            summary = summary or {}

        total_suricata = summary.get("total_suricata_alerts") or 0
        total_correlated = summary.get("total_correlated") or len(threats)
        high_conf = summary.get("high_confidence", 0)
        medium_conf = summary.get("medium_confidence", 0)
        agents_affected = summary.get("agents_affected", 0)
        beaconing = summary.get("beaconing_suspects", [])
        beacon_count = summary.get("beaconing_count", 0)

        lines = [
            "## Cross-Platform Threat Correlation",
            "",
            f"**Suricata Alerts Analyzed:** {total_suricata:,} | "
            f"**Correlated with Endpoint:** {total_correlated} | "
            f"**Agents Affected:** {agents_affected}",
            f"**Confidence:** {high_conf} HIGH (network + endpoint match) | {medium_conf} MEDIUM (network only)",
            "",
        ]

        if threats:
            lines += [
                "### Correlated Threats (Network + Endpoint)",
                "",
                "| Suricata Signature | Severity | Agent | Wazuh Alerts | Confidence | Src IP | Dest IP |",
                "|-------------------|:--------:|-------|:-------------|:----------:|--------|---------|",
            ]
            for t in threats[:20]:
                sig = str(t.get("suricata_signature") or "?")[:50]
                sev = t.get("suricata_severity", "?")
                agent = t.get("agent_name", "?")
                wc = t.get("wazuh_alert_count", 0)
                conf = t.get("confidence", "?")
                src = t.get("src_ip", "?")
                dst = t.get("dest_ip", "?")
                lines.append(f"| {sig} | {sev} | {agent} | {wc} | {conf} | {src} | {dst} |")
            lines.append("")

            # Show endpoint correlation details for high-confidence entries
            high_threats = [t for t in threats if t.get("confidence") == "HIGH"][:5]
            if high_threats:
                lines.append("### High-Confidence Correlations (Endpoint Evidence)")
                lines.append("")
                for t in high_threats:
                    lines.append(f"**{t.get('agent_name', '?')}** — {t.get('suricata_signature', '?')}")
                    wazuh_rules = t.get("wazuh_top_rules", [])
                    if wazuh_rules:
                        for wr in wazuh_rules:
                            lines.append(f"  - Rule {wr.get('id', '?')} (Level {wr.get('level', '?')}): {wr.get('description', '?')}")
                    lines.append("")
        else:
            lines.append("No cross-platform threat correlations found.")
            lines.append("")

        # C2 Beaconing Section
        if beaconing:
            lines += [
                f"### C2 Beaconing Suspects ({beacon_count} detected)",
                "",
                "| Destination IP | Agent | Connections | Avg Interval | CV | Confidence |",
                "|---------------|-------|:-----------:|:------------:|:--:|:----------:|",
            ]
            for b in beaconing[:10]:
                dest = b.get("dest_ip", "?")
                agent = b.get("agent_name", "?")
                count = b.get("connection_count", 0)
                avg = b.get("avg_interval_sec", 0)
                cv = b.get("coefficient_variation", 0)
                conf = b.get("beacon_confidence", "?")
                # Format interval
                if avg >= 3600:
                    interval_str = f"{avg/3600:.1f}h"
                elif avg >= 60:
                    interval_str = f"{avg/60:.1f}m"
                else:
                    interval_str = f"{avg:.0f}s"
                lines.append(f"| {dest} | {agent} | {count} | {interval_str} | {cv:.3f} | {conf} |")
            lines += [
                "",
                "*Low coefficient of variation (CV < 0.3) indicates periodic traffic consistent with C2 beaconing.*",
                "",
            ]

        lines += [
            "---",
            "**Related queries:** `show suricata alerts` | `show attack sequence` | `calculate risk score` | `show MITRE coverage`",
        ]

        return "\n".join(lines)

    def format_dynamic_risk_scoring_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.5: Dynamic risk scoring — composite risk per agent.
        Shows vulnerability, alert, IDS, exposure, and compliance dimensions.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            agents = data.get("correlated_data", [])
        else:
            agents = data if isinstance(data, list) else []
            summary = summary or {}

        total = summary.get("total_agents", len(agents))
        critical = summary.get("critical_risk", 0)
        high = summary.get("high_risk", 0)
        medium = summary.get("medium_risk", 0)
        low = summary.get("low_risk", 0)
        avg_score = summary.get("avg_score", 0)

        lines = [
            "## Dynamic Risk Assessment",
            "",
            f"**Agents Analyzed:** {total} | "
            f"**Critical:** {critical} | **High:** {high} | **Medium:** {medium} | **Low:** {low}",
            f"**Average Risk Score:** {avg_score}/100",
            "",
        ]

        if not agents:
            lines.append("No agent risk data available.")
        else:
            # Summary table
            lines += [
                "### Risk Ranking",
                "",
                "| Rank | Agent | IP | Risk Score | Vuln | Alert | IDS | Exposure | Status |",
                "|-----:|-------|-----|:----------:|-----:|------:|----:|---------:|--------|",
            ]
            for idx, a in enumerate(agents[:20], 1):
                name = a.get("agent_name", "N/A")
                ip = a.get("agent_ip", "N/A")
                score = a.get("composite_score", 0)
                risk = a.get("risk_level", "?")
                vs = a.get("vuln_score", 0)
                als = a.get("alert_score", 0)
                ids = a.get("ids_score", 0)
                exp = a.get("exposure_score", 0)
                status = a.get("agent_status", "?")
                lines.append(f"| {idx} | {name} | {ip} | **{score}** {risk} | {vs} | {als} | {ids} | {exp} | {status} |")
            lines.append("")

            # Detailed breakdown for top 5
            for a in agents[:5]:
                name = a.get("agent_name", "N/A")
                score = a.get("composite_score", 0)
                risk = a.get("risk_level", "?")
                lines.append(f"### Risk Breakdown: {name} (Score: {score} {risk})")
                lines.append("")

                # Vulnerability detail
                vd = a.get("vuln_detail", {})
                vs = a.get("vuln_score", 0)
                lines.append(f"- **Vulnerability Risk ({vs}):** {vd.get('critical', 0)} Critical, {vd.get('high', 0)} High, {vd.get('medium', 0)} Medium ({vd.get('total', 0)} total)")
                top_cves = vd.get("top_cves", [])
                if top_cves:
                    cve_str = ", ".join(f"{c.get('cve', '?')} (CVSS {c.get('cvss', '?')})" for c in top_cves[:3])
                    lines.append(f"  - Top CVEs: {cve_str}")

                # Alert detail
                ad = a.get("alert_detail", {})
                als = a.get("alert_score", 0)
                lines.append(f"- **Alert Risk ({als}):** {ad.get('critical', 0)} Critical, {ad.get('high', 0)} High, {ad.get('medium', 0)} Medium ({ad.get('total', 0)} total)")
                mitre = ad.get("mitre_tactics", [])
                if mitre:
                    lines.append(f"  - MITRE Tactics: {', '.join(mitre[:5])}")

                # IDS detail
                idsd = a.get("ids_detail", {})
                ids = a.get("ids_score", 0)
                lines.append(f"- **IDS Risk ({ids}):** {idsd.get('count', 0)} Suricata detections")
                sigs = idsd.get("signatures", [])
                if sigs:
                    lines.append(f"  - Signatures: {', '.join(sigs[:3])}")

                # Exposure detail
                pd_detail = a.get("port_detail", {})
                exp = a.get("exposure_score", 0)
                ports = pd_detail.get("ports", [])
                hr = pd_detail.get("high_risk", 0)
                lines.append(f"- **Exposure Risk ({exp}):** {pd_detail.get('total', 0)} open ports ({hr} high-risk)")
                if ports:
                    lines.append(f"  - Ports: {', '.join(str(p) for p in ports[:10])}")

                lines.append("")

        lines += [
            "---",
            "**Related queries:** `show critical vulnerabilities` | `show recent alerts` | `show attack sequence` | `show agent health overview`",
        ]

        return "\n".join(lines)

    # ================================================================
    # PALLAS 3.4: AGENT HEALTH ANALYTICS FORMATTERS
    # ================================================================

    def format_agent_health_overview_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.4: Agent fleet health overview.
        Shows OS grouping, version comparison, staleness, enrollment recency.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
            agents = data.get("correlated_data", [])
        else:
            agents = data if isinstance(data, list) else []
            summary = summary or {}

        total = summary.get("total_agents", len(agents))
        os_groups = summary.get("os_groups", {})
        version_counts = summary.get("version_counts", {})
        majority_version = summary.get("majority_version", "Unknown")
        mismatched = summary.get("mismatched_agents", [])
        stale = summary.get("stale_agents", [])
        recently_enrolled = summary.get("recently_enrolled", [])

        lines = [
            "## Agent Fleet Health Overview",
            "",
            f"**Total Agents:** {total} (excluding manager)",
            "",
        ]

        # OS Distribution
        if os_groups:
            lines += [
                "### OS Distribution",
                "",
                "| Operating System | Count | Percentage |",
                "|-----------------|------:|-----------:|",
            ]
            for os_key, count in sorted(os_groups.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"| {os_key} | {count} | {pct:.1f}% |")
            lines.append("")

        # Version Analysis
        if version_counts:
            lines += [
                "### Agent Version Analysis",
                "",
                f"**Majority Version:** {majority_version}",
                "",
                "| Version | Count | Status |",
                "|---------|------:|--------|",
            ]
            for ver, count in sorted(version_counts.items(), key=lambda x: x[1], reverse=True):
                status = "Current" if ver == majority_version else "Mismatch"
                lines.append(f"| {ver} | {count} | {status} |")
            lines.append("")

            if mismatched:
                lines += [
                    f"### Version Mismatches ({len(mismatched)} agents)",
                    "",
                    "| Agent ID | Name | Version | Status |",
                    "|----------|------|---------|--------|",
                ]
                for a in mismatched[:15]:
                    aid = a.get("id", "N/A")
                    name = a.get("name", "N/A")
                    ver = str(a.get("version", "N/A")).replace("Wazuh v", "")
                    status = a.get("status", "N/A")
                    lines.append(f"| {aid} | {name} | {ver} | {status} |")
                if len(mismatched) > 15:
                    lines.append(f"\n*...and {len(mismatched) - 15} more mismatched agents*")
                lines.append("")

        # Stale Agents
        if stale:
            lines += [
                f"### Stale Agents ({len(stale)} agents not reporting >6 hours)",
                "",
                "| Agent ID | Name | Last Contact | Hours Since |",
                "|----------|------|-------------|------------:|",
            ]
            for a in sorted(stale, key=lambda x: x.get("_hours_since_contact", 0), reverse=True)[:15]:
                aid = a.get("id", "N/A")
                name = a.get("name", "N/A")
                last_ka = str(a.get("lastKeepAlive", "N/A"))[:19].replace("T", " ")
                hours = a.get("_hours_since_contact", "?")
                lines.append(f"| {aid} | {name} | {last_ka} | {hours}h |")
            lines.append("")

        # Recently Enrolled
        if recently_enrolled:
            lines += [
                f"### Recently Enrolled ({len(recently_enrolled)} agents in last 7 days)",
                "",
                "| Agent ID | Name | Enrolled | Days Ago |",
                "|----------|------|----------|--------:|",
            ]
            for a in sorted(recently_enrolled, key=lambda x: x.get("_days_since_enrollment", 0))[:15]:
                aid = a.get("id", "N/A")
                name = a.get("name", "N/A")
                reg_date = str(a.get("dateAdd", a.get("registered_date", "N/A")))[:19].replace("T", " ")
                days = a.get("_days_since_enrollment", "?")
                lines.append(f"| {aid} | {name} | {reg_date} | {days} |")
            lines.append("")

        if not os_groups and not version_counts and not stale and not recently_enrolled:
            lines.append("*No detailed fleet health data available. Try `show all agents` for basic listing.*")

        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show active agents` | `show disconnected agents` | `show log source health`")

        return "\n".join(lines)

    def format_agent_event_volume_response(self, data, summary=None, **kwargs):
        """
        Pallas 3.4: Agent event volume anomaly report.
        Highlights silent, high-volume, and low-volume agents relative to fleet average.
        """
        if isinstance(data, dict):
            summary = summary or data.get("summary", {})
        else:
            summary = summary or {}

        total_agents = summary.get("total_agents_reporting", 0)
        avg_count = summary.get("average_event_count", 0)
        total_events = summary.get("total_events", 0)
        anomalies = summary.get("anomalies", [])

        lines = [
            "## Agent Event Volume Analysis",
            "",
            f"**Total Agents Reporting:** {total_agents} | **Total Events:** {total_events:,} | **Average per Agent:** {avg_count:,.1f}",
            "",
        ]

        if anomalies:
            # Group by anomaly type
            silent = [a for a in anomalies if a.get("anomaly_type") == "silent"]
            high_vol = [a for a in anomalies if a.get("anomaly_type") == "high_volume"]
            low_vol = [a for a in anomalies if a.get("anomaly_type") == "low_volume"]

            if silent:
                lines += [
                    f"### Silent Agents ({len(silent)} agents with 0 events)",
                    "",
                    "| Agent ID | Name | Status | OS |",
                    "|----------|------|--------|-----|",
                ]
                for a in silent:
                    lines.append(f"| {a.get('agent_id', 'N/A')} | {a.get('agent_name', 'N/A')} | {a.get('agent_status', 'N/A')} | {a.get('agent_os', 'N/A')} |")
                lines += ["", "*Silent agents may indicate broken log collection, disabled modules, or network issues.*", ""]

            if high_vol:
                lines += [
                    f"### High Volume Agents ({len(high_vol)} agents > 2x average)",
                    "",
                    "| Agent ID | Name | Events | vs Average | Deviation |",
                    "|----------|------|-------:|:-----------|----------:|",
                ]
                for a in sorted(high_vol, key=lambda x: x.get("event_count", 0), reverse=True):
                    count = a.get("event_count", 0)
                    dev = a.get("deviation_factor", 0)
                    lines.append(f"| {a.get('agent_id', 'N/A')} | {a.get('agent_name', 'N/A')} | {count:,} | {avg_count:,.1f} | {dev:.1f}x |")
                lines += ["", "*High volume may indicate noisy rules, active attacks, or misconfigured logging.*", ""]

            if low_vol:
                lines += [
                    f"### Low Volume Agents ({len(low_vol)} agents < half average)",
                    "",
                    "| Agent ID | Name | Events | vs Average | Deviation |",
                    "|----------|------|-------:|:-----------|----------:|",
                ]
                for a in sorted(low_vol, key=lambda x: x.get("event_count", 0)):
                    count = a.get("event_count", 0)
                    dev = a.get("deviation_factor", 0)
                    lines.append(f"| {a.get('agent_id', 'N/A')} | {a.get('agent_name', 'N/A')} | {count:,} | {avg_count:,.1f} | {dev:.1f}x |")
                lines.append("")
        else:
            lines.append("All agents are reporting event volumes within normal range.")
            lines.append("")

        lines.append("---")
        lines.append("**Related queries:** `show agent health overview` | `check agent health` | `show log source health`")

        return "\n".join(lines)

    # ================================================================
    # VULNERABILITY FORMATTERS
    # ================================================================

    def _deduplicate_vulns(self, vulns: List[Dict]) -> List[Dict]:
        """Deduplicate vulnerabilities by CVE ID, aggregate affected agents."""
        seen = {}
        for v in vulns:
            cve_id = v.get('id') or v.get('cve') or v.get('cve_id') or v.get('name') or 'unknown'
            agent_id = None
            agent_data = v.get('agent', {})
            if isinstance(agent_data, dict):
                agent_id = agent_data.get('id')
            if not agent_id:
                agent_id = v.get('agent_id')

            if cve_id not in seen:
                seen[cve_id] = {**v, '_affected_agents': set(), '_occurrence_count': 0}
            seen[cve_id]['_occurrence_count'] += 1
            if agent_id:
                seen[cve_id]['_affected_agents'].add(agent_id)

        result = []
        for cve_id, v in seen.items():
            v['affected_agent_count'] = len(v['_affected_agents'])
            v['affected_agents'] = sorted(v['_affected_agents'])
            del v['_affected_agents']
            result.append(v)
        return result

    def format_vulnerability_response(self, correlated_data, summary, **kwargs):
        """
        Format vulnerability data with deduplication and affected agent counts.
        """
        vulns = self._extract_data(correlated_data)

        logger.info(f"[VULN_FORMATTER] Received {len(vulns)} vulnerabilities")

        if not vulns:
            return "✅ No vulnerabilities found"

        # Deduplicate by CVE ID
        vulns = self._deduplicate_vulns(vulns)
        logger.info(f"[VULN_FORMATTER] After dedup: {len(vulns)} unique CVEs")

        # Count all severities for summary header (case-insensitive normalize to titlecase)
        all_severity_counts = {}
        for v in vulns:
            sev = str(v.get('severity', 'Unknown') or 'Unknown')
            sev_norm = sev.title() if sev.lower() in ('critical', 'high', 'medium', 'low', 'unknown') else sev
            all_severity_counts[sev_norm] = all_severity_counts.get(sev_norm, 0) + 1

        # Decide which view to render based on USER INTENT, not data presence.
        # Intent signals (in priority order):
        #   1. summary['tool'] == 'get_wazuh_critical_vulnerabilities' (explicit critical tool)
        #   2. summary['severity'] or kwargs['severity'] == 'critical' (filter applied)
        # Without an intent signal, render the full breakdown including CVE details.
        summary_dict = summary if isinstance(summary, dict) else {}
        tool_name = str(summary_dict.get('tool', '') or kwargs.get('tool', '')).lower()
        severity_filter = str(summary_dict.get('severity', '') or kwargs.get('severity', '') or '').lower()
        os_filter = str(summary_dict.get('os', '') or kwargs.get('os', '') or '').lower()
        is_critical_query = (
            tool_name == 'get_wazuh_critical_vulnerabilities'
            or severity_filter == 'critical'
        )

        # Defensive OS filter: drop CVEs whose agent OS doesn't match the queried
        # OS (CLAUDE.md §1 STRICT FILTERING — never trust the data layer alone).
        # Match against ALL OS metadata fields the indexer might populate.
        if os_filter:
            os_keywords = self._os_aliases(os_filter)
            pre = len(vulns)

            def _vuln_os_matches(v):
                agent = v.get('agent') or {}
                # Concatenate every OS-bearing field into one searchable blob
                os_obj = agent.get('os') or {}
                blob_parts = [str(agent.get('type') or '')]
                if isinstance(os_obj, str):
                    blob_parts.append(os_obj)
                elif isinstance(os_obj, dict):
                    for k in ('platform', 'family', 'name', 'full', 'type', 'kernel'):
                        blob_parts.append(str(os_obj.get(k) or ''))
                blob = ' '.join(blob_parts).lower()
                if not blob.strip():
                    # Missing OS metadata — keep the doc rather than silently drop it
                    return True
                return any(kw in blob for kw in os_keywords)

            vulns = [v for v in vulns if _vuln_os_matches(v)]
            dropped = pre - len(vulns)
            if dropped:
                logger.info(
                    f"[VULN_FORMATTER] OS filter '{os_filter}' dropped {dropped} CVEs "
                    f"({len(vulns)} kept)"
                )

        if is_critical_query:
            # Always filter by the actual severity field — never trust the tool name
            # to declare data critical. This is the §1 STRICT FILTERING rule.
            critical_vulns = [
                v for v in vulns
                if str(v.get('severity', '') or '').lower() == 'critical'
            ]
            logger.info(
                f"[VULN_FORMATTER] critical-only path (intent=critical): "
                f"{len(critical_vulns)} of {len(vulns)} vulns are Critical"
            )
            if not critical_vulns:
                # Help the analyst see what severities ARE present
                breakdown_str = ", ".join(
                    f"{sev}: {cnt}" for sev, cnt in all_severity_counts.items() if cnt > 0
                ) or "none"
                return (
                    f"## Critical Vulnerabilities\n\n"
                    f"✅ No Critical-severity CVEs found.\n\n"
                    f"**Severities present in result:** {breakdown_str}\n\n"
                    f"_Total CVEs scanned:_ {len(vulns)}"
                )
            return self._format_critical_vulnerabilities(critical_vulns, len(vulns), all_severity_counts)

        # Otherwise: render the full breakdown with CVE details across all severities
        logger.info(f"[VULN_FORMATTER] all-vulns path with {len(vulns)} CVEs")
        return self._format_all_vulnerabilities(vulns)

    def _format_critical_vulnerabilities(self, vulns: List[Dict], total_all: int = 0, severity_counts: Dict = None) -> str:
        """Show critical vulnerabilities with summary header and full details."""
        if not vulns:
            return "No critical vulnerabilities found"

        total = len(vulns)
        top_10 = vulns[:10]

        lines = [
            f"## Critical Vulnerabilities",
            "",
        ]

        # Summary header with total count and severity breakdown
        if total_all and severity_counts:
            lines.append(f"**Total Vulnerabilities:** {total_all} unique CVEs")
            lines.append("")
            breakdown_parts = []
            for sev in ['Critical', 'High', 'Medium', 'Low']:
                count = severity_counts.get(sev, 0)
                if count > 0:
                    icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
                    breakdown_parts.append(f"{icon} {sev}: {count}")
            if breakdown_parts:
                lines.append(" | ".join(breakdown_parts))
                lines.append("")

        lines.extend([
            f"**Critical CVEs Found:** {total}",
            "",
            f"### Top {len(top_10)} Critical CVEs (Detailed)",
            ""
        ])

        for i, vuln in enumerate(top_10, 1):
            cve_id = vuln.get('cve_id') or vuln.get('cve') or vuln.get('id') or 'N/A'
            severity = vuln.get('severity', 'Critical')
            description = vuln.get('description', 'No description available')

            if len(description) > 250:
                description = description[:247] + "..."

            cvss_score = vuln.get('cvss_score')
            if cvss_score is None or cvss_score == 'N/A':
                cvss_score = 'N/A'
                if 'cvss' in vuln:
                    cvss_data = vuln.get('cvss', {})
                    if isinstance(cvss_data, dict):
                        if 'cvss3' in cvss_data:
                            cvss_score = cvss_data['cvss3'].get('base_score', 'N/A')
                        elif 'cvss2' in cvss_data:
                            cvss_score = cvss_data['cvss2'].get('base_score', 'N/A')

            detected = vuln.get('detected_at', 'Unknown')
            if detected and detected != 'Unknown':
                try:
                    detected = detected[:10]
                except Exception:
                    pass

            affected_count = vuln.get('affected_agent_count', 0)
            affected_agents = vuln.get('affected_agents', [])

            references = vuln.get('reference', '')
            ref_links = []
            if references:
                refs = [r.strip() for r in str(references).split(',')]
                ref_links = [r for r in refs if r.startswith('http')][:2]

            lines.append(f"### {i}. {cve_id}")
            agent_info = f" | **Affected Agents:** {affected_count}" if affected_count > 0 else ""
            lines.append(f"**Severity:** 🔴 {severity} | **CVSS:** {cvss_score} | **Detected:** {detected}{agent_info}")
            lines.append("")
            lines.append(f"**Description:**  ")
            lines.append(f"{description}")
            lines.append("")

            if affected_agents:
                lines.append(f"**Agents:** {', '.join(str(a) for a in affected_agents[:10])}")
                if len(affected_agents) > 10:
                    lines.append(f"  *...and {len(affected_agents) - 10} more*")
                lines.append("")

            if ref_links:
                lines.append(f"**References:**")
                for ref in ref_links:
                    lines.append(f"- {ref}")
                lines.append("")

            lines.append(f"**Immediate Action:**")
            lines.append(f"```bash")
            lines.append(f"# Update vulnerable packages")
            lines.append(f"sudo yum update -y  # RHEL/CentOS")
            lines.append(f"sudo apt-get update && sudo apt-get upgrade -y  # Ubuntu/Debian")
            lines.append(f"```")
            lines.append("")
            lines.append("---")
            lines.append("")

        if total > 10:
            lines.append(f"*...and {total - 10} more critical vulnerabilities*")
            lines.append("")

        lines.append("### 📊 View Complete Details in Wazuh Dashboard")
        lines.append("")
        lines.append("🔗 **Navigate to:** Wazuh → Vulnerabilities → Filter by 'Critical'")
        lines.append("")
        lines.append("### 📈 Action Summary")
        lines.append("")
        lines.append(f"- **Total Unique Critical CVEs:** {total}")
        lines.append(f"- **Timeline:** Remediate within 24-48 hours")
        lines.append(f"- **Priority:** Patch internet-facing systems first")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show vulnerability summary` | `show all vulnerabilities` | `correlate alerts and vulnerabilities` | `show active agents`")

        return "\n".join(lines)

    def _format_all_vulnerabilities(self, vulns: List[Dict]) -> str:
        """Format all vulnerabilities with severity breakdown AND CVE details."""
        if not vulns:
            return "✅ No vulnerabilities found"

        # Normalize severity to Title-case (Critical/High/Medium/Low/Unknown)
        def _norm_sev(v):
            s = str(v.get('severity', 'Unknown') or 'Unknown')
            return s.title() if s.lower() in ('critical', 'high', 'medium', 'low', 'unknown') else s

        by_severity = {}
        for v in vulns:
            sev = _norm_sev(v)
            by_severity[sev] = by_severity.get(sev, 0) + 1

        total = len(vulns)
        lines = [
            f"## 🔒 Vulnerabilities: {total} unique CVEs",
            "",
            "### Severity Breakdown",
            "",
            "| Severity | Count | Percentage |",
            "|----------|------:|-----------:|",
        ]

        for sev in ['Critical', 'High', 'Medium', 'Low', 'Unknown']:
            count = by_severity.get(sev, 0)
            if count > 0:
                icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}.get(sev, "⚪")
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"| {icon} {sev} | {count} | {pct:.1f}% |")

        # Composite risk score
        risk_weights = {"Critical": 25, "High": 10, "Medium": 3, "Low": 1}
        risk_score = sum(by_severity.get(s, 0) * w for s, w in risk_weights.items())
        lines.append("")
        lines.append(f"**Composite Risk Score:** {risk_score:,}")
        lines.append("")

        # CVE details — sort by severity priority then CVSS desc, show top 20
        sev_priority = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Unknown": 4}

        def _cvss(v):
            score = v.get('cvss_score')
            if score in (None, 'N/A', ''):
                cvss_data = v.get('cvss', {}) or {}
                if isinstance(cvss_data, dict):
                    if 'cvss3' in cvss_data:
                        score = cvss_data['cvss3'].get('base_score')
                    elif 'cvss2' in cvss_data:
                        score = cvss_data['cvss2'].get('base_score')
            try:
                return float(score)
            except (TypeError, ValueError):
                return 0.0

        sorted_vulns = sorted(
            vulns,
            key=lambda v: (sev_priority.get(_norm_sev(v), 5), -_cvss(v))
        )
        top_n = sorted_vulns[:20]

        lines.append(f"### Top {len(top_n)} CVEs (by severity, then CVSS)")
        lines.append("")
        lines.append("| # | CVE | Severity | CVSS | Package | Agent | Detected |")
        lines.append("|---|-----|----------|-----:|---------|-------|----------|")

        for i, v in enumerate(top_n, 1):
            cve_id = v.get('cve_id') or v.get('cve') or v.get('id') or 'N/A'
            sev = _norm_sev(v)
            icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}.get(sev, "⚪")
            cvss = _cvss(v) or 'N/A'
            pkg_obj = v.get('package', {}) or {}
            pkg = f"{pkg_obj.get('name', 'N/A')} {pkg_obj.get('version', '')}".strip() if isinstance(pkg_obj, dict) else str(pkg_obj)
            agent_obj = v.get('agent', {}) or {}
            agent_str = (
                f"{agent_obj.get('id', 'N/A')} ({agent_obj.get('name', 'N/A')})"
                if isinstance(agent_obj, dict) else str(agent_obj)
            )
            detected = str(v.get('detected_at', 'N/A') or 'N/A')[:10]
            lines.append(f"| {i} | {cve_id} | {icon} {sev} | {cvss} | {pkg[:40]} | {agent_str[:25]} | {detected} |")

        if total > len(top_n):
            lines.append("")
            lines.append(f"_Showing top {len(top_n)} of {total} CVEs. For specific CVE detail run `show vulnerability <CVE-ID>`._")

        return "\n".join(lines)

    # ================================================================
    # VULNERABILITY SUMMARY FORMATTER (NEW)
    # ================================================================

    def format_vulnerability_summary_response(self, raw_text: str) -> str:
        """
        Format vulnerability summary from get_wazuh_vulnerability_summary tool.
        Input is the raw text from the MCP content wrapper, typically:
          "Vulnerability Summary:\n{json}" or just "{json}"
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## 🔒 Vulnerability Summary\n\n*No data available*"

        # Extract JSON from the text
        data = None
        try:
            # Try parsing the whole thing as JSON
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            # Try extracting JSON after a label line
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## 🔒 Vulnerability Summary\n\n```\n{raw_text[:2000]}\n```"

        # Extract summary fields
        summary = data.get('summary', data)
        total = data.get('total', sum(summary.get(k, 0) for k in ['critical', 'high', 'medium', 'low']))
        time_range = data.get('time_range', 'N/A')

        critical = summary.get('critical', 0)
        high = summary.get('high', 0)
        medium = summary.get('medium', 0)
        low = summary.get('low', 0)

        risk_score = critical * 25 + high * 10 + medium * 3 + low * 1

        lines = [
            "## 🔒 Vulnerability Summary",
            "",
            f"**Total Vulnerabilities:** {total:,} | **Time Range:** {time_range}",
            "",
            "### Severity Breakdown",
            "",
            "| Severity | Count | Percentage | Risk Weight |",
            "|----------|------:|-----------:|------------:|",
        ]

        for sev_name, count, weight in [
            ("Critical", critical, 25), ("High", high, 10),
            ("Medium", medium, 3), ("Low", low, 1)
        ]:
            if total > 0:
                pct = (count / total * 100)
            else:
                pct = 0
            icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}[sev_name]
            lines.append(f"| {icon} {sev_name} | {count:,} | {pct:.1f}% | {count * weight:,} |")

        lines.append("")
        lines.append(f"**Composite Risk Score:** {risk_score:,}")
        lines.append("")

        # Remediation timeline
        lines.append("### Remediation Timeline")
        lines.append("")
        if critical > 0:
            lines.append(f"- **Critical ({critical:,})** — remediate within 24-48 hours")
        if high > 0:
            lines.append(f"- **High ({high:,})** — remediate within 7 days")
        if medium > 0:
            lines.append(f"- **Medium ({medium:,})** — remediate within 30 days")
        if low > 0:
            lines.append(f"- **Low ({low:,})** — remediate during next maintenance window")

        if critical > 0:
            lines.append("")
            lines.append("> Prioritize internet-facing and production systems. Use `show critical vulnerabilities` for detailed CVE list.")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show critical vulnerabilities` | `show vulnerabilities` | `correlate alerts and vulnerabilities` | `show active agents`")

        return "\n".join(lines)

    # ================================================================
    # HEALTH FORMATTER (NEW)
    # ================================================================

    def format_health_response(self, raw_text: str) -> str:
        """
        Format agent health response into structured readable output.
        Replaces raw JSON log collector dump with formatted summary.
        Adds stability assessment.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## 🏥 Agent Health\n\n*No data available*"

        # The server.py handler already creates markdown, but embeds logcollector
        # stats as a JSON code block. We replace that with a formatted summary.

        # Check if there's a JSON code block for log collector stats
        json_block_pattern = r'```json\s*\n(\{[\s\S]*?\})\s*\n```'
        json_match = re.search(json_block_pattern, raw_text)

        if json_match:
            try:
                lc_data = json.loads(json_match.group(1))
                lc_formatted = self._format_logcollector_summary(lc_data)
                # Replace the JSON code block with formatted summary
                raw_text = raw_text[:json_match.start()] + lc_formatted + raw_text[json_match.end():]
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[HEALTH_FORMATTER] Could not parse logcollector JSON: {e}")

        # Add stability assessment at the end
        stability = self._assess_stability(raw_text)
        if stability:
            raw_text += "\n" + stability

        return raw_text

    def _format_logcollector_summary(self, lc_data: Dict) -> str:
        """Format logcollector stats as a readable summary table."""
        items = lc_data.get('affected_items', [lc_data])
        if not items:
            return "\n### 📋 Log Collector Summary\n- No data available\n"

        lc = items[0] if isinstance(items, list) else items
        global_stats = lc.get('global', {})
        global_files = global_stats.get('files', [])

        total_events = sum(f.get('events', 0) for f in global_files)
        total_bytes = sum(f.get('bytes', 0) for f in global_files)
        total_drops = sum(
            t.get('drops', 0)
            for f in global_files
            for t in f.get('targets', [])
        )

        def fmt_bytes(b):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if b < 1024:
                    return f"{b:.1f} {unit}"
                b /= 1024
            return f"{b:.1f} TB"

        lines = [
            "\n### 📋 Log Collector Summary",
            "",
            f"- **Period:** {global_stats.get('start', 'N/A')} → {global_stats.get('end', 'N/A')}",
            f"- **Total Events:** {total_events:,}",
            f"- **Total Data Collected:** {fmt_bytes(total_bytes)}",
            f"- **Dropped Events:** {total_drops}",
        ]

        # Show top log sources
        active_sources = sorted(
            [f for f in global_files if f.get('events', 0) > 0],
            key=lambda x: x.get('events', 0), reverse=True
        )
        if active_sources:
            lines.append("")
            lines.append("**Top Log Sources:**")
            lines.append("")
            lines.append("| Source | Events | Data |")
            lines.append("|--------|-------:|-----:|")
            for src in active_sources[:5]:
                loc = src.get('location', 'unknown')
                evts = src.get('events', 0)
                bts = src.get('bytes', 0)
                lines.append(f"| `{loc}` | {evts:,} | {fmt_bytes(bts)} |")

        return "\n".join(lines)

    def _assess_stability(self, health_text: str) -> str:
        """Generate stability assessment from health report text."""
        lines = [
            "",
            "### 📊 Stability Assessment",
            "",
        ]

        # Check status
        status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', health_text)
        status = status_match.group(1).lower() if status_match else "unknown"

        if status == "active":
            lines.append("- **Connection Status:** 🟢 Stable — agent is connected and reporting")
        elif status == "disconnected":
            lines.append("- **Connection Status:** 🔴 Lost — agent is not responding")
        else:
            lines.append(f"- **Connection Status:** 🟡 {status.title()}")

        # Check version
        version_match = re.search(r'\*\*Version:\*\*\s*([\w\s.v]+)', health_text)
        if version_match:
            version = version_match.group(1).strip()
            lines.append(f"- **Agent Version:** {version}")

        # Overall health
        if status == "active":
            lines.append("- **Overall Health:** HEALTHY - no issues detected")
        elif status == "disconnected":
            lines.append("- **Overall Health:** CRITICAL - agent offline, investigate immediately")
        else:
            lines.append("- **Overall Health:** WARNING - status requires attention")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `show ports for agent <ID>` | `show processes for agent <ID>` | `show vulnerabilities` | `show all agents`")

        return "\n".join(lines)

    # ================================================================
    # PORT FORMATTER (NEW)
    # ================================================================

    def format_ports_response(self, raw_text: str) -> str:
        """
        Format port data with suspicious port detection and security summary.
        Input is the markdown text from server.py handler.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## 🔌 Open Ports\n\n*No data available*"

        # Parse port numbers from the table
        port_findings = []
        port_numbers = []
        public_bind_count = 0
        table_rows = re.findall(r'\|\s*(\w+)\s*\|\s*([\d.:]+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|', raw_text)

        for row in table_rows:
            protocol, ip, port_str, process, pid = row
            try:
                port = int(port_str)
                port_numbers.append(port)

                # Track public-binding (listens on all interfaces — biggest exposure signal)
                if ip in ("0.0.0.0", "::", "::0"):
                    public_bind_count += 1

                # Check suspicious ports
                if port in SUSPICIOUS_PORTS:
                    port_findings.append(f"- **Port {port}** ({process}): ⚠️ {SUSPICIOUS_PORTS[port]}")
                elif port in DATABASE_PORTS:
                    port_findings.append(f"- **Port {port}** ({process}): 🔶 {DATABASE_PORTS[port]} port exposed — ensure firewall restricts access")
                elif port in INSECURE_PORTS:
                    port_findings.append(f"- **Port {port}** ({process}): ⚠️ {INSECURE_PORTS[port]}")
                elif port in NON_STANDARD_HTTP:
                    port_findings.append(f"- **Port {port}** ({process}): 🔶 {NON_STANDARD_HTTP[port]}")
                elif port >= SUSPICIOUS_HIGH_PORTS_LISTENING:
                    port_findings.append(f"- **Port {port}** ({process}): 🔶 Dynamic port range — verify if intentional")
            except ValueError:
                continue

        # Port security summary metrics
        well_known = sum(1 for p in port_numbers if p < 1024)
        registered = sum(1 for p in port_numbers if 1024 <= p < 49152)
        dynamic = sum(1 for p in port_numbers if p >= 49152)
        suspicious_count = len(port_findings)

        # Headline summary block — prepended so it becomes the response's KPI grid.
        # Labels are intentionally simple (no hyphens / no parens) so the SPA's
        # KPI parser detects them as a clean KPI_GRID group.
        summary_lines = [
            "## Port Summary",
            "",
            f"**Listening Ports:** {len(port_numbers)}",
            f"**Public Bind:** {public_bind_count}",
            f"**Privileged Open:** {well_known}",
            f"**Findings:** {suspicious_count}",
            "",
            "",
        ]
        result = "\n".join(summary_lines) + raw_text.rstrip()

        if port_findings:
            result += "\n\n### ⚠️ Suspicious Port Analysis\n\n"
            result += "\n".join(port_findings)

        # Detailed range breakdown (kept as prose — labels have hyphens/parens
        # so it stays as a list, not a duplicate KPI grid).
        result += "\n\n### 📊 Port Security Summary\n\n"
        result += f"- **Well-known ports (0-1023):** {well_known} open\n"
        result += f"- **Registered ports (1024-49151):** {registered} open\n"
        result += f"- **Dynamic ports (49152-65535):** {dynamic} open\n"
        result += f"- **Findings requiring review:** {suspicious_count}\n"

        return result

    # ================================================================
    # PROCESS FORMATTER (NEW)
    # ================================================================

    def format_processes_response(self, raw_text: str) -> str:
        """
        Format process data with suspicious process detection and security summary.
        Input is the markdown text from server.py handler.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## 🔧 Processes\n\n*No data available*"

        # Parse process info from table rows
        process_findings = []
        root_count = 0
        total_parsed = 0

        # Match table rows: | PID | Name | User | State | Memory | Command |
        table_rows = re.findall(
            r'\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|',
            raw_text
        )

        for row in table_rows:
            pid, name, user, state, memory, cmd = [x.strip() for x in row]
            total_parsed += 1
            name_lower = name.lower()
            user_lower = user.lower()

            # Check for known malicious process names
            for susp in SUSPICIOUS_PROCESSES:
                if susp in name_lower or susp in cmd.lower():
                    process_findings.append(
                        f"- **{name}** (PID {pid}, user: {user}): ⚠️ Matches known suspicious pattern `{susp}`"
                    )
                    break

            # Check for root processes
            if user_lower == 'root':
                root_count += 1
                is_known_system = (
                    name_lower in SYSTEM_ROOT_PROCESSES
                    or name_lower.startswith(KERNEL_THREAD_PREFIXES)
                )
                if not is_known_system:
                    process_findings.append(
                        f"- **{name}** (PID {pid}): 🔶 Running as root — verify if necessary"
                    )

            # Check high memory
            mem_match = re.search(r'([\d,]+)', memory)
            if mem_match:
                try:
                    mem_kb = int(mem_match.group(1).replace(',', ''))
                    if mem_kb > 500000:  # >500MB
                        process_findings.append(
                            f"- **{name}** (PID {pid}): 🔶 High memory usage ({memory}) — monitor for leaks"
                        )
                except ValueError:
                    pass

        # Headline summary block — prepended so the SPA's KPI parser detects
        # it as the headline KPI grid. Labels are simple (no hyphens / parens).
        summary_lines = [
            "## Process Summary",
            "",
            f"**Total Processes:** {total_parsed}",
            f"**Root Processes:** {root_count}",
            f"**Findings:** {len(process_findings)}",
            "",
            "",
        ]
        result = "\n".join(summary_lines) + raw_text.rstrip()

        if process_findings:
            result += "\n\n### ⚠️ Suspicious Process Detection\n\n"
            result += "\n".join(process_findings[:15])  # Limit to 15 findings
            if len(process_findings) > 15:
                result += f"\n- *...and {len(process_findings) - 15} more findings*"
        else:
            result += "\n\n### ✅ Process Security Check\n\n"
            result += "- No known suspicious processes detected"

        # Detailed prose breakdown (kept in addition to the headline KPIs).
        result += "\n\n### Process Security Detail\n\n"
        result += f"- **Processes analyzed:** {total_parsed}\n"
        result += f"- **Running as root:** {root_count}\n"
        result += f"- **Findings requiring review:** {len(process_findings)}\n"

        # Navigation hints
        result += "\n---\n"
        result += "**Related queries:** `show ports for agent <ID>` | `check agent <ID> health` | `show vulnerabilities` | `show fim events for agent <ID>`"

        return result

    # ================================================================
    # COMPLIANCE FORMATTER
    # ================================================================

    def format_compliance_response(self, raw_text: str) -> str:
        """
        Format compliance check response into structured readable output.
        Input is raw text from MCP content wrapper, typically:
          "Compliance Check:\n{json}" or just "{json}"
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Compliance Check\n\n*No data available*"

        # Extract JSON from the text
        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            # Try extracting JSON after a label line
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Compliance Check\n\n```\n{raw_text[:2000]}\n```"

        framework = data.get('framework', 'Unknown')
        agent_id = data.get('agent_id', 'all')
        status = data.get('status', 'unknown')
        checks_performed = data.get('checks_performed', 0)
        checks_passed = data.get('checks_passed', 0)
        checks_failed = data.get('checks_failed', 0)
        note = data.get('note', '')

        # Status icon
        if status.lower() == 'compliant':
            status_icon = "🟢"
        elif status.lower() in ('non-compliant', 'non_compliant', 'failed'):
            status_icon = "🔴"
        else:
            status_icon = "🟡"

        # Pass rate
        pass_rate = (checks_passed / checks_performed * 100) if checks_performed > 0 else 0

        scope = f"Agent {agent_id}" if agent_id and agent_id != "all" else "All Agents"

        lines = [
            f"## Compliance Check — {framework}",
            "",
            f"**Scope:** {scope}",
            f"**Status:** {status_icon} {status.upper()}",
            f"**Pass Rate:** {pass_rate:.0f}%",
            "",
            "### Results",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Checks Performed | {checks_performed} |",
            f"| Checks Passed | {checks_passed} |",
            f"| Checks Failed | {checks_failed} |",
        ]

        if note:
            lines.append("")
            lines.append(f"**Note:** {note}")

        # Recommendations based on results
        lines.append("")
        lines.append("### Recommendations")
        lines.append("")

        if checks_failed > 0:
            lines.append(f"- **{checks_failed} failed checks** require remediation")
            lines.append(f"- Review {framework} requirements and align configurations")
            lines.append(f"- Re-run compliance check after remediation")
        else:
            lines.append(f"- All checks passed for {framework}")
            lines.append(f"- Schedule periodic compliance reviews")

        if framework.upper() in ("PCI-DSS", "PCI"):
            lines.append("- Ensure network segmentation and access controls")
            lines.append("- Verify encryption of cardholder data in transit and at rest")
        elif framework.upper() == "HIPAA":
            lines.append("- Review PHI access logs and audit trails")
            lines.append("- Verify data encryption and access controls")
        elif framework.upper() == "GDPR":
            lines.append("- Verify data subject rights mechanisms")
            lines.append("- Review data processing agreements")
        elif framework.upper() == "NIST":
            lines.append("- Review NIST CSF control mappings")
            lines.append("- Verify security baseline configurations")

        return "\n".join(lines)

    # ================================================================
    # ALERT FORMATTERS
    # ================================================================

    def format_alert_response(self, correlated_data, summary, **kwargs):
        """
        Show recent alerts with full details.
        Handles BOTH data paths:
          - Path A: Flat dict from correlation engine (keys: timestamp, rule_id, level, agent_name, rule_description)
          - Path B: Nested dict from raw indexer _source (keys: @timestamp, rule.id, rule.level, agent.name, rule.description)
        """
        alerts = self._extract_data(correlated_data)

        if not alerts:
            return "## Recent Alerts — Last 24 Hours\n\nNo alerts in the specified time range"

        total = len(alerts)
        recent = alerts[:20]

        # --- Dual-path field extraction helpers ---
        def _get_timestamp(a):
            ts = a.get('timestamp') or a.get('@timestamp', 'N/A')
            return str(ts)[:19].replace('T', ' ') if ts and ts != 'N/A' else 'N/A'

        def _get_rule_id(a):
            rid = a.get('rule_id')
            if not rid or rid == 'N/A':
                rule = a.get('rule', {}) if isinstance(a.get('rule'), dict) else {}
                rid = rule.get('id', 'N/A')
            return str(rid) if rid else 'N/A'

        def _get_level(a):
            lvl = a.get('level')
            if lvl is None:
                rule = a.get('rule', {}) if isinstance(a.get('rule'), dict) else {}
                lvl = rule.get('level', 0)
            try:
                return int(lvl)
            except (ValueError, TypeError):
                return 0

        def _get_agent_name(a):
            name = a.get('agent_name')
            if not name or name in ('N/A', 'Unknown'):
                agent = a.get('agent', {}) if isinstance(a.get('agent'), dict) else {}
                name = agent.get('name', 'N/A')
            return str(name) if name else 'N/A'

        def _get_description(a):
            desc = a.get('rule_description')
            if not desc or desc == 'N/A':
                rule = a.get('rule', {}) if isinstance(a.get('rule'), dict) else {}
                desc = rule.get('description', 'N/A')
            return str(desc)[:80]

        # --- Aggregate severity counts and top rules ---
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        rule_counts = {}

        for alert in alerts:
            level = _get_level(alert)
            if level >= 12:
                severity_counts['critical'] += 1
            elif level >= 7:
                severity_counts['high'] += 1
            elif level >= 4:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1

            rid = _get_rule_id(alert)
            rule_counts[rid] = rule_counts.get(rid, 0) + 1

        # --- Build output: summary header + severity breakdown + detail table ---
        lines = [
            f"## Recent Alerts - Last 24 Hours",
            "",
            f"**Total Alerts:** {total:,}",
            "",
            "### Severity Breakdown",
            "",
            "| Severity | Count | Percentage |",
            "|----------|------:|-----------:|",
        ]

        for sev_name, count in [("Critical (12+)", severity_counts['critical']),
                                 ("High (7-11)", severity_counts['high']),
                                 ("Medium (4-6)", severity_counts['medium']),
                                 ("Low (0-3)", severity_counts['low'])]:
            if count > 0:
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"| {sev_name} | {count:,} | {pct:.1f}% |")

        lines.append("")
        lines.append(f"### Latest {len(recent)} Alerts")
        lines.append("")
        lines.append("| Time | Rule ID | Level | Agent | Description |")
        lines.append("|------|---------|------:|-------|-------------|")

        for alert in recent:
            ts = _get_timestamp(alert)
            rid = _get_rule_id(alert)
            level = _get_level(alert)
            agent = _get_agent_name(alert)
            desc = _get_description(alert)

            lines.append(f"| {ts} | {rid} | {level} | {agent} | {desc} |")

        # --- Analysis section with top rules ---
        lines.append("")
        lines.append("### Top Firing Rules")
        lines.append("")

        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for rule_id, count in top_rules:
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * max(1, int(pct / 5))
            lines.append(f"- Rule {rule_id}: {count:,} times ({pct:.1f}%) {bar}")

        # Rule concentration warning
        if top_rules:
            top_count = top_rules[0][1]
            top_pct = (top_count / total * 100) if total > 0 else 0
            if top_pct > 80:
                lines.append("")
                lines.append(f"> **Alert Fatigue Warning:** Rule {top_rules[0][0]} generates {top_pct:.0f}% of all alerts. Consider tuning this rule.")

        if severity_counts['critical'] > 0:
            lines.append("")
            lines.append(f"> **{severity_counts['critical']} critical alerts** require immediate investigation")

        # Navigation hints
        lines.append("")
        lines.append("---")
        lines.append("**Related queries:** `analyze alert patterns` | `show rule trigger analysis` | `show mitre coverage` | `show critical vulnerabilities`")

        return "\n".join(lines)

    # ================================================================
    # SEARCH SECURITY EVENTS FORMATTER
    # ================================================================

    def format_search_events_response(
        self,
        correlated_data,
        summary,
        search_query: str = "",
        time_range: str = "24h",
        **kwargs
    ) -> str:
        """Dedicated formatter for search_security_events tool."""
        events = self._extract_data(correlated_data)

        logger.info(f"[SEARCH_FORMATTER] search_query='{search_query}', time_range='{time_range}', events={len(events)}")

        if not events:
            return (
                f"## 🔍 Search Results: `{search_query}`\n\n"
                f"**Time Range:** {time_range}\n\n"
                f"✅ No matching security events found for this query in the specified time range."
            )

        total = len(events)
        recent = events[:20]

        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        rule_counts = {}
        integrations = set()

        for event in events:
            rule = event.get('rule', {}) if isinstance(event.get('rule'), dict) else {}
            level = rule.get('level', 0)
            try:
                level = int(level)
            except Exception:
                level = 0

            if level >= 12:
                severity_counts['critical'] += 1
            elif level >= 7:
                severity_counts['high'] += 1
            elif level >= 4:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1

            rule_id = str(rule.get('id', 'unknown'))
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

            data_field = event.get('data', {}) if isinstance(event.get('data'), dict) else {}
            if 'integration' in data_field:
                integrations.add(data_field['integration'])
            loc = event.get('location', '')
            if loc:
                integrations.add(loc)

        lines = [
            f"## 🔍 Search Results: `{search_query}`",
            "",
            f"**Time Range:** {time_range} | **Total Matching Events:** {total}",
            "",
        ]

        if integrations:
            lines.append(f"**Integrations/Sources:** {', '.join(sorted(integrations))}")
            lines.append("")

        lines += [
            "### Severity Breakdown",
            "",
            f"- 🔴 Critical (L12+): {severity_counts['critical']}",
            f"- 🟠 High (L7-11): {severity_counts['high']}",
            f"- 🟡 Medium (L4-6): {severity_counts['medium']}",
            f"- 🟢 Low (L1-3): {severity_counts['low']}",
            "",
            f"### Latest {len(recent)} Events",
            "",
            "| Time | Rule | Level | Agent | Description |",
            "|------|------|------:|-------|-------------|"
        ]

        for event in recent:
            # Dual-path: try flat keys (correlation), then nested keys (raw indexer)
            ts = event.get('timestamp') or event.get('@timestamp', 'N/A')
            timestamp = str(ts)[:19].replace('T', ' ') if ts and ts != 'N/A' else 'N/A'

            rid = event.get('rule_id')
            if not rid or rid == 'N/A':
                rule = event.get('rule', {}) if isinstance(event.get('rule'), dict) else {}
                rid = rule.get('id', 'N/A')
            rule_id = str(rid) if rid else 'N/A'

            lvl = event.get('level')
            if lvl is None:
                rule = event.get('rule', {}) if isinstance(event.get('rule'), dict) else {}
                lvl = rule.get('level', 0)
            try:
                level_num = int(lvl)
            except (ValueError, TypeError):
                level_num = 0

            aname = event.get('agent_name')
            if not aname or aname in ('N/A', 'Unknown'):
                agent = event.get('agent', {}) if isinstance(event.get('agent'), dict) else {}
                aname = agent.get('name', 'N/A')
            agent_name = str(aname) if aname else 'N/A'

            d = event.get('rule_description')
            if not d or d == 'N/A':
                rule = event.get('rule', {}) if isinstance(event.get('rule'), dict) else {}
                d = rule.get('description', 'N/A')
            desc = str(d)[:60]

            icon = "🔴" if level_num >= 12 else "🟠" if level_num >= 7 else "🟡" if level_num >= 4 else "🟢"

            lines.append(f"| {timestamp} | {rule_id} | {icon} {level_num} | {agent_name} | {desc} |")

        sample_data = []
        for event in recent[:5]:
            data_field = event.get('data', {}) if isinstance(event.get('data'), dict) else {}
            if data_field and search_query.lower() in str(data_field).lower():
                sample_data.append(data_field)

        if sample_data:
            lines.append("")
            lines.append("### 🔎 Event Data Details")
            lines.append("")
            lines.append(f"```json\n{json.dumps(sample_data[:3], indent=2, default=str)[:1500]}\n```")

        lines.append("")
        lines.append("### 📊 Analysis")
        lines.append("")

        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_rules:
            lines.append("**Most Frequent Rules:**")
            for rule_id, count in top_rules:
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"- Rule {rule_id}: {count} times ({pct:.1f}%)")

        if severity_counts['critical'] > 0:
            lines.append("")
            lines.append(f"⚠️ **{severity_counts['critical']} critical events** require immediate investigation")

        return "\n".join(lines)

    # ================================================================
    # ALERT SUMMARY FORMATTER (aggregation data)
    # ================================================================

    def format_alert_summary_response(self, raw_text: str) -> str:
        """
        Format alert summary (Elasticsearch aggregation by rule.level).
        Input: JSON with {"aggregations":{"group_by":{"buckets":[{"key":"7","doc_count":150},...]}}}
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Alert Summary\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Alert Summary\n\n```\n{raw_text[:2000]}\n```"

        buckets = data.get("aggregations", {}).get("group_by", {}).get("buckets", [])
        if not buckets:
            return "## Alert Summary\n\n*No alert data found for the specified time range.*"

        total_alerts = sum(b.get("doc_count", 0) for b in buckets)

        level_map = {
            "critical": {"min": 12, "max": 99, "count": 0, "icon": "🔴"},
            "high": {"min": 7, "max": 11, "count": 0, "icon": "🟠"},
            "medium": {"min": 4, "max": 6, "count": 0, "icon": "🟡"},
            "low": {"min": 0, "max": 3, "count": 0, "icon": "🟢"},
        }

        for bucket in buckets:
            try:
                level = int(bucket.get("key", 0))
            except (ValueError, TypeError):
                level = 0
            count = bucket.get("doc_count", 0)
            if level >= 12:
                level_map["critical"]["count"] += count
            elif level >= 7:
                level_map["high"]["count"] += count
            elif level >= 4:
                level_map["medium"]["count"] += count
            else:
                level_map["low"]["count"] += count

        lines = [
            "## Alert Summary -- Last 24 Hours",
            "",
            f"**Total Alerts:** {total_alerts:,}",
            "",
            "### Severity Distribution",
            "",
            "| Severity | Level Range | Count | Percentage | Bar |",
            "|----------|:----------:|------:|-----------:|-----|",
        ]

        for sev_name, info in level_map.items():
            pct = (info["count"] / total_alerts * 100) if total_alerts > 0 else 0
            bar_len = int(pct / 5)
            bar = "█" * bar_len if bar_len > 0 else ""
            lines.append(
                f"| {info['icon']} {sev_name.capitalize()} | L{info['min']}-{info['max'] if info['max'] < 99 else '15+'} "
                f"| {info['count']:,} | {pct:.1f}% | {bar} |"
            )

        lines.append("")
        lines.append("### Level Breakdown (per level)")
        lines.append("")
        lines.append("| Level | Count | Severity |")
        lines.append("|------:|------:|----------|")

        sorted_buckets = sorted(buckets, key=lambda b: int(b.get("key", 0)), reverse=True)
        for bucket in sorted_buckets[:15]:
            try:
                level = int(bucket.get("key", 0))
            except (ValueError, TypeError):
                level = 0
            count = bucket.get("doc_count", 0)
            sev = "Critical" if level >= 12 else "High" if level >= 7 else "Medium" if level >= 4 else "Low"
            icon = "🔴" if level >= 12 else "🟠" if level >= 7 else "🟡" if level >= 4 else "🟢"
            lines.append(f"| {level} | {count:,} | {icon} {sev} |")

        lines.append("")
        lines.append("### Key Observations")
        lines.append("")
        if level_map["critical"]["count"] > 0:
            lines.append(f"- **{level_map['critical']['count']:,} critical alerts** (level 12+) require immediate investigation")
        if level_map["high"]["count"] > 0:
            lines.append(f"- {level_map['high']['count']:,} high-severity alerts (level 7-11) should be reviewed")
        top_level = max(buckets, key=lambda b: b.get("doc_count", 0)) if buckets else {}
        if top_level:
            try:
                top_key = int(top_level.get("key", 0))
            except (ValueError, TypeError):
                top_key = 0
            top_pct = (top_level.get("doc_count", 0) / total_alerts * 100) if total_alerts > 0 else 0
            lines.append(f"- Most frequent alert level: **{top_key}** ({top_pct:.1f}% of total)")

        return "\n".join(lines)

    # ================================================================
    # ALERT PATTERNS FORMATTER
    # ================================================================

    def format_alert_patterns_response(self, raw_text: str) -> str:
        """
        Format alert pattern analysis results.
        Input: JSON with {"time_range":"24h","min_frequency":5,"patterns":[{"rule_id":"5710","count":50,"frequency":"high"},...]}
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Alert Pattern Analysis\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Alert Pattern Analysis\n\n```\n{raw_text[:2000]}\n```"

        time_range = data.get("time_range", "24h")
        min_freq = data.get("min_frequency", 5)
        patterns = data.get("patterns", [])

        lines = [
            f"## Alert Pattern Analysis -- Last {time_range}",
            "",
            f"**Minimum Frequency Threshold:** {min_freq}",
            f"**Patterns Detected:** {len(patterns)}",
            "",
        ]

        if not patterns:
            lines.append("*No recurring alert patterns detected above the frequency threshold.*")
            return "\n".join(lines)

        high_patterns = [p for p in patterns if p.get("frequency") == "high"]
        medium_patterns = [p for p in patterns if p.get("frequency") == "medium"]

        lines += [
            "### Detected Patterns",
            "",
            "| Rule ID | Count | Frequency | Status |",
            "|---------|------:|-----------|--------|",
        ]

        sorted_patterns = sorted(patterns, key=lambda p: p.get("count", 0), reverse=True)
        for p in sorted_patterns:
            freq = p.get("frequency", "unknown")
            freq_display = freq.capitalize()
            if freq == "high":
                status = "🔴 Investigate"
            elif freq == "medium":
                status = "🟡 Monitor"
            else:
                status = "🟢 Normal"
            lines.append(f"| {p.get('rule_id', 'N/A')} | {p.get('count', 0):,} | {freq_display} | {status} |")

        lines.append("")
        lines.append("### Key Findings")
        lines.append("")
        lines.append(f"- **{len(patterns)}** recurring alert patterns detected")
        if high_patterns:
            lines.append(f"- **{len(high_patterns)} high-frequency patterns** require investigation")
        if medium_patterns:
            lines.append(f"- {len(medium_patterns)} medium-frequency patterns should be monitored")
        total_events = sum(p.get("count", 0) for p in patterns)
        lines.append(f"- Total events from detected patterns: **{total_events:,}**")

        return "\n".join(lines)

    # ================================================================
    # THREAT ANALYSIS FORMATTER
    # ================================================================

    def format_threat_analysis_response(self, raw_text: str) -> str:
        """
        Format threat analysis results for an indicator.
        Input: JSON with indicator, total_hits, risk_level, severity_breakdown, top_rules, etc.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Threat Analysis\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Threat Analysis\n\n```\n{raw_text[:2000]}\n```"

        indicator = data.get("indicator", "Unknown")
        ind_type = data.get("indicator_type", "unknown").upper()
        total_hits = data.get("total_hits", 0)
        risk_level = data.get("risk_level", "unknown")
        severity = data.get("severity_breakdown", {})
        top_rules = data.get("top_rules", {})
        affected_agents = data.get("affected_agents", 0)
        first_seen = data.get("first_seen")
        last_seen = data.get("last_seen")
        error = data.get("error")
        note = data.get("note")

        risk_icons = {"critical": "🔴 CRITICAL", "high": "🟠 HIGH", "medium": "🟡 MEDIUM",
                      "low": "🟢 LOW", "none": "⚪ NONE", "unknown": "⚪ UNKNOWN"}
        risk_display = risk_icons.get(risk_level, f"⚪ {risk_level.upper()}")

        lines = [
            f"## Threat Analysis -- {indicator} ({ind_type})",
            "",
            f"**Risk Level:** {risk_display}",
            f"**Total Alert Hits:** {total_hits:,} (last 30 days)",
            f"**Affected Agents:** {affected_agents}",
            "",
        ]

        if error:
            lines.append(f"**Error:** {error}")
            lines.append("")
        if note:
            lines.append(f"**Note:** {note}")
            lines.append("")

        if severity:
            lines += [
                "### Severity Breakdown",
                "",
                f"🔴 Critical: {severity.get('critical', 0)} | "
                f"🟠 High: {severity.get('high', 0)} | "
                f"🟡 Medium: {severity.get('medium', 0)} | "
                f"🟢 Low: {severity.get('low', 0)}",
                "",
            ]

        if top_rules:
            lines += [
                "### Associated Rules",
                "",
                "| Rule ID | Count | Description |",
                "|---------|------:|-------------|",
            ]
            for rid, info in top_rules.items():
                if isinstance(info, dict):
                    count = info.get("count", 0)
                    desc = str(info.get("description") or "N/A")[:60]
                else:
                    count = info
                    desc = "N/A"
                lines.append(f"| {rid} | {count:,} | {desc} |")
            lines.append("")

        if first_seen or last_seen:
            lines += [
                "### Timeline",
                "",
            ]
            if first_seen:
                lines.append(f"**First Seen:** {str(first_seen)[:19].replace('T', ' ')}")
            if last_seen:
                lines.append(f"**Last Seen:** {str(last_seen)[:19].replace('T', ' ')}")
            lines.append("")

        if total_hits == 0:
            lines.append("*No alert activity found for this indicator in the last 30 days.*")
        elif risk_level in ("critical", "high"):
            lines.append("**Recommendation:** Immediate investigation recommended. Block or isolate this indicator and review affected agents.")
        elif risk_level == "medium":
            lines.append("**Recommendation:** Monitor this indicator closely. Cross-reference with external threat intelligence.")
        else:
            lines.append("**Recommendation:** Continue monitoring. Low activity detected.")

        return "\n".join(lines)

    # ================================================================
    # IOC REPUTATION FORMATTER
    # ================================================================

    def format_ioc_reputation_response(self, raw_text: str) -> str:
        """
        Format IoC reputation check results.
        Input: JSON with indicator, total_hits, risk_score, verdict, severity_breakdown, etc.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## IoC Reputation Check\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## IoC Reputation Check\n\n```\n{raw_text[:2000]}\n```"

        indicator = data.get("indicator", "Unknown")
        ind_type = data.get("indicator_type", "unknown").upper()
        verdict = data.get("verdict", "unknown")
        error = data.get("error")

        verdict_icons = {"malicious": "🔴 MALICIOUS", "suspicious": "🟡 SUSPICIOUS",
                         "low_risk": "🟢 LOW RISK", "clean": "⚪ CLEAN", "unknown": "⚪ UNKNOWN"}
        verdict_display = verdict_icons.get(verdict, f"⚪ {verdict.upper()}")

        lines = [
            f"## Threat Intelligence — {indicator} ({ind_type})",
            "",
            f"**Verdict:** {verdict_display}",
            "",
        ]

        if error:
            lines.append(f"**Error:** {error}")
            lines.append("")

        # AbuseIPDB Intelligence section
        abuseipdb = data.get("abuseipdb", {})
        if abuseipdb and not abuseipdb.get("error"):
            abuse_score = abuseipdb.get("abuseConfidenceScore", 0)
            total_reports = abuseipdb.get("totalReports", 0)
            isp = abuseipdb.get("isp", "N/A")
            country = abuseipdb.get("countryCode", "N/A")
            usage_type = abuseipdb.get("usageType", "N/A")
            domain = abuseipdb.get("domain", "N/A")
            is_whitelisted = abuseipdb.get("isWhitelisted", False)
            last_reported = abuseipdb.get("lastReportedAt", "N/A")
            is_tor = abuseipdb.get("isTor", False)
            hostnames = abuseipdb.get("hostnames", [])

            abuse_icon = "🔴" if abuse_score >= 75 else "🟡" if abuse_score >= 25 else "🟢"

            lines += [
                "### AbuseIPDB Intelligence",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Abuse Confidence | {abuse_icon} {abuse_score}% |",
                f"| Total Reports | {total_reports} |",
                f"| ISP | {isp} |",
                f"| Country | {country} |",
                f"| Usage Type | {usage_type} |",
                f"| Domain | {domain} |",
                f"| Whitelisted | {'Yes' if is_whitelisted else 'No'} |",
            ]
            if is_tor:
                lines.append("| Tor Exit Node | Yes |")
            if hostnames:
                lines.append(f"| Hostnames | {', '.join(hostnames[:3])} |")
            lines.append(f"| Last Reported | {str(last_reported)[:19].replace('T',' ') if last_reported != 'N/A' else 'N/A'} |")
            lines.append("")

            # Show recent abuse reports if available
            reports = abuseipdb.get("reports", [])
            if reports:
                lines += ["### Recent Abuse Reports", "",
                           "| Date | Categories | Comment |",
                           "|------|-----------|---------|"]
                for r in reports[:5]:
                    date = str(r.get("reportedAt", "N/A"))[:10]
                    cats = ", ".join(str(c) for c in r.get("categories", []))
                    comment = str(r.get("comment", ""))[:80]
                    lines.append(f"| {date} | {cats} | {comment} |")
                lines.append("")
        elif abuseipdb and abuseipdb.get("error"):
            lines += [
                "### AbuseIPDB Intelligence",
                "",
                f"*Lookup failed: {abuseipdb.get('error', 'unknown error')}*",
                "",
            ]

        return "\n".join(lines)

    # ================================================================
    # SECURITY REPORT FORMATTER
    # ================================================================

    def format_security_report_response(self, raw_text: str) -> str:
        """
        Format comprehensive security report.
        Input: JSON with report_type, generated_at, agents, alerts, vulnerabilities, recommendations.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Security Report\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Security Report\n\n```\n{raw_text[:2000]}\n```"

        report_type = data.get("report_type", "daily").capitalize()
        generated_at = data.get("generated_at", "N/A")
        if generated_at and generated_at != "N/A":
            generated_at = str(generated_at)[:19].replace("T", " ")

        agents = data.get("agents", {})
        alerts = data.get("alerts", {})
        vulns = data.get("vulnerabilities", {})
        recs = data.get("recommendations", [])

        lines = [
            f"## Security Report -- {report_type}",
            "",
            f"**Generated:** {generated_at}",
            "",
            "---",
            "",
            "### Agent Overview",
            "",
            "| Metric | Count |",
            "|--------|------:|",
            f"| Total Agents | {agents.get('total', 0)} |",
            f"| Active | {agents.get('active', 0)} |",
            f"| Disconnected | {agents.get('disconnected', 0)} |",
            "",
        ]

        # Agent health indicator
        total_ag = agents.get("total", 0)
        active_ag = agents.get("active", 0)
        disc_ag = agents.get("disconnected", 0)
        if total_ag > 0:
            health_pct = (active_ag / total_ag * 100)
            health_icon = "🟢" if health_pct >= 90 else "🟡" if health_pct >= 70 else "🔴"
            lines.append(f"**Agent Health:** {health_icon} {health_pct:.0f}% online")
            if disc_ag > 0:
                lines.append(f"**Warning:** {disc_ag} agent(s) disconnected")
            lines.append("")

        # Alerts section
        alert_total = alerts.get("total", 0)
        by_level = alerts.get("by_level", {})
        lines += [
            "### Alert Summary (Last 24h)",
            "",
            f"**Total Alerts:** {alert_total:,}",
            "",
        ]

        if by_level:
            crit_count = sum(v for k, v in by_level.items() if int(k) >= 12)
            high_count = sum(v for k, v in by_level.items() if 7 <= int(k) < 12)
            med_count = sum(v for k, v in by_level.items() if 4 <= int(k) < 7)
            low_count = sum(v for k, v in by_level.items() if int(k) < 4)
            lines += [
                f"🔴 Critical: {crit_count} | 🟠 High: {high_count} | 🟡 Medium: {med_count} | 🟢 Low: {low_count}",
                "",
            ]
        elif alert_total == 0:
            lines.append("*No alert data available for this period.*")
            lines.append("")

        # Vulnerabilities section
        vuln_total = vulns.get("total", 0)
        vuln_sev = vulns.get("by_severity", {})
        lines += [
            "### Vulnerability Summary",
            "",
            f"**Total Vulnerabilities:** {vuln_total:,}",
            "",
        ]

        if vuln_sev:
            lines += [
                "| Severity | Count |",
                "|----------|------:|",
                f"| 🔴 Critical | {vuln_sev.get('critical', 0):,} |",
                f"| 🟠 High | {vuln_sev.get('high', 0):,} |",
                f"| 🟡 Medium | {vuln_sev.get('medium', 0):,} |",
                f"| 🟢 Low | {vuln_sev.get('low', 0):,} |",
                "",
            ]
        elif vuln_total == 0:
            lines.append("*No vulnerability data available.*")
            lines.append("")

        # Recommendations
        if recs:
            lines += [
                "### Recommendations",
                "",
            ]
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # RULES SUMMARY FORMATTER
    # ================================================================

    def format_rules_summary_response(self, raw_text: str) -> str:
        """
        Format Wazuh rules summary.
        Input: JSON with total_rules, rules_loaded, by_level, by_file, top_groups, sample_rules.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "## Wazuh Rules Summary\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Wazuh Rules Summary\n\n```\n{raw_text[:2000]}\n```"

        total_rules = data.get("total_rules", 0)
        rules_loaded = data.get("rules_loaded", 0)
        by_level = data.get("by_level", {})
        by_file = data.get("by_file", {})
        top_groups = data.get("top_groups", {})
        sample_rules = data.get("sample_rules", [])

        lines = [
            "## Wazuh Rules Summary",
            "",
            f"**Total Rules:** {total_rules:,} | **Loaded (sample):** {rules_loaded:,}",
            "",
        ]

        # Rules by severity level (aggregated)
        if by_level:
            crit = sum(v for k, v in by_level.items() if int(k) >= 12)
            high = sum(v for k, v in by_level.items() if 7 <= int(k) < 12)
            med = sum(v for k, v in by_level.items() if 4 <= int(k) < 7)
            low = sum(v for k, v in by_level.items() if int(k) < 4)

            lines += [
                "### Rules by Severity",
                "",
                "| Category | Level Range | Count |",
                "|----------|:----------:|------:|",
                f"| 🔴 Critical | 12-15 | {crit:,} |",
                f"| 🟠 High | 7-11 | {high:,} |",
                f"| 🟡 Medium | 4-6 | {med:,} |",
                f"| 🟢 Low | 0-3 | {low:,} |",
                "",
            ]

            # Detailed per-level breakdown
            lines += [
                "### Level Breakdown",
                "",
                "| Level | Count |",
                "|------:|------:|",
            ]
            for lvl in sorted(by_level.keys(), key=lambda x: int(x)):
                lines.append(f"| {lvl} | {by_level[lvl]:,} |")
            lines.append("")

        # Top rule files
        if by_file:
            lines += [
                "### Top Rule Files",
                "",
                "| Filename | Rules |",
                "|----------|------:|",
            ]
            for fname, count in list(by_file.items())[:10]:
                lines.append(f"| {fname} | {count:,} |")
            lines.append("")

        # Top groups
        if top_groups:
            lines += [
                "### Top Rule Groups",
                "",
                "| Group | Rules |",
                "|-------|------:|",
            ]
            for group, count in list(top_groups.items())[:10]:
                lines.append(f"| {group} | {count:,} |")
            lines.append("")

        # Sample rules
        if sample_rules:
            lines += [
                "### Sample Rules",
                "",
                "| ID | Level | Description | Groups |",
                "|----|------:|-------------|--------|",
            ]
            for rule in sample_rules[:10]:
                rid = rule.get("id", "N/A")
                level = rule.get("level", 0)
                desc = str(rule.get("description", "N/A"))[:50]
                groups = ", ".join(rule.get("groups", [])[:3])
                lines.append(f"| {rid} | {level} | {desc} | {groups} |")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # RULE TRIGGER ANALYSIS FORMATTER
    # ================================================================

    def format_rule_trigger_response(self, raw_text: str) -> str:
        """Format rule trigger analysis — top firing rules, noisy rules, high-fidelity rules."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Rule Trigger Analysis\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Rule Trigger Analysis\n\n```\n{raw_text[:2000]}\n```"

        rules = data.get("rules", [])
        total_alerts = data.get("total_alerts", 0)
        time_range = data.get("time_range", "24h")
        noisy_rules = data.get("noisy_rules", [])
        high_fidelity = data.get("high_fidelity_rules", [])

        lines = [
            f"## Rule Trigger Analysis -- Last {time_range}",
            "",
            f"**Total Alerts:** {total_alerts:,}",
            f"**Rules Analyzed:** {len(rules)}",
            "",
        ]

        if rules:
            lines += [
                "### Top Firing Rules",
                "",
                "| Rule ID | Description | Level | Triggers | MITRE Technique |",
                "|---------|-------------|------:|--------:|-----------------|",
            ]
            for r in rules[:20]:
                rid = r.get("rule_id", "N/A")
                desc = str(r.get("description", "N/A"))[:50]
                level = r.get("level", "N/A")
                count = r.get("count", 0)
                mitre = ", ".join(r.get("mitre_techniques", [])[:2]) or "N/A"
                icon = "🔴" if isinstance(level, int) and level >= 12 else "🟠" if isinstance(level, int) and level >= 7 else "🟡" if isinstance(level, int) and level >= 4 else "🟢"
                lines.append(f"| {rid} | {desc} | {icon} {level} | {count:,} | {mitre} |")
            lines.append("")

        if noisy_rules:
            lines += [
                "### Noisy Rules (High Volume, Low Severity)",
                "",
            ]
            for r in noisy_rules[:10]:
                rid = r.get("rule_id", "N/A")
                count = r.get("count", 0)
                desc = str(r.get("description", ""))[:60]
                lines.append(f"- **Rule {rid}** ({count:,} triggers): {desc}")
            lines.append("")
            lines.append("*Consider tuning or suppressing these rules to reduce alert fatigue.*")
            lines.append("")

        if high_fidelity:
            lines += [
                "### High Fidelity Rules (Low Volume, High Severity)",
                "",
            ]
            for r in high_fidelity[:10]:
                rid = r.get("rule_id", "N/A")
                count = r.get("count", 0)
                level = r.get("level", "N/A")
                desc = str(r.get("description", ""))[:60]
                lines.append(f"- **Rule {rid}** (level {level}, {count} triggers): {desc}")
            lines.append("")
            lines.append("> These rules indicate genuine threats - prioritize investigation.")
            lines.append("")

        # Alert fatigue warning if top rules dominate
        if rules and total_alerts > 0:
            top_rule_count = rules[0].get("count", 0) if rules else 0
            top_pct = (top_rule_count / total_alerts * 100) if total_alerts > 0 else 0
            if top_pct > 80:
                lines.append(f"> **Alert Fatigue Warning:** Top rule generates {top_pct:.0f}% of all alerts. Consider tuning or suppressing to reduce noise.")
                lines.append("")

        # Navigation hints
        lines.append("---")
        lines.append("**Related queries:** `show mitre coverage` | `show recent alerts` | `analyze alert patterns` | `show alert summary`")

        return "\n".join(lines)

    # ================================================================
    # MITRE COVERAGE FORMATTER
    # ================================================================

    def format_mitre_coverage_response(self, raw_text: str) -> str:
        """Format MITRE ATT&CK coverage analysis."""
        if not raw_text or not isinstance(raw_text, str):
            return "## MITRE ATT&CK Coverage\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## MITRE ATT&CK Coverage\n\n```\n{raw_text[:2000]}\n```"

        techniques = data.get("techniques", [])
        tactic_counts = data.get("tactic_counts", {})
        coverage_count = data.get("coverage_count", len(techniques))
        uncovered_tactics = data.get("uncovered_tactics", [])
        time_range = data.get("time_range", "7d")

        lines = [
            f"## MITRE ATT&CK Coverage -- Last {time_range}",
            "",
            f"**Techniques Detected:** {coverage_count}",
            "",
        ]

        if tactic_counts:
            lines += [
                "### Tactic Coverage",
                "",
                "| Tactic | Detections |",
                "|--------|----------:|",
            ]
            for tactic, count in sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {tactic} | {count:,} |")
            lines.append("")

        if techniques:
            lines += [
                "### Top Techniques",
                "",
                "| Technique | Tactic | Detection Count |",
                "|-----------|--------|----------------:|",
            ]
            for t in techniques[:25]:
                tech_name = t.get("technique", "N/A")
                tactics = ", ".join(t.get("tactics", [])[:2]) or "N/A"
                count = t.get("count", 0)
                lines.append(f"| {tech_name} | {tactics} | {count:,} |")
            lines.append("")

        if uncovered_tactics:
            lines += [
                "### Coverage Gaps",
                "",
                "The following MITRE ATT&CK tactics have **no active detections:**",
                "",
            ]
            for tactic in uncovered_tactics:
                lines.append(f"- {tactic}")
            lines.append("")
            lines.append("*Consider adding detection rules for these tactics to improve coverage.*")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # ALERT TIMELINE FORMATTER
    # ================================================================

    def format_alert_timeline_response(self, raw_text: str) -> str:
        """Format alert timeline with spike detection and ASCII bar chart."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Alert Timeline\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Alert Timeline\n\n```\n{raw_text[:2000]}\n```"

        buckets = data.get("buckets", [])
        total_alerts = data.get("total_alerts", 0)
        average_rate = data.get("average_rate", 0)
        peak_time = data.get("peak_time", "N/A")
        spikes = data.get("spikes", [])
        interval = data.get("interval", "1h")
        time_range = data.get("time_range", "24h")

        lines = [
            f"## Alert Timeline -- Last {time_range} (interval: {interval})",
            "",
            f"**Total Alerts:** {total_alerts:,} | **Average Rate:** {average_rate:.1f}/interval | **Peak:** {peak_time}",
            "",
        ]

        # ASCII bar chart
        if buckets:
            max_count = max((b.get("count", 0) for b in buckets), default=1) or 1
            lines.append("### Activity Chart")
            lines.append("")
            lines.append("```")
            for b in buckets:
                ts = str(b.get("timestamp", ""))
                # Try to extract hour portion
                if "T" in ts:
                    label = ts.split("T")[1][:5] if len(ts) > 11 else ts[:16]
                else:
                    label = ts[:16]
                count = b.get("count", 0)
                bar_len = int((count / max_count) * 40) if max_count > 0 else 0
                bar = "█" * bar_len
                spike_marker = " *** SPIKE" if any(s.get("timestamp") == b.get("timestamp") for s in spikes) else ""
                lines.append(f"{label:>16} | {bar:<40} {count:>6}{spike_marker}")
            lines.append("```")
            lines.append("")

        if spikes:
            lines += [
                "### Spike Detection",
                "",
                f"**{len(spikes)} spike(s) detected** (>2 standard deviations above mean):",
                "",
            ]
            for s in spikes:
                ts = str(s.get("timestamp", "N/A"))[:19].replace("T", " ")
                count = s.get("count", 0)
                deviation = s.get("deviation", 0)
                lines.append(f"- **{ts}**: {count:,} alerts ({deviation:.1f}x std deviation)")
            lines.append("")
        else:
            lines.append("### Spike Detection")
            lines.append("")
            lines.append("No significant spikes detected in this time range.")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # LOG SOURCE HEALTH FORMATTER
    # ================================================================

    def format_log_source_health_response(self, raw_text: str) -> str:
        """Format log source health — silent agents, event counts, ingestion status."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Log Source Health\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Log Source Health\n\n```\n{raw_text[:2000]}\n```"

        total_events = data.get("total_events", 0)
        agent_event_counts = data.get("agent_event_counts", [])
        silent_agents = data.get("silent_agents", [])
        low_volume_agents = data.get("low_volume_agents", [])
        remoted_summary = data.get("remoted_summary", {})
        time_range = data.get("time_range", "24h")
        ingestion_streams = data.get("ingestion_streams") or {}
        any_stream_active = data.get("any_stream_active", total_events > 0)
        silent_stream_count = data.get("silent_stream_count", 0)

        # Pre-compute headline KPI counts (used both for the Health Summary
        # block prepended below and re-used by the per-stream breakdown later).
        active_stream_count = sum(
            1 for s in ingestion_streams.values() if s.get("status") == "active"
        ) if ingestion_streams else 0
        errored_stream_count = sum(
            1 for s in ingestion_streams.values() if s.get("status") == "error"
        ) if ingestion_streams else 0
        silent_agent_count = len(silent_agents) if isinstance(silent_agents, list) else 0
        low_vol_agent_count = len(low_volume_agents) if isinstance(low_volume_agents, list) else 0

        # Overall verdict — only "all silent / errored" is a true blackout.
        # Mixed states mean some pipelines are flowing; the LLM SOC analysis
        # should treat that accordingly instead of escalating to "complete
        # failure" (which is what the old fallback did).
        if ingestion_streams:
            stream_count = len(ingestion_streams)
            active_count = sum(
                1 for s in ingestion_streams.values()
                if s.get("status") == "active"
            )
            error_count = sum(
                1 for s in ingestion_streams.values()
                if s.get("status") == "error"
            )
            silent_only_count = sum(
                1 for s in ingestion_streams.values()
                if s.get("status") == "silent"
            )
            if not any_stream_active:
                overall = "🔴 **CRITICAL** — all ingestion streams silent or errored"
            elif silent_stream_count == 0:
                overall = "🟢 **Healthy** — all ingestion streams active"
            else:
                parts = [f"{active_count} of {stream_count} streams active"]
                if silent_only_count:
                    parts.append(f"{silent_only_count} silent")
                if error_count:
                    parts.append(f"{error_count} errored")
                overall = "🟡 **Mixed** — " + ", ".join(parts)
        else:
            overall = "**Total Events Ingested:** {:,}".format(total_events)

        lines = [
            "## Health Summary",
            "",
            f"**Total Events:** {total_events}",
            f"**Active Streams:** {active_stream_count}",
            f"**Silent Streams:** {silent_stream_count}",
            f"**Errored Streams:** {errored_stream_count}",
            f"**Silent Agents:** {silent_agent_count}",
            f"**Low Volume Agents:** {low_vol_agent_count}",
            "",
            f"## Log Source Health -- Last {time_range}",
            "",
            f"**Overall Health:** {overall}",
            "",
        ]

        # Per-stream breakdown — the key fix: shows that vulnerabilities can be
        # flowing even when alerts are silent, AND distinguishes "endpoint
        # returned 0 events" (silent) from "endpoint returned 404 / errored"
        # (error) so the SOC sees the right remediation hint.
        if ingestion_streams:
            STREAM_LABELS = {
                "alerts":          "Alerts",
                "vulnerabilities": "Vulnerabilities",
                "remoted":         "Remoted Service",
                "logcollector":    "Log Collector",
            }
            STATUS_ICONS = {
                "active":     "🟢",
                "low_volume": "🟡",
                "silent":     "🔴",
                "error":      "⚠️",
            }

            lines += [
                "### Ingestion Streams",
                "",
                "| Source | Status | Events | Index / Source |",
                "|--------|--------|-------:|----------------|",
            ]
            for key, info in ingestion_streams.items():
                label = STREAM_LABELS.get(key, key.title())
                status = info.get("status", "silent")
                icon = STATUS_ICONS.get(status, "⚪")
                count = info.get("total", 0)
                index = info.get("indexed_in", "-")
                lines.append(
                    f"| {icon} {label} | {status} | {count:,} | `{index}` |"
                )
            lines.append("")

            # If any stream surfaced an error, render its message verbatim so
            # the SOC can act on it. This is what made "Total Events: 0" so
            # opaque before — the underlying 404/timeout was being swallowed.
            errored = [
                (k, v.get("error", "")) for k, v in ingestion_streams.items()
                if v.get("status") == "error" and v.get("error")
            ]
            if errored:
                lines.append("### Stream Errors")
                lines.append("")
                for k, msg in errored:
                    label = STREAM_LABELS.get(k, k.title())
                    lines.append(f"- **{label}:** `{msg}`")
                lines.append("")
                lines.append(
                    "> ⚠️ One or more ingestion endpoints failed to respond. "
                    "The streams above are NOT confirmed silent — they're "
                    "uncheckable. Common causes: module disabled in `ossec.conf`, "
                    "endpoint returns 404 on this Wazuh version, or the indexer "
                    "is missing the expected index template."
                )
                lines.append("")

            # Helpful nuance line for genuine silence (not error)
            silent_streams = [k for k, v in ingestion_streams.items()
                              if v.get("status") == "silent"]
            if silent_streams and any_stream_active:
                silent_names = ", ".join(
                    STREAM_LABELS.get(k, k.title()) for k in silent_streams
                )
                lines.append(
                    f"> ℹ️ The following stream(s) are silent: **{silent_names}**. "
                    f"Other streams are flowing — this is NOT a total blackout, "
                    f"only the silent stream(s) need investigation."
                )
                lines.append("")

        if remoted_summary:
            lines += [
                "### Remoted Service Detail",
                "",
            ]
            for key, val in remoted_summary.items():
                lines.append(f"- **{key}:** {val}")
            lines.append("")

        if agent_event_counts:
            lines += [
                "### Agent Event Counts",
                "",
                "| Agent ID | Agent Name | Alerts | Vulns | Total | Status |",
                "|----------|------------|-------:|------:|------:|--------|",
            ]
            sorted_agents = sorted(
                agent_event_counts,
                key=lambda x: (x.get("alert_count", 0) + x.get("vuln_count", 0)),
                reverse=True,
            )
            for a in sorted_agents[:20]:
                aid = a.get("agent_id", "N/A")
                name = a.get("agent_name", "Unknown")
                alert_count = a.get("alert_count", a.get("event_count", 0))
                vuln_count = a.get("vuln_count", 0)
                total = alert_count + vuln_count
                if total >= 10:
                    icon, status = "🟢", "Normal"
                elif total > 0:
                    icon, status = "🟡", "Low Volume"
                else:
                    icon, status = "🔴", "Silent"
                lines.append(
                    f"| {aid} | {name} | {alert_count:,} | {vuln_count:,} | "
                    f"{total:,} | {icon} {status} |"
                )
            lines.append("")

        if silent_agents:
            lines += [
                "### Silent Agents (Active but 0 Events)",
                "",
            ]
            for a in silent_agents:
                aid = a.get("id", a.get("agent_id", "N/A"))
                name = a.get("name", a.get("agent_name", "Unknown"))
                lines.append(f"- **Agent {aid}** ({name}) — active but sending no events")
            lines.append("")
            lines.append("*Silent agents may indicate broken log collection or network issues.*")
            lines.append("")

        if low_volume_agents:
            lines += [
                "### Low Volume Agents (<10 events)",
                "",
            ]
            for a in low_volume_agents:
                aid = a.get("id", a.get("agent_id", "N/A"))
                name = a.get("name", a.get("agent_name", "Unknown"))
                count = a.get("event_count", 0)
                lines.append(f"- **Agent {aid}** ({name}): {count} events")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # DECODER ANALYSIS FORMATTER
    # ================================================================

    def format_decoder_analysis_response(self, raw_text: str) -> str:
        """Format decoder analysis — active decoders, coverage, custom vs built-in."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Decoder Analysis\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Decoder Analysis\n\n```\n{raw_text[:2000]}\n```"

        decoders = data.get("decoders", [])
        total = data.get("total", len(decoders))
        custom_count = data.get("custom_count", 0)
        builtin_count = data.get("builtin_count", 0)
        by_file = data.get("by_file", {})

        lines = [
            "## Decoder Analysis",
            "",
            f"**Total Decoders:** {total}",
            f"**Built-in:** {builtin_count} | **Custom:** {custom_count}",
            "",
        ]

        if by_file:
            lines += [
                "### Decoders by File",
                "",
                "| File | Count | Type |",
                "|------|------:|------|",
            ]
            for fname, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:15]:
                dtype = "Custom" if "local" in fname.lower() or "custom" in fname.lower() else "Built-in"
                lines.append(f"| {fname} | {count} | {dtype} |")
            lines.append("")

        if decoders:
            lines += [
                "### Decoder List (sample)",
                "",
                "| Decoder | Parent | File |",
                "|---------|--------|------|",
            ]
            for d in decoders[:30]:
                name = d.get("name", "N/A")
                parent = d.get("parent", "N/A") or "N/A"
                dfile = d.get("file", "N/A")
                lines.append(f"| {name} | {parent} | {dfile} |")
            if len(decoders) > 30:
                lines.append(f"| ... | ... | *{len(decoders) - 30} more* |")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # FIM EVENTS FORMATTER
    # ================================================================

    def format_fim_events_response(self, raw_text: str) -> str:
        """Format File Integrity Monitoring events."""
        if not raw_text or not isinstance(raw_text, str):
            return "## File Integrity Monitoring\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## File Integrity Monitoring\n\n```\n{raw_text[:2000]}\n```"

        events = data.get("events", [])
        summary = data.get("summary", {})
        total = data.get("total", len(events))
        time_range = data.get("time_range", "24h")
        agent_id = data.get("agent_id", "all")

        added = summary.get("added", 0)
        modified = summary.get("modified", 0)
        deleted = summary.get("deleted", 0)

        scope = f"Agent {agent_id}" if agent_id and agent_id != "all" else "All Agents"

        lines = [
            f"## File Integrity Monitoring -- Last {time_range}",
            "",
            f"**Scope:** {scope} | **Total Events:** {total}",
            "",
            "### Change Summary",
            "",
            f"- **Added:** {added}",
            f"- **Modified:** {modified}",
            f"- **Deleted:** {deleted}",
            "",
        ]

        if events:
            lines += [
                "### FIM Events",
                "",
                "| Time | Agent | File | Change | User |",
                "|------|-------|------|--------|------|",
            ]
            for e in events[:30]:
                ts = str(e.get("timestamp", "N/A"))[:19].replace("T", " ")
                aid = e.get("agent_id", "N/A")
                aname = e.get("agent_name", "")
                agent_label = f"{aid}" if not aname else f"{aid} ({aname})"
                fpath = str(e.get("file_path") or e.get("syscheck_path", "N/A"))
                if len(fpath) > 50:
                    fpath = "..." + fpath[-47:]
                change = e.get("change_type") or e.get("syscheck_event", "N/A")
                user = e.get("user") or e.get("syscheck_uname", "N/A")
                lines.append(f"| {ts} | {agent_label} | `{fpath}` | {change} | {user} |")

            if total > 30:
                lines.append(f"| ... | ... | ... | ... | *{total - 30} more events* |")
            lines.append("")

        # Highlight critical file changes
        critical_paths = ["/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/ssh/sshd_config",
                         "/etc/hosts", "/etc/crontab", "/root/.ssh", "/etc/pam.d"]
        critical_events = [e for e in events if any(cp in str(e.get("file_path", "")) for cp in critical_paths)]
        if critical_events:
            lines += [
                "### Critical File Changes",
                "",
            ]
            for e in critical_events[:10]:
                fpath = e.get("file_path") or e.get("syscheck_path", "N/A")
                change = e.get("change_type") or e.get("syscheck_event", "N/A")
                user = e.get("user") or e.get("syscheck_uname", "N/A")
                lines.append(f"- **{fpath}** — {change} by {user}")
            lines.append("")
            lines.append("> Changes to critical system files require immediate investigation.")
            lines.append("")

        # Navigation hints
        lines.append("---")
        lines.append("**Related queries:** `show processes for agent <ID>` | `check agent <ID> health` | `show recent alerts` | `show sca results for agent <ID>`")

        return "\n".join(lines)

    # ================================================================
    # SCA RESULTS FORMATTER
    # ================================================================

    def format_sca_results_response(self, raw_text: str) -> str:
        """Format Security Configuration Assessment results."""
        if not raw_text or not isinstance(raw_text, str):
            return "## SCA Results\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## SCA Results\n\n```\n{raw_text[:2000]}\n```"

        agent_id = data.get("agent_id", "N/A")
        policies = data.get("policies", [])
        summary = data.get("summary", {})
        failed_checks = data.get("failed_checks", [])

        total_checks = summary.get("total_checks", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        score = summary.get("score", 0)

        score_icon = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"

        lines = [
            f"## SCA Results -- Agent {agent_id}",
            "",
            f"**Overall Score:** {score_icon} {score:.0f}%",
            f"**Total Checks:** {total_checks} | **Passed:** {passed} | **Failed:** {failed}",
            "",
        ]

        if policies:
            lines += [
                "### Policies",
                "",
                "| Policy | Score | Pass | Fail | Not Applicable |",
                "|--------|------:|-----:|-----:|---------------:|",
            ]
            for p in policies:
                pname = str(p.get("name", p.get("policy_id", "N/A")))[:40]
                pscore = p.get("score", 0)
                ppass = p.get("pass", 0)
                pfail = p.get("fail", 0)
                pna = p.get("invalid", 0)
                p_icon = "🟢" if pscore >= 80 else "🟡" if pscore >= 60 else "🔴"
                lines.append(f"| {pname} | {p_icon} {pscore}% | {ppass} | {pfail} | {pna} |")
            lines.append("")

        if failed_checks:
            lines += [
                "### Top Failed Checks",
                "",
                "| Check | Result | Remediation |",
                "|-------|--------|-------------|",
            ]
            for c in failed_checks[:15]:
                title = str(c.get("title", c.get("description", "N/A")))[:50]
                result = c.get("result", "failed")
                remediation = str(c.get("remediation", c.get("rationale", "N/A")))[:60]
                lines.append(f"| {title} | {result} | {remediation} |")
            if len(failed_checks) > 15:
                lines.append(f"| ... | ... | *{len(failed_checks) - 15} more failed checks* |")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # AGENT INVENTORY FORMATTER
    # ================================================================

    def format_agent_inventory_response(self, raw_text: str) -> str:
        """Format agent system inventory — hardware, OS, packages, network interfaces."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Agent Inventory\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Agent Inventory\n\n```\n{raw_text[:2000]}\n```"

        agent_id = data.get("agent_id", "N/A")
        hardware = data.get("hardware", {})
        os_info = data.get("os", {})
        packages = data.get("packages", [])
        netiface = data.get("network_interfaces", [])
        package_count = data.get("package_count", len(packages))

        # Compute headline numeric KPIs for the summary block.
        # Only numeric values — the SPA's KPI parser requires the value to
        # start with a digit. OS name lives in the Operating System table below.
        cpu = hardware.get("cpu", {}) if isinstance(hardware.get("cpu"), dict) else {}
        ram = hardware.get("ram", {}) if isinstance(hardware.get("ram"), dict) else {}
        cpu_cores = cpu.get("cores") or 0
        # ram.total is in KB (per existing code at line ~3755). 1 GB = 1024*1024 KB.
        ram_kb = ram.get("total") or 0
        try:
            ram_gb = round(int(ram_kb) / 1048576) if ram_kb else 0
        except (TypeError, ValueError):
            ram_gb = 0
        netiface_count = len(netiface) if isinstance(netiface, list) else 0

        lines = [
            f"## System Inventory -- Agent {agent_id}",
            "",
            "## System Summary",
            "",
            f"**CPU Cores:** {cpu_cores}",
            f"**RAM GB:** {ram_gb}",
            f"**Packages:** {package_count}",
            f"**Network Interfaces:** {netiface_count}",
            "",
        ]

        # OS Information
        if os_info:
            lines += [
                "### Operating System",
                "",
                "| Field | Value |",
                "|-------|-------|",
            ]
            for key in ["name", "version", "architecture", "os_release", "os_name", "os_version", "hostname", "sysname"]:
                val = os_info.get(key)
                if val:
                    lines.append(f"| {key.replace('_', ' ').title()} | {val} |")
            lines.append("")

        # Hardware
        if hardware:
            lines += [
                "### Hardware",
                "",
                "| Field | Value |",
                "|-------|-------|",
            ]
            cpu = hardware.get("cpu", {})
            if isinstance(cpu, dict):
                lines.append(f"| CPU | {cpu.get('name', 'N/A')} |")
                lines.append(f"| CPU Cores | {cpu.get('cores', 'N/A')} |")
                lines.append(f"| CPU MHz | {cpu.get('mhz', 'N/A')} |")
            ram = hardware.get("ram", {})
            if isinstance(ram, dict):
                total_mb = ram.get("total", 0)
                free_mb = ram.get("free", 0)
                if total_mb:
                    lines.append(f"| RAM Total | {total_mb:,} KB |")
                if free_mb:
                    lines.append(f"| RAM Free | {free_mb:,} KB |")
            board = hardware.get("board_serial", hardware.get("serial", ""))
            if board:
                lines.append(f"| Board Serial | {board} |")
            lines.append("")

        # Network Interfaces
        if netiface:
            lines += [
                "### Network Interfaces",
                "",
                "| Name | MAC | State | MTU | Type |",
                "|------|-----|-------|----:|------|",
            ]
            for iface in netiface[:10]:
                name = iface.get("name", "N/A")
                mac = iface.get("mac", "N/A")
                state = iface.get("state", "N/A")
                mtu = iface.get("mtu", "N/A")
                itype = iface.get("type", "N/A")
                lines.append(f"| {name} | {mac} | {state} | {mtu} | {itype} |")
            lines.append("")

        # Packages
        if packages:
            lines += [
                f"### Installed Packages ({package_count} total)",
                "",
                "| Name | Version | Architecture | Format |",
                "|------|---------|-------------|--------|",
            ]
            for p in packages[:20]:
                pname = str(p.get("name", "N/A"))[:30]
                pver = str(p.get("version", "N/A"))[:20]
                parch = p.get("architecture", "N/A")
                pfmt = p.get("format", "N/A")
                lines.append(f"| {pname} | {pver} | {parch} | {pfmt} |")
            if package_count > 20:
                lines.append(f"| ... | ... | ... | *{package_count - 20} more packages* |")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # ROOTCHECK FORMATTER
    # ================================================================

    def format_rootcheck_response(self, raw_text: str) -> str:
        """Format rootcheck scan results — rootkits, trojans, anomalies."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Rootcheck Results\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Rootcheck Results\n\n```\n{raw_text[:2000]}\n```"

        agent_id = data.get("agent_id", "N/A")
        results = data.get("results", [])
        summary = data.get("summary", {})
        last_scan = data.get("last_scan", "N/A")

        total = summary.get("total", len(results))
        trojans = summary.get("trojans", 0)
        rootkits = summary.get("rootkits", 0)
        anomalies = summary.get("anomalies", 0)
        system_audit = summary.get("system_audit", 0)

        threat_icon = "🔴" if (trojans + rootkits) > 0 else "🟡" if anomalies > 0 else "🟢"

        lines = [
            f"## Rootcheck Results -- Agent {agent_id}",
            "",
            f"**Status:** {threat_icon} {'Threats Detected' if (trojans + rootkits) > 0 else 'No Rootkits Found' if (trojans + rootkits) == 0 else 'Anomalies Found'}",
            f"**Last Scan:** {str(last_scan)[:19].replace('T', ' ') if last_scan != 'N/A' else 'N/A'}",
            "",
            "### Summary",
            "",
            "| Category | Count |",
            "|----------|------:|",
            f"| Total Findings | {total} |",
            f"| Trojans | {trojans} |",
            f"| Rootkits | {rootkits} |",
            f"| Anomalies | {anomalies} |",
            f"| System Audit | {system_audit} |",
            "",
        ]

        if results:
            lines += [
                "### Findings",
                "",
                "| Type | Description | Status |",
                "|------|-------------|--------|",
            ]
            for r in results[:30]:
                rtype = r.get("type", r.get("event", "N/A"))
                desc = str(r.get("description", r.get("title", "N/A")))[:60]
                status = r.get("status", "N/A")
                lines.append(f"| {rtype} | {desc} | {status} |")
            if len(results) > 30:
                lines.append(f"| ... | ... | *{len(results) - 30} more findings* |")
            lines.append("")

        if trojans > 0 or rootkits > 0:
            lines += [
                "### Immediate Actions Required",
                "",
                "1. Isolate the affected agent from the network",
                "2. Perform a full forensic investigation",
                "3. Compare file hashes against known-good baselines",
                "4. Check for unauthorized user accounts and SSH keys",
                "",
            ]

        return "\n".join(lines)

    # ================================================================
    # ALERT-VULNERABILITY CORRELATION FORMATTER
    # ================================================================

    def format_alert_vuln_correlation_response(self, raw_text: str) -> str:
        """Format alert-vulnerability cross-correlation results."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Alert-Vulnerability Correlation\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Alert-Vulnerability Correlation\n\n```\n{raw_text[:2000]}\n```"

        correlated_agents = data.get("correlated_agents", [])
        high_risk = data.get("high_risk_agents", [])
        summary = data.get("summary", {})
        time_range = data.get("time_range", "24h")

        agents_with_both = summary.get("agents_with_both", len(correlated_agents))
        critical_overlap = summary.get("critical_overlap", len(high_risk))

        lines = [
            f"## Alert-Vulnerability Correlation -- Last {time_range}",
            "",
            f"**Agents with Both Alerts & Vulnerabilities:** {agents_with_both}",
            f"**High Risk Agents (combined risk >= 50):** {critical_overlap}",
            "",
        ]

        if correlated_agents:
            lines += [
                "### Risk Matrix",
                "",
                "| Agent ID | Agent Name | Alert Total | Alert Critical | Vuln Total | Vuln Critical | Combined Risk |",
                "|----------|------------|------------:|---------------:|-----------:|--------------:|--------------:|",
            ]
            for a in correlated_agents[:20]:
                aid = a.get("agent_id", "N/A")
                aname = a.get("agent_name", "Unknown")
                ac = a.get("alert_counts", {})
                vc = a.get("vulnerability_counts", {})
                risk = a.get("combined_risk", 0)
                risk_icon = "🔴" if risk >= 75 else "🟠" if risk >= 50 else "🟡" if risk >= 25 else "🟢"
                lines.append(
                    f"| {aid} | {aname} | {ac.get('total', 0)} | {ac.get('critical', 0)} "
                    f"| {vc.get('total', 0)} | {vc.get('critical', 0)} | {risk_icon} {risk} |"
                )
            lines.append("")

        if high_risk:
            lines += [
                "### Prioritized Actions",
                "",
            ]
            for i, a in enumerate(high_risk[:5], 1):
                aid = a.get("agent_id", "N/A")
                aname = a.get("agent_name", "Unknown")
                risk = a.get("combined_risk", 0)
                lines.append(f"{i}. **Agent {aid} ({aname})** — Risk: {risk}/100 — Patch vulnerabilities and investigate alerts")
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # BEHAVIORAL BASELINE FORMATTER
    # ================================================================

    def format_behavioral_baseline_response(self, raw_text: str) -> str:
        """Format behavioral baseline comparison — deviation detection."""
        if not raw_text or not isinstance(raw_text, str):
            return "## Behavioral Baseline\n\n*No data available*"

        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            parts = raw_text.split('\n', 1)
            if len(parts) == 2:
                try:
                    data = json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            return f"## Behavioral Baseline\n\n```\n{raw_text[:2000]}\n```"

        baseline_period = data.get("baseline_period", "7d")
        daily_average = data.get("daily_average", 0)
        today_count = data.get("today_count", 0)
        deviation_sigma = data.get("deviation_sigma", 0)
        status = data.get("status", "unknown")
        daily_breakdown = data.get("daily_breakdown", [])
        agent_id = data.get("agent_id")
        stddev = data.get("stddev", 0)

        status_icons = {"normal": "🟢 NORMAL", "elevated": "🟡 ELEVATED", "anomalous": "🔴 ANOMALOUS"}
        status_display = status_icons.get(status, f"⚪ {status.upper()}")

        scope = f"Agent {agent_id}" if agent_id else "All Agents"

        lines = [
            f"## Behavioral Baseline -- {scope}",
            "",
            f"**Status:** {status_display}",
            f"**Baseline Period:** {baseline_period}",
            f"**Daily Average:** {daily_average:.1f} alerts/day",
            f"**Today's Count:** {today_count:,}",
            f"**Deviation:** {deviation_sigma:.2f} sigma",
            "",
        ]

        # Sparkline / daily trend
        if daily_breakdown:
            max_count = max((d.get("count", 0) for d in daily_breakdown), default=1) or 1
            lines.append("### Daily Trend")
            lines.append("")
            lines.append("```")
            for d in daily_breakdown:
                day = str(d.get("date", d.get("timestamp", "")))[:10]
                count = d.get("count", 0)
                bar_len = int((count / max_count) * 30) if max_count > 0 else 0
                bar = "█" * bar_len
                lines.append(f"{day:>12} | {bar:<30} {count:>6}")
            lines.append("```")
            lines.append("")

        # Deviation explanation
        lines.append("### Analysis")
        lines.append("")
        if status == "normal":
            lines.append(f"Alert activity ({today_count}) is within normal range (avg: {daily_average:.0f}, stddev: {stddev:.1f}).")
            lines.append("No anomalous behavior detected.")
        elif status == "elevated":
            lines.append(f"Alert activity ({today_count}) is **elevated** — {deviation_sigma:.1f}x standard deviations above the mean.")
            lines.append("This may indicate increased threat activity or configuration changes.")
            lines.append("")
            lines.append("**Recommended Actions:**")
            lines.append("1. Review the latest alerts for new rule triggers")
            lines.append("2. Check for recent configuration or deployment changes")
            lines.append("3. Monitor over the next few hours for further increases")
        elif status == "anomalous":
            lines.append(f"Alert activity ({today_count}) is **anomalous** — {deviation_sigma:.1f}x standard deviations above the mean.")
            lines.append("This is a significant deviation requiring immediate attention.")
            lines.append("")
            lines.append("**Recommended Actions:**")
            lines.append("1. Immediately review high-severity alerts")
            lines.append("2. Check for active security incidents")
            lines.append("3. Investigate any new attack vectors or compromised systems")
            lines.append("4. Consider escalating to incident response")
        lines.append("")

        return "\n".join(lines)

    def format_decoder_generator_response(self, raw_text: str) -> str:
        """Format decoder generation results with validation details."""
        lines = []
        lines.append("## Wazuh Decoder Generator")
        lines.append("")

        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            lines.append("Could not parse decoder generation results.")
            lines.append("")
            lines.append(f"```\n{str(raw_text)[:2000]}\n```")
            return "\n".join(lines)

        status = data.get("status", "unknown")
        attempts = data.get("attempts", [])
        attempt_count = len(attempts)

        if status == "success":
            lines.append(f"**Status:** SUCCESS (validated in {attempt_count} attempt{'s' if attempt_count != 1 else ''})")
        else:
            lines.append(f"**Status:** FAILED ({attempt_count} attempt{'s' if attempt_count != 1 else ''})")

        log_format = data.get("log_format", "syslog")
        device_type = data.get("device_type", "auto-detected")
        vendor = data.get("vendor", "auto-detected")
        lines.append(f"**Log Format:** {log_format} | **Device:** {device_type} | **Vendor:** {vendor}")
        lines.append("")

        # Input log
        raw_log = data.get("raw_log", "")
        if raw_log:
            lines.append("### Input Log")
            lines.append(f"```")
            lines.append(raw_log[:500])
            lines.append(f"```")
            lines.append("")

        # Generated decoder XML
        decoder_xml = data.get("decoder_xml")
        if decoder_xml:
            lines.append("### Generated Decoder")
            lines.append("```xml")
            lines.append(decoder_xml)
            lines.append("```")
            lines.append("")

        # Extracted fields preview
        extracted = data.get("extracted_fields", {})
        if extracted:
            lines.append("### Extracted Fields Preview")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            for field, value in extracted.items():
                lines.append(f"| {field} | {value} |")
            lines.append("")

        # Validation details per attempt
        if attempts:
            lines.append("### Validation Details")
            lines.append("")
            for att in attempts:
                num = att.get("attempt", "?")
                att_status = att.get("status", "unknown")
                errors = att.get("errors", [])
                validation = att.get("validation", {})
                field_count = len(validation.get("extracted_fields", {}))

                if att_status == "success":
                    lines.append(f"- Attempt {num}: Passed -- XML valid, regex matches, {field_count} fields extracted")
                elif att_status == "validation_failed":
                    err_summary = "; ".join(errors[:2]) if errors else "validation failed"
                    lines.append(f"- Attempt {num}: Failed -- {err_summary}")
                else:
                    err_summary = "; ".join(errors[:2]) if errors else att_status
                    lines.append(f"- Attempt {num}: {att_status} -- {err_summary}")
            lines.append("")

        # Logtest output summary
        logtest = data.get("logtest_output")
        if logtest and isinstance(logtest, dict) and not logtest.get("error"):
            output = logtest.get("output", logtest)
            if isinstance(output, dict):
                predecoder = output.get("predecoder", {})
                if predecoder:
                    pd_parts = []
                    if predecoder.get("hostname"):
                        pd_parts.append(f"hostname={predecoder['hostname']}")
                    if predecoder.get("program_name"):
                        pd_parts.append(f"program={predecoder['program_name']}")
                    if pd_parts:
                        lines.append(f"**Predecoder:** {', '.join(pd_parts)}")
                        lines.append("")

        # Suggested next steps
        lines.append("### Suggested Next Steps")
        lines.append("")
        if status == "success":
            lines.append("1. Save decoder to `/var/ossec/etc/decoders/local_decoder.xml`")
            lines.append("2. Run: `sudo /var/ossec/bin/wazuh-logtest` to verify manually")
            lines.append("3. Restart Wazuh: `sudo systemctl restart wazuh-manager`")
            lines.append("4. Consider creating matching rules for this decoder")
        else:
            lines.append("1. Review the validation errors above")
            lines.append("2. Try providing more hints: device_type, vendor, expected_fields")
            lines.append("3. Manually adjust the decoder XML and test with `wazuh-logtest`")
            lines.append("4. Check Wazuh documentation for decoder syntax reference")
        lines.append("")

        return "\n".join(lines)

    # ================================================================
    # FALLBACK METHODS
    # ================================================================

    def format_general_response(self, data, title="Results", **kwargs):
        """Generic formatter for unknown types"""
        if isinstance(data, list) and data:
            return f"## {title}\n\n```json\n{json.dumps(data[:5], indent=2, default=str)[:2000]}\n```"
        elif isinstance(data, dict):
            return f"## {title}\n\n```json\n{json.dumps(data, indent=2, default=str)[:2000]}\n```"
        return f"## {title}\n\n*No data available*"

    def format_raw_json_response(self, raw_text: str, title: str = "Results") -> str:
        """
        Present raw text from MCP content wrapper cleanly.
        Handles: plain text, "Label:\n{json}", pure JSON strings.
        """
        if not raw_text or not isinstance(raw_text, str):
            return f"## {title}\n\n*No data available*"

        raw_text = raw_text.strip()

        lines = raw_text.split('\n', 1)
        if len(lines) == 2 and lines[0].strip().endswith(':'):
            label = lines[0].strip().rstrip(':')
            body = lines[1].strip()
            try:
                parsed = json.loads(body)
                return (
                    f"## {label}\n\n"
                    f"```json\n{json.dumps(parsed, indent=2, default=str)[:3000]}\n```"
                )
            except json.JSONDecodeError:
                return f"## {label}\n\n```\n{body[:3000]}\n```"

        try:
            parsed = json.loads(raw_text)
            return (
                f"## {title}\n\n"
                f"```json\n{json.dumps(parsed, indent=2, default=str)[:3000]}\n```"
            )
        except json.JSONDecodeError:
            pass

        return f"## {title}\n\n```\n{raw_text[:3000]}\n```"

    def format_statistics_response(self, correlated_data, summary, **kwargs):
        """Format statistics — legacy fallback, prefer format_raw_json_response"""
        try:
            if isinstance(correlated_data, str):
                return self.format_raw_json_response(correlated_data, title="Statistics")
            return f"```json\n{json.dumps(correlated_data, indent=2, default=str)[:2000]}\n```"
        except Exception:
            return "```\nStatistics data unavailable\n```"

    def format_analysis_response(self, correlated_data, summary, **kwargs):
        """Format analysis"""
        return self.format_general_response(correlated_data, "Analysis")

    # ========================================================================
    # SURICATA IDS FORMATTERS
    # ========================================================================

    SURICATA_SEVERITY_MAP = {
        "1": "Critical",
        "2": "High",
        "3": "Medium",
        1: "Critical",
        2: "High",
        3: "Medium",
    }

    def _suricata_severity_label(self, sev) -> str:
        """Map Suricata numeric severity to label."""
        return self.SURICATA_SEVERITY_MAP.get(sev, "Low")

    def format_suricata_alerts_response(self, raw_text: str) -> str:
        """Format Suricata IDS alerts into a SOC-grade table."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata IDS Alerts\n\n{raw_text}"

        items = []
        if isinstance(data, dict):
            items = data.get("data", {}).get("affected_items", [])
            if not items:
                items = data.get("affected_items", [])
        elif isinstance(data, list):
            items = data

        total = len(items)
        if not items:
            return "## Suricata IDS Alerts\n\nNo Suricata alerts found for the specified filters.\n"

        # Filter out non-alert events (HTTP/TLS/flow) that lack alert fields
        items = [a for a in items if a.get("alert")]

        lines = [f"## Suricata IDS Alerts\n\n**Total Alerts:** {total}\n"]

        # Severity breakdown
        sev_counts = {}
        for a in items:
            alert = a.get("alert", {})
            sev = alert.get("severity", "?")
            label = self._suricata_severity_label(sev)
            sev_counts[label] = sev_counts.get(label, 0) + 1
        if sev_counts:
            lines.append("**Severity Breakdown:** " + " | ".join(
                f"{k}: {v}" for k, v in sorted(sev_counts.items())
            ) + "\n")

        # Check if MITRE data is present in any alert
        has_mitre = any(
            a.get("alert", {}).get("metadata", {}).get("mitre_tactic_name")
            for a in items[:50]
        )

        # Alert table (top 50)
        if has_mitre:
            lines.append("| # | Timestamp | Severity | Signature | Category | MITRE Tactic | Src IP | Dest IP | Action |")
            lines.append("|---|-----------|----------|-----------|----------|-------------|--------|---------|--------|")
        else:
            lines.append("| # | Timestamp | Severity | Signature | Category | Src IP | Dest IP | Proto |")
            lines.append("|---|-----------|----------|-----------|----------|--------|---------|-------|")
        for i, a in enumerate(items[:50], 1):
            ts = a.get("@timestamp", a.get("timestamp", ""))[:19]
            alert = a.get("alert", {})
            sev = self._suricata_severity_label(alert.get("severity", "?"))
            sig = (alert.get("signature", "N/A"))[:60]
            cat = (alert.get("category", "N/A"))[:30]
            src = a.get("src_ip", a.get("source", {}).get("ip", "N/A"))
            # Show XFF if present
            xff = alert.get("xff", "")
            if xff:
                src = f"{src} (XFF: {xff})"
            dst = a.get("dest_ip", a.get("destination", {}).get("ip", "N/A"))
            if has_mitre:
                mitre_tactics = alert.get("metadata", {}).get("mitre_tactic_name", [])
                mitre = ", ".join(mitre_tactics)[:30] if mitre_tactics else "N/A"
                action = alert.get("action", "N/A")
                lines.append(f"| {i} | {ts} | {sev} | {sig} | {cat} | {mitre} | {src} | {dst} | {action} |")
            else:
                proto = a.get("app_proto", a.get("proto", "N/A"))
                lines.append(f"| {i} | {ts} | {sev} | {sig} | {cat} | {src} | {dst} | {proto} |")

        if total > 50:
            lines.append(f"\n*Showing 50 of {total} alerts.*\n")

        return "\n".join(lines)

    def format_suricata_alert_summary_response(self, raw_text: str) -> str:
        """Format Suricata alert summary with breakdowns."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Alert Summary\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Alert Summary\n\n{raw_text}"

        total = data.get("total_alerts", 0)
        time_range = data.get("time_range", "24h")
        lines = [f"## Suricata IDS Alert Summary ({time_range})\n\n**Total Alerts:** {total}\n"]

        # Severity breakdown
        sev = data.get("severity_breakdown", {})
        if sev:
            lines.append("### Severity Breakdown\n")
            lines.append("| Severity | Label | Count |")
            lines.append("|----------|-------|-------|")
            for level, count in sorted(sev.items(), key=lambda x: str(x[0])):
                label = self._suricata_severity_label(level)
                lines.append(f"| {level} | {label} | {count} |")
            lines.append("")

        # Top categories
        cats = data.get("top_categories", [])
        if cats:
            lines.append("### Top Alert Categories\n")
            lines.append("| # | Category | Count |")
            lines.append("|---|----------|-------|")
            for i, c in enumerate(cats[:15], 1):
                lines.append(f"| {i} | {c.get('category', 'N/A')} | {c.get('count', 0)} |")
            lines.append("")

        # Top signatures
        sigs = data.get("top_signatures", [])
        if sigs:
            lines.append("### Top Firing Signatures\n")
            lines.append("| # | Signature | Count |")
            lines.append("|---|-----------|-------|")
            for i, s in enumerate(sigs[:15], 1):
                lines.append(f"| {i} | {s.get('signature', 'N/A')[:70]} | {s.get('count', 0)} |")
            lines.append("")

        return "\n".join(lines)

    # Suricata severity level → human-readable name.
    # MUST match the user's Network Threats Summary dashboard cards:
    #   severity 1 → "Alert"   (high-severity tier; biggest "alert" count)
    #   severity 2 → "Critical"
    #   severity 3 → "Warning" (medium tier)
    #   severity 4+ → "Other"  (low / info tier)
    # The dashboard intentionally inverts standard Suricata 1=most-critical
    # ordering for the top two tiers, so we follow the dashboard.
    _SURICATA_SEVERITY_NAMES = {1: "Alert", 2: "Critical", 3: "Warning", 4: "Other", 5: "Other"}

    def format_suricata_severity_alerts_response(
        self, raw_text: str, expected_severity: int = 1
    ) -> str:
        """Format Suricata alerts at a specific severity level.

        Defensively drops items whose `alert.severity` doesn't equal the expected
        value (CLAUDE.md §1: STRICT FILTERING). For ``expected_severity == 4`` we
        accept severity >= 4 (covers Suricata "low" + "info").
        """
        sev_name = self._SURICATA_SEVERITY_NAMES.get(
            expected_severity, f"Severity {expected_severity}"
        )

        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## {sev_name} Suricata Alerts\n\n{raw_text}"

        items = []
        true_total = 0  # from track_total_hits — accurate count in time range
        if isinstance(data, dict):
            items = data.get("data", {}).get("affected_items", []) or data.get("affected_items", [])
            # The indexer fills total_affected_items from hits.total.value (track_total_hits=True),
            # giving the TRUE count of matching docs even when only a sample is returned.
            true_total = data.get("data", {}).get("total_affected_items", 0) or len(items)

        time_range = (data.get("data", {}) or {}).get("time_range") or data.get("time_range") or "24h"

        # Defensive severity filter — never trust the data layer alone (CLAUDE.md §1).
        pre_count = len(items)

        def _sev(a):
            try:
                return int((a.get("alert") or {}).get("severity") or 0)
            except (TypeError, ValueError):
                return 0

        # Strict filter: tier filter MUST equal the expected severity exactly.
        # 'low' is the only family that spans severity 4+5; everything else is
        # a single integer match. This prevents Medium leaking into Low when
        # the indexer's range query would otherwise let it through.
        if expected_severity == 4:
            items = [a for a in items if _sev(a) in (4, 5)]
        else:
            items = [a for a in items if _sev(a) == expected_severity]

        dropped = pre_count - len(items)
        if dropped > 0:
            logger.info(
                f"[SURICATA_FORMATTER] dropped {dropped} non-{sev_name} alerts "
                f"(defensive severity filter, expected={expected_severity})"
            )

        sampled_count = len(items)
        if not items:
            return (
                f"## {sev_name} Suricata Alerts (Severity {expected_severity}) — Last {time_range}\n\n"
                f"No {sev_name.lower()} Suricata alerts found in the selected time range.\n"
            )

        # Header reflects TRUE total (from track_total_hits) and the size of the
        # sample we'll show details for. Analysts get the headline number that
        # matches the dashboard, plus full detail rows on the top 100.
        DETAIL_LIMIT = 100
        if true_total and true_total > sampled_count:
            lines = [
                f"## {sev_name} Suricata Alerts (Severity {expected_severity}) — Last {time_range}\n",
                f"**Total alerts in time range:** {true_total:,} | "
                f"**Detailed analysis below:** top {min(DETAIL_LIMIT, sampled_count)} of {sampled_count:,} returned\n",
                f"> ℹ️ The indexer returned a sample of {sampled_count:,} alerts (capped at the API limit). "
                f"For the full distribution, use `show alert summary` or narrow the time range.\n",
            ]
        else:
            lines = [
                f"## {sev_name} Suricata Alerts (Severity {expected_severity}) — Last {time_range}\n",
                f"**Total:** {sampled_count:,}\n",
            ]
        lines.append("| # | Timestamp | Severity | Signature | Category | Src IP | Dest IP | Proto |")
        lines.append("|---|-----------|---------:|-----------|----------|--------|---------|-------|")
        for i, a in enumerate(items[:DETAIL_LIMIT], 1):
            ts = str(a.get("@timestamp", ""))[:19]
            alert = a.get("alert", {}) or {}
            sev = _sev(a)
            sig = str(alert.get("signature", "N/A"))[:60]
            cat = str(alert.get("category", "N/A"))[:30]
            src = a.get("src_ip", "N/A")
            dst = a.get("dest_ip", "N/A")
            proto = a.get("app_proto", "N/A")
            lines.append(f"| {i} | {ts} | {sev} | {sig} | {cat} | {src} | {dst} | {proto} |")

        # If we capped detail rows, surface the remainder.
        # NOTE: do NOT reference an undefined `total` — that's what triggered
        # the "name 'total' is not defined" NameError that kicked every
        # Suricata severity response into Safe Fallback Mode.
        if sampled_count > DETAIL_LIMIT:
            lines.append(
                f"\n_Showing top {DETAIL_LIMIT} of {sampled_count:,} sampled alerts._"
            )

        return "\n".join(lines)

    def format_suricata_critical_alerts_response(self, raw_text: str) -> str:
        """Backwards-compat alias.

        Per the user's dashboard convention: Critical = severity 2 (NOT 1).
        Standard Suricata uses severity=1 for the highest tier, but this
        deployment's UI inverts the top two — so the alias must agree."""
        return self.format_suricata_severity_alerts_response(raw_text, expected_severity=2)

    def format_suricata_network_analysis_response(self, raw_text: str) -> str:
        """Format Suricata network analysis with top talkers and services."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Network Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Network Analysis\n\n{raw_text}"

        lines = ["## Suricata Network Analysis\n"]

        # Top source IPs (attackers) — also check "top_attackers" key from get_suricata_top_attackers
        src_ips = data.get("top_source_ips", []) or data.get("top_attackers", [])
        if src_ips:
            lines.append("### Top Source IPs (Potential Attackers)\n")
            lines.append("| # | Source IP | Alert Count |")
            lines.append("|---|----------|-------------|")
            for i, item in enumerate(src_ips[:15], 1):
                lines.append(f"| {i} | {item.get('ip', 'N/A')} | {item.get('count', 0)} |")
            lines.append("")

        # Top destination IPs (targets)
        dst_ips = data.get("top_dest_ips", [])
        if dst_ips:
            lines.append("### Top Destination IPs (Targets)\n")
            lines.append("| # | Destination IP | Alert Count |")
            lines.append("|---|----------------|-------------|")
            for i, item in enumerate(dst_ips[:15], 1):
                lines.append(f"| {i} | {item.get('ip', 'N/A')} | {item.get('count', 0)} |")
            lines.append("")

        # Top services/protocols
        services = data.get("top_services", [])
        if services:
            lines.append("### Top Services/Protocols\n")
            lines.append("| # | Service | Count |")
            lines.append("|---|---------|-------|")
            for i, item in enumerate(services[:10], 1):
                lines.append(f"| {i} | {item.get('service', 'N/A')} | {item.get('count', 0)} |")
            lines.append("")

        # Top hostnames
        src_hosts = data.get("top_src_hostnames", [])
        if src_hosts and src_hosts[0].get("hostname") != "N/A":
            lines.append("### Top Source Hostnames\n")
            lines.append("| # | Hostname | Count |")
            lines.append("|---|----------|-------|")
            for i, item in enumerate(src_hosts[:10], 1):
                lines.append(f"| {i} | {item.get('hostname', 'N/A')} | {item.get('count', 0)} |")
            lines.append("")

        return "\n".join(lines)

    def format_suricata_search_response(self, raw_text: str) -> str:
        """Format Suricata search results."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Search Results\n\n{raw_text}"

        items = []
        if isinstance(data, dict):
            items = data.get("data", {}).get("affected_items", [])
            if not items:
                items = data.get("affected_items", [])

        total = len(items)
        if not items:
            return "## Suricata Search Results\n\nNo matching Suricata alerts found.\n"

        lines = [f"## Suricata Search Results\n\n**Matches Found:** {total}\n"]
        lines.append("| # | Timestamp | Severity | Signature | Category | Src IP | Dest IP |")
        lines.append("|---|-----------|----------|-----------|----------|--------|---------|")
        for i, a in enumerate(items[:50], 1):
            ts = a.get("@timestamp", "")[:19]
            alert = a.get("alert", {})
            sev = self._suricata_severity_label(alert.get("severity", "?"))
            sig = (alert.get("signature", "N/A"))[:60]
            cat = (alert.get("category", "N/A"))[:30]
            src = a.get("src_ip", "N/A")
            dst = a.get("dest_ip", "N/A")
            lines.append(f"| {i} | {ts} | {sev} | {sig} | {cat} | {src} | {dst} |")

        return "\n".join(lines)

    def format_suricata_signatures_response(self, raw_text: str) -> str:
        """Format top Suricata signatures."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Top Suricata Signatures\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Top Suricata Signatures\n\n{raw_text}"

        sigs = data.get("top_signatures", [])
        total = data.get("total_alerts", 0)
        time_range = data.get("time_range", "24h")

        if not sigs:
            return f"## Top Suricata Signatures ({time_range})\n\nNo signatures found.\n"

        lines = [f"## Top Suricata Signatures ({time_range})\n\n**Total Alerts:** {total}\n"]
        lines.append("| # | Signature | Count | % of Total |")
        lines.append("|---|-----------|-------|------------|")
        for i, s in enumerate(sigs[:20], 1):
            sig = s.get("signature", "N/A")[:70]
            count = s.get("count", 0)
            pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
            lines.append(f"| {i} | {sig} | {count} | {pct} |")

        return "\n".join(lines)

    def format_suricata_health_response(self, raw_text: str) -> str:
        """Format Suricata cluster health."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Health\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Health\n\n{raw_text}"

        status = data.get("status", "unknown")
        status_icon = {"green": "GREEN", "yellow": "YELLOW", "red": "RED"}.get(status, "UNKNOWN")

        lines = ["## Suricata Elasticsearch Health\n"]
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Cluster Status | {status_icon} |")
        lines.append(f"| Cluster Name | {data.get('cluster_name', 'N/A')} |")
        lines.append(f"| Nodes | {data.get('number_of_nodes', 'N/A')} |")
        lines.append(f"| Active Shards | {data.get('active_shards', 'N/A')} |")
        if data.get("error"):
            lines.append(f"| Error | {data.get('error')} |")

        return "\n".join(lines)

    def format_suricata_category_breakdown_response(self, raw_text: str) -> str:
        """Format Suricata category breakdown."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Category Breakdown\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Category Breakdown\n\n{raw_text}"

        cats = data.get("top_categories", [])
        total = data.get("total_alerts", 0)
        time_range = data.get("time_range", "24h")

        lines = [f"## Suricata Alert Categories ({time_range})\n\n**Total Alerts:** {total}\n"]

        # Severity breakdown
        sev = data.get("severity_breakdown", {})
        if sev:
            lines.append("### Severity Distribution\n")
            lines.append("| Severity | Label | Count |")
            lines.append("|----------|-------|-------|")
            for level, count in sorted(sev.items(), key=lambda x: str(x[0])):
                label = self._suricata_severity_label(level)
                lines.append(f"| {level} | {label} | {count} |")
            lines.append("")

        # Categories
        if cats:
            lines.append("### Categories\n")
            lines.append("| # | Category | Count | % of Total |")
            lines.append("|---|----------|-------|------------|")
            for i, c in enumerate(cats[:20], 1):
                cat = c.get("category", "N/A")
                count = c.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {i} | {cat} | {count} | {pct} |")
        else:
            lines.append("No categories found.\n")

        return "\n".join(lines)

    # ========================================================================
    # SURICATA DEEP VISIBILITY FORMATTERS (HTTP, TLS, MITRE, JA3, Suspicious)
    # ========================================================================

    HTTP_STATUS_LABELS = {
        200: "OK", 201: "Created", 204: "No Content",
        301: "Moved Permanently", 302: "Found", 304: "Not Modified",
        400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
        404: "Not Found", 405: "Method Not Allowed", 429: "Too Many Requests",
        500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable",
    }

    def format_suricata_http_analysis_response(self, raw_text: str) -> str:
        """Format Suricata HTTP traffic analysis."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata HTTP Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata HTTP Analysis\n\n{raw_text}"

        total = data.get("total_events", 0)
        time_range = data.get("time_range", "24h")

        lines = [f"## Suricata HTTP Traffic Analysis ({time_range})\n\n**Total HTTP Events:** {total:,}\n"]

        # Top HTTP Methods
        methods = data.get("top_methods", [])
        if methods:
            lines.append("### HTTP Methods\n")
            lines.append("| Method | Count | % of Total |")
            lines.append("|--------|-------|------------|")
            for m in methods:
                method = m.get("method", "N/A")
                count = m.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {method} | {count:,} | {pct} |")
            lines.append("")

        # Top Status Codes
        status_codes = data.get("top_status_codes", [])
        if status_codes:
            lines.append("### HTTP Status Codes\n")
            lines.append("| Code | Meaning | Count | % of Total |")
            lines.append("|------|---------|-------|------------|")
            for s in status_codes:
                code = s.get("status_code", "?")
                meaning = self.HTTP_STATUS_LABELS.get(int(code) if str(code).isdigit() else 0, "Unknown")
                count = s.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {code} | {meaning} | {count:,} | {pct} |")
            lines.append("")

        # Top URLs
        urls = data.get("top_urls", [])
        if urls:
            lines.append("### Top URLs\n")
            lines.append("| # | URL | Count |")
            lines.append("|---|-----|-------|")
            for i, u in enumerate(urls[:15], 1):
                url = (u.get("url", "N/A"))[:80]
                count = u.get("count", 0)
                lines.append(f"| {i} | `{url}` | {count:,} |")
            lines.append("")

        # Top User Agents
        agents = data.get("top_user_agents", [])
        if agents:
            lines.append("### Top User Agents\n")
            lines.append("| # | User Agent | Count |")
            lines.append("|---|------------|-------|")
            for i, a in enumerate(agents[:10], 1):
                ua = (a.get("user_agent", "N/A"))[:70]
                count = a.get("count", 0)
                lines.append(f"| {i} | {ua} | {count:,} |")
            lines.append("")

        # Top Hostnames
        hostnames = data.get("top_hostnames", [])
        if hostnames:
            lines.append("### Top Target Hostnames\n")
            lines.append("| # | Hostname | Count |")
            lines.append("|---|----------|-------|")
            for i, h in enumerate(hostnames[:10], 1):
                hostname = h.get("hostname", "N/A")
                count = h.get("count", 0)
                lines.append(f"| {i} | {hostname} | {count:,} |")

        return "\n".join(lines)

    def format_suricata_http_search_response(self, raw_text: str) -> str:
        """Format Suricata HTTP event search results."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata HTTP Search Results\n\n{raw_text}"

        items = []
        if isinstance(data, dict):
            items = data.get("data", {}).get("affected_items", [])
            if not items:
                items = data.get("affected_items", [])

        total = len(items)
        if not items:
            return "## Suricata HTTP Search Results\n\nNo HTTP events found for the specified filters.\n"

        total_found = 0
        if isinstance(data, dict):
            total_found = data.get("data", {}).get("total_affected_items", total)

        lines = [f"## Suricata HTTP Search Results\n\n**Showing:** {total} of {total_found:,} events\n"]

        lines.append("| # | Timestamp | Method | URL | Status | User Agent | Src IP | Dest IP |")
        lines.append("|---|-----------|--------|-----|--------|------------|--------|---------|")
        for i, e in enumerate(items[:50], 1):
            ts = e.get("@timestamp", "")[:19]
            http = e.get("http", {})
            method = http.get("http_method", "N/A")
            url = (http.get("url", "N/A"))[:50]
            status = http.get("status", "N/A")
            ua = (http.get("http_user_agent", "N/A"))[:40]
            src = e.get("src_ip", "N/A")
            dst = e.get("dest_ip", "N/A")
            lines.append(f"| {i} | {ts} | {method} | `{url}` | {status} | {ua} | {src} | {dst} |")

        if total > 50:
            lines.append(f"\n*Showing 50 of {total_found:,} events.*\n")

        return "\n".join(lines)

    def format_suricata_tls_analysis_response(self, raw_text: str) -> str:
        """Format Suricata TLS analysis with JA3/JA3S/JA4 fingerprints."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata TLS Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata TLS Analysis\n\n{raw_text}"

        total = data.get("total_events", 0)
        time_range = data.get("time_range", "24h")

        lines = [f"## Suricata TLS/SSL Analysis ({time_range})\n\n**Total TLS Events:** {total:,}\n"]

        # TLS Version Distribution
        versions = data.get("tls_versions", [])
        if versions:
            lines.append("### TLS Version Distribution\n")
            lines.append("| Version | Count | % of Total | Status |")
            lines.append("|---------|-------|------------|--------|")
            legacy_versions = {"TLS 1.0", "TLS 1.1", "SSLv3", "SSLv2"}
            for v in versions:
                ver = v.get("version", "N/A")
                count = v.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                status = "LEGACY" if ver in legacy_versions else "Current"
                lines.append(f"| {ver} | {count:,} | {pct} | {status} |")
            lines.append("")

        # Top JA3 Hashes
        ja3 = data.get("top_ja3", [])
        if ja3:
            lines.append("### Top JA3 Client Fingerprints\n")
            lines.append("| # | JA3 Hash | Count | % of Total |")
            lines.append("|---|----------|-------|------------|")
            for i, j in enumerate(ja3[:15], 1):
                hash_val = j.get("hash", "N/A")[:16]
                count = j.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {i} | `{hash_val}...` | {count:,} | {pct} |")
            lines.append("")

        # Top JA3S Hashes
        ja3s = data.get("top_ja3s", [])
        if ja3s:
            lines.append("### Top JA3S Server Fingerprints\n")
            lines.append("| # | JA3S Hash | Count |")
            lines.append("|---|-----------|-------|")
            for i, j in enumerate(ja3s[:10], 1):
                hash_val = j.get("hash", "N/A")[:16]
                count = j.get("count", 0)
                lines.append(f"| {i} | `{hash_val}...` | {count:,} |")
            lines.append("")

        # Top JA4 Fingerprints
        ja4 = data.get("top_ja4", [])
        if ja4:
            lines.append("### Top JA4 Fingerprints\n")
            lines.append("| # | JA4 Fingerprint | Count |")
            lines.append("|---|-----------------|-------|")
            for i, j in enumerate(ja4[:10], 1):
                fp = j.get("fingerprint", "N/A")
                count = j.get("count", 0)
                lines.append(f"| {i} | `{fp}` | {count:,} |")
            lines.append("")

        # Top Destination Services
        services = data.get("top_dest_services", [])
        if services:
            lines.append("### Top TLS Destination Services\n")
            lines.append("| # | Service | Count |")
            lines.append("|---|---------|-------|")
            for i, s in enumerate(services[:10], 1):
                svc = s.get("service", "N/A")
                count = s.get("count", 0)
                lines.append(f"| {i} | {svc} | {count:,} |")

        return "\n".join(lines)

    def format_suricata_mitre_mapping_response(self, raw_text: str) -> str:
        """Format MITRE ATT&CK mapping from Suricata alerts."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata MITRE ATT&CK Mapping\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata MITRE ATT&CK Mapping\n\n{raw_text}"

        total = data.get("total_mitre_alerts", 0)
        time_range = data.get("time_range", "24h")
        tactics = data.get("tactics", [])
        techniques = data.get("techniques", [])
        tactic_ids = data.get("tactic_ids", [])
        technique_ids = data.get("technique_ids", [])

        lines = [f"## Suricata MITRE ATT&CK Mapping ({time_range})\n"]
        lines.append(f"**Total MITRE-Tagged Alerts:** {total:,}")
        lines.append(f"**Unique Tactics:** {len(tactics)} | **Unique Techniques:** {len(techniques)}\n")

        # Severity breakdown
        sev = data.get("severity_breakdown", {})
        if sev:
            lines.append("### Alert Severity Distribution\n")
            lines.append("| Severity | Label | Count |")
            lines.append("|----------|-------|-------|")
            for level, count in sorted(sev.items(), key=lambda x: str(x[0])):
                label = self._suricata_severity_label(level)
                lines.append(f"| {level} | {label} | {count:,} |")
            lines.append("")

        # Tactics
        if tactics:
            # Build ID lookup
            id_lookup = {t.get("tactic_id"): t.get("count") for t in tactic_ids} if tactic_ids else {}
            lines.append("### MITRE Tactics\n")
            lines.append("| # | Tactic | Count | % of Total |")
            lines.append("|---|--------|-------|------------|")
            for i, t in enumerate(sorted(tactics, key=lambda x: x.get("count", 0), reverse=True), 1):
                name = t.get("tactic_name", "N/A")
                count = t.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {i} | {name} | {count:,} | {pct} |")
            lines.append("")

        # Techniques
        if techniques:
            lines.append("### MITRE Techniques\n")
            lines.append("| # | Technique | Count | % of Total |")
            lines.append("|---|-----------|-------|------------|")
            for i, t in enumerate(sorted(techniques, key=lambda x: x.get("count", 0), reverse=True)[:20], 1):
                name = t.get("technique_name", "N/A")
                count = t.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {i} | {name} | {count:,} | {pct} |")
            lines.append("")

        # Top signatures with MITRE tags
        sigs = data.get("top_signatures", [])
        if sigs:
            lines.append("### Top MITRE-Tagged Signatures\n")
            lines.append("| # | Signature | Count |")
            lines.append("|---|-----------|-------|")
            for i, s in enumerate(sigs[:15], 1):
                sig = (s.get("signature", "N/A"))[:70]
                count = s.get("count", 0)
                lines.append(f"| {i} | {sig} | {count:,} |")

        return "\n".join(lines)

    def format_suricata_ja3_response(self, raw_text: str) -> str:
        """Format JA3 fingerprint deep analysis."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## JA3 Fingerprint Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## JA3 Fingerprint Analysis\n\n{raw_text}"

        total = data.get("total_tls_events", 0)
        time_range = data.get("time_range", "24h")
        fingerprints = data.get("fingerprints", [])

        lines = [f"## JA3 Fingerprint Analysis ({time_range})\n\n**Total TLS Events:** {total:,}\n"]

        if not fingerprints:
            lines.append("No JA3 fingerprints found.\n")
            return "\n".join(lines)

        lines.append(f"**Unique JA3 Fingerprints:** {len(fingerprints)}\n")

        for i, fp in enumerate(fingerprints[:15], 1):
            ja3 = fp.get("ja3_hash", "N/A")
            count = fp.get("count", 0)
            pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"

            lines.append(f"### {i}. `{ja3}`")
            lines.append(f"**Hits:** {count:,} ({pct} of TLS traffic)\n")

            # Source IPs
            src_ips = fp.get("top_src_ips", [])
            if src_ips:
                lines.append("**Source IPs:** " + ", ".join(
                    f"{s.get('ip')} ({s.get('count')})" for s in src_ips
                ))

            # Dest IPs
            dest_ips = fp.get("top_dest_ips", [])
            if dest_ips:
                lines.append("**Dest IPs:** " + ", ".join(
                    f"{d.get('ip')} ({d.get('count')})" for d in dest_ips
                ))

            # Services
            services = fp.get("top_services", [])
            if services:
                lines.append("**Services:** " + ", ".join(
                    f"{s.get('service')} ({s.get('count')})" for s in services
                ))

            # TLS versions
            versions = fp.get("tls_versions", [])
            if versions:
                lines.append("**TLS Versions:** " + ", ".join(
                    f"{v.get('version')} ({v.get('count')})" for v in versions
                ))

            lines.append("")

        return "\n".join(lines)

    def format_suricata_suspicious_activity_response(self, raw_text: str) -> str:
        """Format suspicious activity detection results."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Suspicious Activity\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Suspicious Activity\n\n{raw_text}"

        time_range = data.get("time_range", "24h")
        sus_agents = data.get("suspicious_user_agents", [])
        legacy_tls = data.get("legacy_tls", [])
        unusual_methods = data.get("unusual_http_methods", [])
        http_errors = data.get("http_errors", [])

        total_findings = len(sus_agents) + len(legacy_tls) + len(unusual_methods) + len(http_errors)

        lines = [f"## Suricata Suspicious Activity Report ({time_range})\n"]
        lines.append(f"**Finding Categories:** {total_findings}\n")

        # Suspicious User Agents
        if sus_agents:
            lines.append("### Suspicious User Agents\n")
            lines.append("| # | User Agent | Count | Source IPs |")
            lines.append("|---|------------|-------|------------|")
            for i, a in enumerate(sus_agents[:15], 1):
                ua = (a.get("user_agent", "N/A"))[:60]
                count = a.get("count", 0)
                ips = ", ".join(a.get("src_ips", [])[:3])
                lines.append(f"| {i} | {ua} | {count:,} | {ips} |")
            lines.append("")
        else:
            lines.append("### Suspicious User Agents\n\nNo scanner-like user agents detected.\n")

        # Legacy TLS
        if legacy_tls:
            lines.append("### Legacy TLS Connections\n")
            lines.append("| Version | Count | Source IPs | Dest IPs |")
            lines.append("|---------|-------|------------|----------|")
            for t in legacy_tls:
                ver = t.get("version", "N/A")
                count = t.get("count", 0)
                src = ", ".join(t.get("src_ips", [])[:3])
                dst = ", ".join(t.get("dest_ips", [])[:3])
                lines.append(f"| {ver} | {count:,} | {src} | {dst} |")
            lines.append("")
        else:
            lines.append("### Legacy TLS Connections\n\nNo legacy TLS versions detected.\n")

        # Unusual HTTP Methods
        if unusual_methods:
            lines.append("### Unusual HTTP Methods\n")
            lines.append("| Method | Count | Top URLs | Source IPs |")
            lines.append("|--------|-------|----------|------------|")
            for m in unusual_methods:
                method = m.get("method", "N/A")
                count = m.get("count", 0)
                urls = ", ".join((u[:40] for u in m.get("top_urls", [])[:2]))
                ips = ", ".join(m.get("src_ips", [])[:3])
                lines.append(f"| {method} | {count:,} | {urls} | {ips} |")
            lines.append("")
        else:
            lines.append("### Unusual HTTP Methods\n\nNo unusual HTTP methods detected.\n")
        # HTTP Errors
        if http_errors:
            lines.append("### HTTP Error Responses (4xx/5xx)\n")
            lines.append("| Status Code | Meaning | Count | Source IPs |")
            lines.append("|-------------|---------|-------|------------|")
            for e in http_errors[:15]:
                code = e.get("status_code", "?")
                meaning = self.HTTP_STATUS_LABELS.get(int(code) if str(code).isdigit() else 0, "Unknown")
                count = e.get("count", 0)
                ips = ", ".join(e.get("src_ips", [])[:3])
                lines.append(f"| {code} | {meaning} | {count:,} | {ips} |")
        else:
            lines.append("### HTTP Error Responses\n\nNo HTTP errors detected.\n")

        return "\n".join(lines)

    def format_suricata_traffic_overview_response(self, raw_text: str) -> str:
        """Format Suricata traffic overview."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Suricata Traffic Overview\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Suricata Traffic Overview\n\n{raw_text}"

        total = data.get("total_events", 0)
        time_range = data.get("time_range", "24h")

        lines = [f"## Suricata Traffic Overview ({time_range})\n\n**Total Events:** {total:,}\n"]

        # Event Type Distribution
        event_types = data.get("event_types", [])
        if event_types:
            lines.append("### Event Type Distribution\n")
            lines.append("| Type | Count | % of Total |")
            lines.append("|------|-------|------------|")
            for et in event_types:
                etype = et.get("type", "N/A")
                count = et.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {etype} | {count:,} | {pct} |")
            lines.append("")

        # Protocol Distribution
        protocols = data.get("protocols", [])
        if protocols:
            lines.append("### Protocol Distribution\n")
            lines.append("| Protocol | Count | % of Total |")
            lines.append("|----------|-------|------------|")
            for p in protocols:
                proto = p.get("protocol", "N/A")
                count = p.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {proto} | {count:,} | {pct} |")
            lines.append("")

        # Top Services
        services = data.get("services", [])
        if services:
            lines.append("### Top Services\n")
            lines.append("| # | Service | Count |")
            lines.append("|---|---------|-------|")
            for i, s in enumerate(services[:15], 1):
                svc = s.get("service", "N/A")
                count = s.get("count", 0)
                lines.append(f"| {i} | {svc} | {count:,} |")
            lines.append("")

        # Traffic Locality
        locality = data.get("traffic_locality", [])
        if locality:
            lines.append("### Traffic Locality\n")
            lines.append("| Locality | Count | % of Total |")
            lines.append("|----------|-------|------------|")
            for loc in locality:
                name = loc.get("locality", "N/A")
                count = loc.get("count", 0)
                pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"
                lines.append(f"| {name} | {count:,} | {pct} |")
            lines.append("")

        # Interfaces
        interfaces = data.get("interfaces", [])
        if interfaces:
            lines.append("### Network Interfaces\n")
            lines.append("| Interface | Count |")
            lines.append("|-----------|-------|")
            for iface in interfaces:
                name = iface.get("interface", "N/A")
                count = iface.get("count", 0)
                lines.append(f"| {name} | {count:,} |")

        return "\n".join(lines)


    # ==================================================================
    # PALLAS 3.2: UNIVERSAL QUERY COVERAGE FORMATTERS
    # ==================================================================

    def format_ip_investigation_response(self, data, summary=None, **kwargs):
        """Format cross-source IP investigation results."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "IP Investigation")

        target_ip = data.get("target_ip", "Unknown")
        lines = [f"## IP Investigation: {target_ip}\n"]

        # Agent match
        agent = data.get("matching_agent")
        if agent:
            lines.append(f"**Known Agent:** {agent.get('name', 'N/A')} (ID: {agent.get('id', 'N/A')}, Status: {agent.get('status', 'N/A')})\n")
        else:
            lines.append("**Known Agent:** No matching agent found\n")

        # Wazuh alerts
        wazuh = data.get("wazuh_alerts", {})
        wazuh_total = wazuh.get("total", 0)
        lines.append(f"### Wazuh Alerts ({wazuh_total:,} total)\n")
        top_rules = wazuh.get("top_rules", [])
        if top_rules:
            lines.append("| Rule | Count |")
            lines.append("|------|-------|")
            for r in top_rules[:10]:
                lines.append(f"| {r.get('rule', 'N/A')} | {r.get('count', 0):,} |")
            lines.append("")
        elif wazuh_total == 0:
            lines.append("No Wazuh alerts found for this IP.\n")

        # Suricata alerts
        suricata = data.get("suricata_alerts", {})
        suricata_total = suricata.get("total", 0)
        lines.append(f"### Suricata IDS Alerts ({suricata_total:,} total)\n")
        top_sigs = suricata.get("top_signatures", [])
        if top_sigs:
            lines.append("| Signature | Count |")
            lines.append("|-----------|-------|")
            for s in top_sigs[:10]:
                lines.append(f"| {s.get('signature', 'N/A')} | {s.get('count', 0):,} |")
            lines.append("")
        elif suricata_total == 0:
            lines.append("No Suricata alerts found for this IP.\n")

        # HTTP activity
        http = data.get("http_activity", {})
        http_total = http.get("total_events", 0)
        if http_total > 0:
            lines.append(f"### HTTP Activity ({http_total:,} events)\n")
            urls = http.get("top_urls", [])
            if urls:
                lines.append("**Top URLs:**")
                for u in urls[:5]:
                    url = u.get("url", "N/A") if isinstance(u, dict) else str(u)
                    count = u.get("count", "") if isinstance(u, dict) else ""
                    lines.append(f"- `{url}` ({count})" if count else f"- `{url}`")
                lines.append("")
            uas = http.get("top_user_agents", [])
            if uas:
                lines.append("**Top User Agents:**")
                for ua in uas[:5]:
                    agent_str = ua.get("user_agent", "N/A") if isinstance(ua, dict) else str(ua)
                    count = ua.get("count", "") if isinstance(ua, dict) else ""
                    lines.append(f"- `{agent_str}` ({count})" if count else f"- `{agent_str}`")
                lines.append("")

        return "\n".join(lines)

    def format_full_agent_investigation_response(self, data, summary=None, **kwargs):
        """Format complete agent security investigation."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Agent Investigation")

        agent = data.get("agent", {}) or {}
        queried_id = (
            agent.get("id")
            or data.get("queried_agent_id")
            or (summary or {}).get("agent_id")
            or kwargs.get("agent_id")
            or ""
        )

        # Only declare "Agent Not Found" when EVERY data source came back empty.
        # If secondary tools (vulns/ports/processes/alerts/fim) returned data for
        # this agent, render the report even if the primary agent profile lookup
        # failed — partial data is better than a misleading "not found".
        any_data_present = (
            (data.get("vulnerabilities", {}) or {}).get("summary", {}).get("total", 0) > 0
            or (data.get("open_ports", {}) or {}).get("total", 0) > 0
            or (data.get("processes", {}) or {}).get("total", 0) > 0
            or (data.get("alerts", {}) or {}).get("total", 0) > 0
            or (data.get("fim_events", {}) or {}).get("total", 0) > 0
        )

        if not agent.get("id") and not any_data_present:
            queried_id_str = queried_id or "the requested agent"
            return (
                f"## Agent {queried_id_str} Not Found\n\n"
                f"The Wazuh manager has no record of this agent, so a full investigation cannot be performed.\n\n"
                f"**Possible causes:**\n"
                f"- The agent ID is incorrect (typo or wrong number)\n"
                f"- The agent has been removed or decommissioned\n"
                f"- The agent has not registered with the manager yet\n\n"
                f"**Next steps:**\n"
                f"- Run `show active agents` to see registered agent IDs\n"
                f"- Run `show disconnected agents` to see agents that registered but are offline\n"
                f"- Try `full investigation for agent <valid-id>`"
            )

        # Profile lookup failed but secondary data exists — synthesize a placeholder
        # so the report header still has the queried id, and add a note for the analyst.
        partial_profile_note = ""
        if not agent.get("id") and any_data_present:
            agent = dict(agent)
            agent["id"] = queried_id
            partial_profile_note = (
                "\n> ⚠️ **Note:** Agent profile lookup returned no record, but secondary "
                "data sources (vulnerabilities / ports / alerts / FIM) found activity for "
                f"agent `{queried_id}`. Rendering with available data.\n"
            )

        risk_score = data.get("risk_score", 0)
        risk_label = "CRITICAL" if risk_score >= 60 else "HIGH" if risk_score >= 40 else "MEDIUM" if risk_score >= 20 else "LOW"

        lines = [f"## Full Agent Investigation: {agent.get('name', 'N/A')} (ID: {agent.get('id', 'N/A')})\n"]
        if partial_profile_note:
            lines.append(partial_profile_note)
        lines.append(f"**Risk Score:** {risk_score}/100 ({risk_label})\n")

        # Agent profile
        lines.append("### Agent Profile\n")
        lines.append(f"- **Status:** {agent.get('status', 'N/A')}")
        lines.append(f"- **IP:** {agent.get('ip', 'N/A')}")
        lines.append(f"- **OS:** {agent.get('os', 'N/A')}")
        lines.append(f"- **Version:** {agent.get('version', 'N/A')}")
        lines.append(f"- **Last Keep Alive:** {agent.get('last_keep_alive', 'N/A')}")
        lines.append("")

        # Vulnerabilities
        vuln_data = data.get("vulnerabilities", {})
        vuln_sum = vuln_data.get("summary", {})
        lines.append(f"### Vulnerabilities ({vuln_sum.get('total', 0)} total)\n")
        if vuln_sum.get("total", 0) > 0:
            lines.append(f"Critical: {vuln_sum.get('critical', 0)} | High: {vuln_sum.get('high', 0)} | Medium: {vuln_sum.get('medium', 0)} | Low: {vuln_sum.get('low', 0)}\n")
            top_vulns = vuln_data.get("top", [])
            if top_vulns:
                lines.append("| CVE | Severity | CVSS | Package |")
                lines.append("|-----|----------|------|---------|")
                for v in top_vulns[:10]:
                    lines.append(f"| {v.get('cve_id', 'N/A')} | {v.get('severity', 'N/A')} | {v.get('cvss_score', 'N/A')} | {v.get('package', 'N/A')} |")
                lines.append("")

        # Open ports
        port_data = data.get("open_ports", {})
        lines.append(f"### Open Ports ({port_data.get('total', 0)} total)\n")
        suspicious = port_data.get("suspicious", [])
        if suspicious:
            lines.append("**Suspicious Ports:**")
            for p in suspicious:
                lines.append(f"- Port {p.get('port')} ({p.get('protocol', 'tcp')}) — Process: {p.get('process', 'N/A')}")
            lines.append("")
        port_items = port_data.get("items", [])
        if port_items:
            lines.append("| Port | Protocol | State | Process |")
            lines.append("|------|----------|-------|---------|")
            for p in port_items[:15]:
                lines.append(f"| {p.get('port', 'N/A')} | {p.get('protocol', 'N/A')} | {p.get('state', 'N/A')} | {p.get('process', 'N/A')} |")
            lines.append("")

        # Alerts
        alert_data = data.get("alerts", {})
        lines.append(f"### Recent Alerts ({alert_data.get('total', 0)} total)\n")
        alert_items = alert_data.get("items", [])
        if alert_items:
            lines.append("| Rule ID | Description | Level | Timestamp |")
            lines.append("|---------|-------------|-------|-----------|")
            for a in alert_items[:10]:
                lines.append(f"| {a.get('rule_id', 'N/A')} | {str(a.get('description') or 'N/A')[:60]} | {a.get('level', 'N/A')} | {str(a.get('timestamp', 'N/A'))[:19]} |")
            lines.append("")

        # FIM events
        fim_data = data.get("fim_events", {})
        if fim_data.get("total", 0) > 0:
            lines.append(f"### File Integrity Events ({fim_data.get('total', 0)} total)\n")
            for f in fim_data.get("items", [])[:10]:
                lines.append(f"- [{f.get('event', 'N/A')}] `{f.get('path', 'N/A')}` ({str(f.get('timestamp', 'N/A'))[:19]})")
            lines.append("")

        # Processes
        proc_data = data.get("processes", {})
        if proc_data.get("total", 0) > 0:
            lines.append(f"### Running Processes ({proc_data.get('total', 0)} total)\n")
            lines.append("| PID | Name | User |")
            lines.append("|-----|------|------|")
            for p in proc_data.get("items", [])[:15]:
                lines.append(f"| {p.get('pid', 'N/A')} | {p.get('name', 'N/A')} | {p.get('user', 'N/A')} |")
            lines.append("")

        return "\n".join(lines)

    def format_attack_chain_response(self, data, summary=None, **kwargs):
        """Format MITRE ATT&CK kill chain analysis."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Attack Chain Analysis")

        s = summary or {}
        lines = ["## MITRE ATT&CK Kill Chain Analysis\n"]
        lines.append(f"**Coverage:** {s.get('tactics_observed', 0)}/{s.get('tactics_total', 14)} tactics observed ({s.get('coverage_pct', 0)}%)")
        lines.append(f"**Unique Techniques:** {s.get('unique_techniques', 0)}\n")

        # Kill chain matrix
        chain = data.get("kill_chain", [])
        if chain:
            lines.append("### Kill Chain Progression\n")
            lines.append("| # | Tactic ID | Tactic | Status | Alerts | Source |")
            lines.append("|---|-----------|--------|--------|--------|--------|")
            for i, stage in enumerate(chain):
                status = "OBSERVED" if stage.get("observed") else "---"
                lines.append(f"| {i+1} | {stage.get('tactic_id', 'N/A')} | {stage.get('tactic_name', 'N/A')} | {status} | {stage.get('alert_count', 0):,} | {stage.get('source', 'N/A')} |")
            lines.append("")

        # Observed techniques
        techniques = data.get("observed_techniques", [])
        if techniques:
            lines.append("### Top Observed Techniques\n")
            lines.append("| Technique | Alert Count | Source |")
            lines.append("|-----------|------------|--------|")
            for t in techniques[:15]:
                lines.append(f"| {t.get('technique', 'N/A')} | {t.get('count', 0):,} | {t.get('source', 'N/A')} |")
            lines.append("")

        # Gaps
        gaps = [c for c in chain if not c.get("observed")]
        if gaps:
            lines.append("### Detection Gaps\n")
            lines.append("The following kill chain stages have **no observed detections**:\n")
            for g in gaps:
                lines.append(f"- **{g.get('tactic_name', 'N/A')}** ({g.get('tactic_id', '')})")
            lines.append("")

        return "\n".join(lines)

    def format_unified_threat_summary_response(self, data, summary=None, **kwargs):
        """Format prioritized threat summary."""
        if isinstance(data, list):
            threats = data
        elif isinstance(data, dict) and "correlated_data" in data:
            threats = data["correlated_data"]
        else:
            threats = []
        if not isinstance(threats, list):
            return self._safe_fallback(data, "Threat Summary")

        s = summary or {}
        lines = ["## Prioritized Threat Summary\n"]
        lines.append(f"**Threats Identified:** {s.get('total_threats_identified', len(threats))}")
        lines.append(f"Critical: {s.get('critical_items', 0)} | High: {s.get('high_items', 0)} | Medium: {s.get('medium_items', 0)}\n")

        if threats:
            lines.append("| Rank | Threat | Severity | Source | Count | Recommended Action |")
            lines.append("|------|--------|----------|--------|-------|--------------------|")
            for t in threats[:10]:
                lines.append(f"| {t.get('rank', '-')} | {t.get('threat', 'N/A')} | {t.get('severity', 'N/A')} | {t.get('source', 'N/A')} | {t.get('count', 0):,} | {t.get('action', 'N/A')} |")
            lines.append("")
        else:
            lines.append("No significant threats identified in the current time window.\n")

        return "\n".join(lines)

    def format_vulnerability_exploit_response(self, data, summary=None, **kwargs):
        """Format vulnerability exploit correlation results."""
        if isinstance(data, list):
            vulns = data
        elif isinstance(data, dict) and "correlated_data" in data:
            vulns = data["correlated_data"]
        else:
            vulns = []
        if not isinstance(vulns, list):
            return self._safe_fallback(data, "Vulnerability Exploit Correlation")

        s = summary or {}
        lines = ["## Vulnerability Exploit Correlation\n"]
        lines.append(f"**Analyzed:** {s.get('total_vulnerabilities_analyzed', 0)} vulnerabilities against {s.get('total_suricata_signatures', 0)} IDS signatures")
        lines.append(f"**Actively Targeted:** {s.get('actively_targeted', 0)} | **Potentially Targeted:** {s.get('potentially_targeted', 0)} | **Not Targeted:** {s.get('not_targeted', 0)}\n")

        active = [v for v in vulns if v.get("targeting_status") == "ACTIVE"]
        if active:
            lines.append("### ACTIVELY TARGETED Vulnerabilities\n")
            lines.append("| CVE | Severity | CVSS | Package | Agent | Related IDS Signatures |")
            lines.append("|-----|----------|------|---------|-------|----------------------|")
            for v in active[:15]:
                sigs = ", ".join(sg.get("signature", "")[:40] for sg in v.get("related_signatures", []))
                lines.append(f"| {v.get('cve_id', 'N/A')} | {v.get('severity', 'N/A')} | {v.get('cvss_score', 'N/A')} | {v.get('package', 'N/A')} | {v.get('agent_id', 'N/A')} | {sigs or 'N/A'} |")
            lines.append("")

        potential = [v for v in vulns if v.get("targeting_status") == "POTENTIAL"]
        if potential:
            lines.append("### POTENTIALLY TARGETED Vulnerabilities\n")
            lines.append("| CVE | Severity | CVSS | Package | Agent |")
            lines.append("|-----|----------|------|---------|-------|")
            for v in potential[:10]:
                lines.append(f"| {v.get('cve_id', 'N/A')} | {v.get('severity', 'N/A')} | {v.get('cvss_score', 'N/A')} | {v.get('package', 'N/A')} | {v.get('agent_id', 'N/A')} |")
            lines.append("")

        return "\n".join(lines)

    def format_detection_coverage_gap_response(self, data, summary=None, **kwargs):
        """Format MITRE ATT&CK detection coverage gap analysis."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Detection Coverage Gap")

        s = summary or {}
        lines = ["## MITRE ATT&CK Detection Coverage Analysis\n"]
        lines.append(f"**Tactic Coverage:** {s.get('tactics_covered', 0)}/{s.get('tactics_total', 14)} ({s.get('coverage_pct', 0)}%)")
        lines.append(f"**Combined Techniques:** {s.get('combined_unique_techniques', 0)} (Wazuh: {s.get('wazuh_unique_techniques', 0)}, Suricata: {s.get('suricata_unique_techniques', 0)})")
        lines.append(f"**Detection Gaps:** {s.get('gap_count', 0)} tactics with no coverage\n")

        matrix = data.get("coverage_matrix", [])
        if matrix:
            lines.append("### Tactic Coverage Matrix\n")
            lines.append("| Tactic ID | Tactic | Wazuh | Suricata | Status |")
            lines.append("|-----------|--------|-------|----------|--------|")
            for c in matrix:
                w = "Yes" if c.get("wazuh_covered") else "---"
                s_cov = "Yes" if c.get("suricata_covered") else "---"
                status = c.get("status", "N/A")
                lines.append(f"| {c.get('tactic_id', 'N/A')} | {c.get('tactic_name', 'N/A')} | {w} | {s_cov} | {status} |")
            lines.append("")

        gaps = data.get("gaps", [])
        if gaps:
            lines.append("### Critical Detection Gaps\n")
            lines.append("The following tactics have **NO detection coverage** from either platform:\n")
            for g in gaps:
                lines.append(f"- **{g.get('tactic_name', 'N/A')}** ({g.get('tactic_id', '')})")
            lines.append("")

        return "\n".join(lines)

    def format_fim_alert_correlation_response(self, data, summary=None, **kwargs):
        """Format FIM-alert correlation timeline."""
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "correlated_data" in data:
            events = data["correlated_data"]
        else:
            events = []
        if not isinstance(events, list):
            return self._safe_fallback(data, "FIM Alert Correlation")

        s = summary or {}
        lines = ["## File Integrity — Alert Correlation\n"]
        lines.append(f"**FIM Events:** {s.get('total_fim_events', 0)} | **With Related Alerts:** {s.get('events_with_alerts', 0)}")
        lines.append(f"**High Risk:** {s.get('high_risk_changes', 0)} | **System Binary Changes:** {s.get('system_binary_changes', 0)}\n")

        if events:
            lines.append("| Risk | File Path | Change | Agent | Related Alerts | Timestamp |")
            lines.append("|------|-----------|--------|-------|----------------|-----------|")
            for e in events[:30]:
                risk = e.get("risk_level", "LOW")
                path = e.get("file_path", "N/A")
                if len(path) > 50:
                    path = "..." + path[-47:]
                change = e.get("change_type", "N/A")
                agent_name = e.get("agent_name", "N/A")
                alert_count = e.get("related_alerts_count", 0)
                ts = str(e.get("timestamp", "N/A"))[:19]
                lines.append(f"| {risk} | `{path}` | {change} | {agent_name} | {alert_count} | {ts} |")
            lines.append("")

            high_risk = [e for e in events if e.get("risk_level") == "HIGH"]
            if high_risk:
                lines.append("### High-Risk File Changes\n")
                for e in high_risk[:5]:
                    lines.append(f"**`{e.get('file_path', 'N/A')}`** — {e.get('change_type', 'N/A')} (Agent: {e.get('agent_name', 'N/A')})")
                    for a in e.get("related_alerts", [])[:3]:
                        lines.append(f"  - Alert {a.get('rule_id', 'N/A')}: {a.get('description', 'N/A')} (Level {a.get('level', 'N/A')})")
                    lines.append("")

        return "\n".join(lines)

    def format_port_exposure_risk_response(self, data, summary=None, **kwargs):
        """Format port exposure risk assessment."""
        if isinstance(data, list):
            ports = data
        elif isinstance(data, dict) and "correlated_data" in data:
            ports = data["correlated_data"]
        else:
            ports = []
        if not isinstance(ports, list):
            return self._safe_fallback(data, "Port Exposure Risk")

        s = summary or {}
        lines = ["## Port Exposure Risk Assessment\n"]
        lines.append(f"**Open Ports:** {s.get('total_open_ports', 0)} | **Critical Risk:** {s.get('critical_risk', 0)} | **High Risk:** {s.get('high_risk', 0)}")
        lines.append(f"**Ports with IDS Alerts:** {s.get('ports_with_ids_hits', 0)} | **Ports with Known CVEs:** {s.get('ports_with_cves', 0)}\n")

        if ports:
            lines.append("| Port | Service | Risk | IDS Alerts | CVEs | Risk Factors |")
            lines.append("|------|---------|------|-----------|------|--------------|")
            for p in ports[:25]:
                cves = ", ".join(p.get("related_cves", [])[:2]) or "None"
                factors = "; ".join(p.get("risk_factors", [])[:2]) or "None"
                lines.append(f"| {p.get('port', 'N/A')} | {p.get('service', 'N/A')} | {p.get('risk_level', 'LOW')} | {p.get('ids_alerts', 0)} | {cves} | {factors} |")
            lines.append("")

        return "\n".join(lines)

    def format_top_risk_agents_response(self, data, summary=None, **kwargs):
        """Format composite agent risk ranking."""
        if isinstance(data, list):
            agents = data
        elif isinstance(data, dict) and "correlated_data" in data:
            agents = data["correlated_data"]
        else:
            agents = []
        if not isinstance(agents, list):
            return self._safe_fallback(data, "Agent Risk Ranking")

        s = summary or {}
        lines = ["## Agent Risk Ranking\n"]
        lines.append(f"**Agents Analyzed:** {s.get('total_agents_analyzed', 0)}")
        lines.append(f"**Critical Risk:** {s.get('agents_with_critical_risk', 0)} | **High Risk:** {s.get('agents_with_high_risk', 0)}")
        if s.get("highest_risk_agent"):
            lines.append(f"**Highest Risk:** {s.get('highest_risk_agent', 'N/A')} (Score: {s.get('highest_risk_score', 0)})\n")

        if agents:
            lines.append("| Rank | Agent | ID | IP | Status | Risk Score | Vulns | Alerts | IDS Hits | Top Risk Factor |")
            lines.append("|------|-------|----|----|--------|------------|-------|--------|----------|-----------------|")
            for a in agents[:20]:
                top_factor = a.get("top_risk_factors", ["None"])[0] if a.get("top_risk_factors") else "None"
                lines.append(f"| {a.get('rank', '-')} | {a.get('agent_name', 'N/A')} | {a.get('agent_id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('status', 'N/A')} | {a.get('risk_score', 0)} | {a.get('vuln_count', 0)} | {a.get('alert_count', 0)} | {a.get('ids_hits', 0)} | {top_factor} |")
            lines.append("")

        return "\n".join(lines)

    def format_cross_platform_ip_response(self, data, summary=None, **kwargs):
        """Format cross-platform IP correlation results."""
        if isinstance(data, list):
            ips = data
        elif isinstance(data, dict) and "correlated_data" in data:
            ips = data["correlated_data"]
        else:
            ips = []
        if not isinstance(ips, list):
            return self._safe_fallback(data, "Cross-Platform IP Correlation")

        s = summary or {}
        lines = ["## Cross-Platform IP Correlation\n"]
        lines.append(f"**Unique IPs:** {s.get('total_unique_ips', 0)} | **In Both Platforms:** {s.get('ips_in_both_platforms', 0)}")
        lines.append(f"**Suricata Only:** {s.get('suricata_only', 0)} | **Wazuh Only:** {s.get('wazuh_only', 0)} | **Matching Agents:** {s.get('ips_matching_agents', 0)}\n")

        if ips:
            lines.append("| IP | Suricata Hits | Wazuh Hits | Total | Detected By | Agent? | Agent Name |")
            lines.append("|----|--------------|------------|-------|-------------|--------|------------|")
            for ip in ips[:30]:
                is_agent = "Yes" if ip.get("is_agent") else "No"
                agent_name = ip.get("agent_name", "-") or "-"
                lines.append(f"| {ip.get('ip', 'N/A')} | {ip.get('suricata_hits', 0):,} | {ip.get('wazuh_hits', 0):,} | {ip.get('total_hits', 0):,} | {ip.get('detected_by', 'N/A')} | {is_agent} | {agent_name} |")
            lines.append("")

        return "\n".join(lines)

    def format_unified_scan_detection_response(self, data, summary=None, **kwargs):
        """Format unified scanning detection results."""
        if isinstance(data, list):
            scanners = data
        elif isinstance(data, dict) and "correlated_data" in data:
            scanners = data["correlated_data"]
        else:
            scanners = []
        if not isinstance(scanners, list):
            return self._safe_fallback(data, "Scanning Detection")

        s = summary or {}
        lines = ["## Unified Scanning Detection\n"]
        lines.append(f"**Scanner IPs Detected:** {s.get('total_scanner_ips', 0)} | **Multi-Platform Detections:** {s.get('multi_platform_detections', 0)}")
        lines.append(f"**Suricata Only:** {s.get('suricata_only', 0)} | **Wazuh Only:** {s.get('wazuh_only', 0)}\n")

        if scanners:
            lines.append("| Scanner IP | Detected By | Details |")
            lines.append("|-----------|-------------|---------|")
            for sc in scanners[:25]:
                details = "; ".join(sc.get("details", [])[:3]) or "N/A"
                if len(details) > 80:
                    details = details[:77] + "..."
                lines.append(f"| {sc.get('ip', 'N/A')} | {sc.get('detected_by', 'N/A')} | {details} |")
            lines.append("")

        return "\n".join(lines)

    def format_alert_enrichment_response(self, data, summary=None, **kwargs):
        """Format enriched alert context."""
        if isinstance(data, list):
            alerts = data
        elif isinstance(data, dict) and "correlated_data" in data:
            alerts = data["correlated_data"]
        else:
            alerts = []
        if not isinstance(alerts, list):
            return self._safe_fallback(data, "Alert Enrichment")

        s = summary or {}
        lines = ["## Alert Context Enrichment\n"]
        lines.append(f"**Alerts Enriched:** {s.get('total_alerts_enriched', 0)} | **With MITRE:** {s.get('alerts_with_mitre', 0)} | **With IDS Context:** {s.get('alerts_with_ids_context', 0)}\n")

        for i, entry in enumerate(alerts[:10]):
            alert = entry.get("alert", {})
            agent = entry.get("agent", {})
            vuln_status = entry.get("vulnerability_status", {})
            mitre = entry.get("mitre_context")
            ids_detections = entry.get("related_ids_detections", [])

            lines.append(f"### Alert {i+1}: {alert.get('description', 'N/A')}\n")
            lines.append(f"- **Rule ID:** {alert.get('rule_id', 'N/A')} | **Level:** {alert.get('level', 'N/A')} | **Time:** {str(alert.get('timestamp', 'N/A'))[:19]}")
            lines.append(f"- **Agent:** {agent.get('name', 'N/A')} (Status: {agent.get('status', 'N/A')}, IP: {agent.get('ip', 'N/A')})")
            lines.append(f"- **Agent Vulnerabilities:** {vuln_status.get('total', 0)} total ({vuln_status.get('critical', 0)} critical, {vuln_status.get('high', 0)} high)")

            if mitre:
                tactics = ", ".join(mitre.get("tactics", [])) or "N/A"
                techniques = ", ".join(mitre.get("techniques", [])) or "N/A"
                lines.append(f"- **MITRE Tactics:** {tactics}")
                lines.append(f"- **MITRE Techniques:** {techniques}")

            if ids_detections:
                lines.append(f"- **Related IDS Detections:** {', '.join(ids_detections[:3])}")

            groups = alert.get("groups", [])
            if groups:
                lines.append(f"- **Rule Groups:** {', '.join(groups[:5])}")
            lines.append("")

        return "\n".join(lines)


    # ========================================================================
    # Pallas 3.3: Wazuh Correlation Formatters (12 methods)
    # ========================================================================

    def format_vulnerability_with_agents_response(self, data, summary=None, **kwargs):
        """Format CVE-centric vulnerability list with affected agent details."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Vulnerabilities with Agent Details")

        s = summary or {}
        queried_aid = s.get("queried_agent_id") or kwargs.get("agent_id") or ""
        queried_aname = s.get("queried_agent_name") or ""
        queried_os = str(s.get("queried_os") or kwargs.get("os") or "").lower()

        # Defensive OS filter (CLAUDE.md §1 STRICT FILTERING) — drop CVEs whose
        # rendered agent doesn't match the queried OS. Belt-and-suspenders on top
        # of the indexer / correlation filters in case docs leak through.
        if queried_os:
            os_keywords = self._os_aliases(queried_os)
            pre = len(data)
            def _matches_os(v):
                agent = v.get("agent") or {}
                os_obj = agent.get("os") or {}
                parts = [str(agent.get("type") or "")]
                if isinstance(os_obj, str):
                    parts.append(os_obj)
                elif isinstance(os_obj, dict):
                    for k in ("platform", "family", "name", "full", "type", "kernel"):
                        parts.append(str(os_obj.get(k) or ""))
                blob = " ".join(parts).lower()
                if not blob.strip():
                    return True   # missing OS metadata — keep, don't silently drop
                return any(kw in blob for kw in os_keywords)
            data = [v for v in data if _matches_os(v)]
            dropped = pre - len(data)
            if dropped:
                logger.info(
                    f"[VULN_FORMATTER] dropped {dropped} non-{queried_os} CVEs "
                    f"(defensive OS filter)"
                )

        # Header reflects whether this is an agent-scoped, OS-scoped or fleet-wide view
        scope_parts = []
        if queried_aid:
            agent_label = f"agent {queried_aid}"
            if queried_aname:
                agent_label += f" ({queried_aname})"
            scope_parts.append(agent_label)
        if queried_os:
            os_pretty = {"linux": "Linux", "windows": "Windows", "darwin": "macOS"}.get(queried_os, queried_os.title())
            scope_parts.append(f"{os_pretty} hosts")
        if scope_parts:
            header = f"## Vulnerabilities for {' / '.join(scope_parts)}\n"
        else:
            header = "## Vulnerabilities Correlated with Agent Details\n"
        lines = [header]

        # When scoped to a single agent, "Affected Agents" should reflect THAT one
        # agent — not "0" — so the analyst sees the right context.
        affected = s.get("affected_agents", 0)
        if queried_aid and affected == 0 and len(data) > 0:
            affected = 1
        lines.append(
            f"**Total Vulnerabilities:** {s.get('total_vulnerabilities', 0)} | "
            f"**Unique CVEs:** {s.get('unique_cves', 0)} | "
            f"**Affected Agents:** {affected}"
        )
        by_sev = s.get("by_severity", {})
        if by_sev:
            lines.append(f"**By Severity:** Critical: {by_sev.get('Critical', 0)} | High: {by_sev.get('High', 0)} | Medium: {by_sev.get('Medium', 0)} | Low: {by_sev.get('Low', 0)}\n")

        if data:
            lines.append("| CVE ID | Severity | CVSS | Package | Agent | Agent Status | Risk Factors |")
            lines.append("|--------|----------|------|---------|-------|-------------|--------------|")
            for v in data[:50]:
                cve = v.get("cve_id", "N/A")
                sev = v.get("severity", "N/A")
                cvss = v.get("cvss_score", "N/A")
                pkg = v.get("package", {})
                pkg_str = f"{pkg.get('name', 'N/A')} {pkg.get('version', '')}" if isinstance(pkg, dict) else str(pkg)
                agent = v.get("agent", {})
                agent_name = agent.get("name", "N/A") if isinstance(agent, dict) else "N/A"
                agent_status = agent.get("status", "N/A") if isinstance(agent, dict) else "N/A"
                risk = ", ".join(v.get("risk_factors", [])[:3]) or "None"
                lines.append(f"| {cve} | {sev} | {cvss} | {pkg_str} | {agent_name} | {agent_status} | {risk} |")
            lines.append("")

        return "\n".join(lines)

    def format_agents_with_vulnerabilities_response(self, data, summary=None, **kwargs):
        """Format agent-centric view with vulnerability counts per agent."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Agents with Vulnerabilities")

        s = summary or {}
        lines = ["## Agents Ranked by Vulnerability Exposure\n"]
        lines.append(f"**Total Agents:** {s.get('total_agents', 0)} | **With Vulnerabilities:** {s.get('agents_with_vulnerabilities', 0)} | **Total Vulns:** {s.get('total_vulnerabilities', 0)}\n")

        if data:
            lines.append("| Agent | ID | IP | Status | OS | Vulns | Critical | High | Medium | Low | Risk Score |")
            lines.append("|-------|----|----|--------|-----|-------|----------|------|--------|-----|------------|")
            for a in data[:30]:
                by_sev = a.get("by_severity", {})
                lines.append(f"| {a.get('name', 'N/A')} | {a.get('id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('status', 'N/A')} | {a.get('os', 'N/A')} | {a.get('vulnerability_count', 0)} | {by_sev.get('critical', 0)} | {by_sev.get('high', 0)} | {by_sev.get('medium', 0)} | {by_sev.get('low', 0)} | {a.get('risk_score', 0)} |")

            # Top vulnerabilities per agent
            for a in data[:5]:
                top_vulns = a.get("top_vulnerabilities", [])
                if top_vulns:
                    lines.append(f"\n**Top CVEs for {a.get('name', 'N/A')}:** {', '.join(v.get('cve_id', '') for v in top_vulns[:5])}")
            lines.append("")

        return "\n".join(lines)

    def format_disconnected_critical_vulns_response(self, data, summary=None, **kwargs):
        """Format disconnected agents with critical vulnerabilities."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Disconnected Agents with Critical Vulnerabilities")

        s = summary or {}
        lines = ["## Disconnected Agents with Critical Vulnerabilities\n"]
        lines.append(f"**Disconnected with Critical Vulns:** {s.get('disconnected_agents_with_critical_vulns', 0)} | **Total Disconnected:** {s.get('total_disconnected_agents', 0)} | **Critical Vulns Found:** {s.get('total_critical_vulns_seen', 0)}\n")

        if data:
            lines.append("| Agent | ID | IP | OS | Last Seen | Critical Vulns | Risk Score |")
            lines.append("|-------|----|----|----|-----------|---------------|------------|")
            for a in data[:30]:
                lines.append(f"| {a.get('name', 'N/A')} | {a.get('id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('os', 'N/A')} | {a.get('last_keep_alive', 'N/A')} | {a.get('critical_vulnerability_count', 0)} | {a.get('risk_score', 0)} |")

            for a in data[:5]:
                top_cves = a.get("top_critical_vulnerabilities", [])
                if top_cves:
                    cve_list = ", ".join(f"{v.get('cve_id', '?')} (CVSS:{v.get('cvss_score', '?')})" for v in top_cves[:5])
                    lines.append(f"\n**Critical CVEs on {a.get('name', 'N/A')}:** {cve_list}")
            lines.append("")
        else:
            lines.append("No disconnected agents with critical vulnerabilities found.\n")

        return "\n".join(lines)

    def format_top_agents_vuln_count_response(self, data, summary=None, **kwargs):
        """Format top agents ranked by vulnerability count."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Top Agents by Vulnerability Count")

        s = summary or {}
        lines = ["## Top Agents by Vulnerability Count\n"]
        lines.append(f"**Showing Top:** {s.get('top_n', len(data))} | **Total Vulnerabilities:** {s.get('total_vulnerabilities_seen', 0)}\n")

        if data:
            lines.append("| Rank | Agent | ID | IP | Status | OS | Vuln Count |")
            lines.append("|------|-------|----|----|--------|-----|-----------|")
            for i, a in enumerate(data[:20], 1):
                lines.append(f"| {i} | {a.get('name', 'N/A')} | {a.get('id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('status', 'N/A')} | {a.get('os', 'N/A')} | {a.get('vulnerability_count', 0)} |")
            lines.append("")

        return "\n".join(lines)

    def format_active_agents_threshold_response(self, data, summary=None, **kwargs):
        """Format active agents exceeding vulnerability threshold."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Active Agents Over Vulnerability Threshold")

        s = summary or {}
        threshold = s.get("threshold", "N/A")
        lines = ["## Active Agents Exceeding Vulnerability Threshold\n"]
        lines.append(f"**Threshold:** {threshold} vulnerabilities | **Agents Over Threshold:** {s.get('active_agents_over_threshold', len(data))}\n")

        if data:
            lines.append("| Agent | ID | IP | OS | Vuln Count |")
            lines.append("|-------|----|----|----|-----------|")
            for a in data[:30]:
                lines.append(f"| {a.get('name', 'N/A')} | {a.get('id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('os', 'N/A')} | {a.get('vulnerability_count', 0)} |")
            lines.append("")
        else:
            lines.append(f"No active agents exceed the threshold of {threshold} vulnerabilities.\n")

        return "\n".join(lines)

    def format_active_agents_high_vulns_response(self, data, summary=None, **kwargs):
        """Format active agents with high/critical vulnerabilities."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Active Agents with High Vulnerabilities")

        s = summary or {}
        lines = ["## Active Agents with High/Critical Vulnerabilities\n"]
        lines.append(f"**Agents Affected:** {s.get('active_agents_with_high_vulns', len(data))} | **Total High Vulns:** {s.get('total_high_vulns_seen', 0)}\n")

        if data:
            lines.append("| Agent | ID | IP | OS | High Vuln Count |")
            lines.append("|-------|----|----|----|----------------|")
            for a in data[:30]:
                lines.append(f"| {a.get('name', 'N/A')} | {a.get('id', 'N/A')} | {a.get('ip', 'N/A')} | {a.get('os', 'N/A')} | {a.get('high_vulnerability_count', 0)} |")
            lines.append("")

        return "\n".join(lines)

    def format_compare_vulns_status_response(self, data, summary=None, **kwargs):
        """Format vulnerability comparison: active vs disconnected agents."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0:
            comparison = data[0] if isinstance(data[0], dict) else {}
        elif isinstance(data, dict):
            comparison = data
        else:
            return self._safe_fallback(data, "Vulnerability Comparison")

        s = summary or comparison
        lines = ["## Vulnerability Comparison: Active vs Disconnected Agents\n"]
        lines.append("| Status | Vulnerability Count |")
        lines.append("|--------|-------------------|")
        lines.append(f"| Active Agents | {s.get('active_vulnerabilities', 0)} |")
        lines.append(f"| Disconnected Agents | {s.get('disconnected_vulnerabilities', 0)} |")
        lines.append(f"| Unknown Status | {s.get('unknown_status_vulnerabilities', 0)} |")
        total = s.get('active_vulnerabilities', 0) + s.get('disconnected_vulnerabilities', 0) + s.get('unknown_status_vulnerabilities', 0)
        lines.append(f"| **Total** | **{total}** |")
        lines.append("")

        active = s.get('active_vulnerabilities', 0)
        disconnected = s.get('disconnected_vulnerabilities', 0)
        if disconnected > active and disconnected > 0:
            lines.append("**Warning:** Disconnected agents have MORE vulnerabilities than active agents. These unpatched, offline systems represent significant risk.\n")
        elif active > disconnected and active > 0:
            lines.append("**Note:** Active agents carry the majority of vulnerabilities. Focus patching efforts on online systems.\n")

        return "\n".join(lines)

    def format_alerts_with_agents_response(self, data, summary=None, **kwargs):
        """Format alerts enriched with agent context."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Alerts with Agent Context")

        s = summary or {}
        total_alerts = s.get('total_alerts', 0)
        shown_alerts = s.get('shown_alerts', len(data))

        lines = ["## Alerts Enriched with Agent Context\n"]
        # Distinguish the TRUE total in the time range from the sample we're rendering.
        # `total_alerts` comes from track_total_hits (accurate count); `shown_alerts`
        # is the number actually returned by the API (capped at the request limit).
        if total_alerts > shown_alerts:
            lines.append(
                f"**Total Alerts (in time range):** {total_alerts:,} | "
                f"**Sample Analyzed:** {shown_alerts:,} | "
                f"**Unique Agents (in sample):** {s.get('unique_agents', 0)}"
            )
            lines.append(
                f"\n> ⚠️ The indexer returned **{shown_alerts:,} of {total_alerts:,}** alerts. "
                f"The severity breakdown and rule counts below reflect this sample. "
                f"For a full distribution, run `show alert summary` (uses aggregations, not sampling).\n"
            )
        else:
            lines.append(
                f"**Total Alerts:** {total_alerts:,} | "
                f"**Unique Agents:** {s.get('unique_agents', 0)}"
            )

        sev_bd = s.get("severity_breakdown", {})
        if sev_bd:
            sample_label = "Severity (in sample)" if total_alerts > shown_alerts else "Severity"
            lines.append(
                f"**{sample_label}:** Critical: {sev_bd.get('critical', 0)} | "
                f"High: {sev_bd.get('high', 0)} | "
                f"Medium: {sev_bd.get('medium', 0)} | "
                f"Low: {sev_bd.get('low', 0)}\n"
            )

        if data:
            lines.append("| Time | Rule ID | Description | Level | Agent | Agent Status | Agent IP |")
            lines.append("|------|---------|-------------|-------|-------|-------------|----------|")
            for alert in data[:50]:
                ts = str(alert.get("timestamp", "N/A"))[:19]
                desc = str(alert.get("rule_description", "N/A"))[:60]
                lines.append(f"| {ts} | {alert.get('rule_id', 'N/A')} | {desc} | {alert.get('level', 'N/A')} | {alert.get('agent_name', 'N/A')} | {alert.get('agent_status', 'N/A')} | {alert.get('agent_ip', 'N/A')} |")

            top_rules = s.get("top_rules", [])
            if top_rules:
                lines.append("\n**Top Triggering Rules:**")
                for r in top_rules[:5]:
                    lines.append(f"- Rule {r.get('rule_id', '?')}: {r.get('count', 0)} triggers")
            lines.append("")

        return "\n".join(lines)

    def format_alerts_with_vulns_response(self, data, summary=None, **kwargs):
        """Format agents with both alerts and vulnerabilities (combined risk)."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Alerts with Vulnerabilities")

        s = summary or {}
        lines = ["## Combined Risk: Agents with Alerts AND Vulnerabilities\n"]
        lines.append(f"**Agents with Both:** {s.get('agents_with_both', 0)} | **High Risk (>=50):** {s.get('high_risk_agents', 0)}")
        lines.append(f"**Total Alerts:** {s.get('total_alerts_analyzed', 0)} | **Total Vulns:** {s.get('total_vulns_analyzed', 0)}\n")

        if data:
            lines.append("| Agent | ID | Status | Alerts (C/H/M/L) | Vulns (C/H/M/L) | Combined Risk |")
            lines.append("|-------|----|--------|------------------|-----------------|---------------|")
            for a in data[:30]:
                ac = a.get("alert_counts", {})
                vc = a.get("vulnerability_counts", {})
                alert_str = f"{ac.get('critical', 0)}/{ac.get('high', 0)}/{ac.get('medium', 0)}/{ac.get('low', 0)}"
                vuln_str = f"{vc.get('critical', 0)}/{vc.get('high', 0)}/{vc.get('medium', 0)}/{vc.get('low', 0)}"
                risk = a.get("combined_risk", 0)
                risk_label = f"**{risk}**" if risk >= 50 else str(risk)
                lines.append(f"| {a.get('agent_name', 'N/A')} | {a.get('agent_id', 'N/A')} | {a.get('agent_status', 'N/A')} | {alert_str} | {vuln_str} | {risk_label} |")
            lines.append("")

        return "\n".join(lines)

    def format_rule_mitre_agents_response(self, data, summary=None, **kwargs):
        """Format MITRE ATT&CK technique-to-rule-to-agent mapping."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "MITRE ATT&CK Rule Mapping")

        s = summary or {}
        lines = ["## MITRE ATT&CK Coverage with Agent Mapping\n"]
        lines.append(f"**Techniques Detected:** {s.get('techniques_detected', 0)} | **Rules with MITRE:** {s.get('total_rules_with_mitre', 0)} | **Total Agents:** {s.get('total_agents', 0)}\n")

        if data:
            lines.append("| Technique | Rules | Total Triggers | Top Rule IDs |")
            lines.append("|-----------|-------|---------------|-------------|")
            for t in data[:30]:
                rules = t.get("rules", [])
                rule_ids = ", ".join(str(r.get("rule_id", "?")) for r in rules[:5])
                lines.append(f"| {t.get('technique', 'N/A')} | {t.get('rule_count', 0)} | {t.get('total_triggers', 0)} | {rule_ids} |")
            lines.append("")

        return "\n".join(lines)

    def format_fim_agent_posture_response(self, data, summary=None, **kwargs):
        """Format FIM events with agent posture context."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "FIM Events with Agent Context")

        s = summary or {}
        lines = ["## FIM Events with Agent and Process Context\n"]
        lines.append(f"**Total FIM Events:** {s.get('total_fim_events', 0)} | **Agents Affected:** {s.get('agents_affected', 0)}")
        change_types = s.get("change_types", {})
        if change_types:
            ct_str = " | ".join(f"{k}: {v}" for k, v in change_types.items())
            lines.append(f"**Change Types:** {ct_str}\n")

        if data:
            lines.append("| Time | Agent | ID | Status | File Path | Change Type | User |")
            lines.append("|------|-------|----|--------|-----------|-------------|------|")
            for e in data[:50]:
                ts = str(e.get("timestamp", "N/A"))[:19]
                fp = str(e.get("file_path", "N/A"))[:60]
                lines.append(f"| {ts} | {e.get('agent_name', 'N/A')} | {e.get('agent_id', 'N/A')} | {e.get('agent_status', 'N/A')} | {fp} | {e.get('change_type', 'N/A')} | {e.get('user', 'N/A')} |")
            lines.append("")

        return "\n".join(lines)

    def format_vuln_trend_severity_response(self, data, summary=None, **kwargs):
        """Format vulnerability distribution by severity with top CVEs."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Vulnerability Trend by Severity")

        s = summary or {}
        lines = ["## Vulnerability Distribution by Severity\n"]
        lines.append(f"**Total Vulnerabilities:** {s.get('total_vulnerabilities', 0)}")
        lines.append(f"**Critical:** {s.get('critical', 0)} | **High:** {s.get('high', 0)} | **Medium:** {s.get('medium', 0)} | **Low:** {s.get('low', 0)}\n")

        for sev_group in data:
            if not isinstance(sev_group, dict):
                continue
            severity = sev_group.get("severity", "unknown").upper()
            count = sev_group.get("count", 0)
            lines.append(f"### {severity} ({count} vulnerabilities)\n")
            top10 = sev_group.get("top_10", [])
            if top10:
                lines.append("| CVE | CVSS | Package | Agent | Agent Status |")
                lines.append("|-----|------|---------|-------|-------------|")
                for v in top10[:10]:
                    lines.append(f"| {v.get('cve_id', 'N/A')} | {v.get('cvss_score', 'N/A')} | {v.get('package', 'N/A')} | {v.get('agent_name', 'N/A')} | {v.get('agent_status', 'N/A')} |")
                lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # Pallas 3.3: Suricata Correlation Formatters (6 methods)
    # ========================================================================

    def format_suricata_attacker_target_response(self, data, summary=None, **kwargs):
        """Format Suricata attacker-to-target IP mapping."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Attacker-Target Mapping")

        s = summary or {}
        lines = ["## Suricata Attacker-Target Mapping\n"]
        lines.append(f"**Total Attackers:** {s.get('total_attackers', 0)} | **Total Alerts:** {s.get('total_alerts', 0)} | **Showing:** {s.get('shown', len(data))}\n")

        if data:
            lines.append("| Source IP | Alerts | Unique Targets | Min Severity | Top Signatures |")
            lines.append("|-----------|--------|---------------|-------------|----------------|")
            for a in data[:20]:
                sigs = ", ".join(str(sig) for sig in a.get("top_signatures", [])[:3]) or "N/A"
                lines.append(f"| {a.get('src_ip', 'N/A')} | {a.get('alert_count', 0)} | {a.get('unique_targets', 0)} | {a.get('min_severity', 'N/A')} | {sigs} |")

            # Show top targets for the most active attacker
            if data[0].get("top_targets"):
                lines.append(f"\n**Top Targets for {data[0].get('src_ip', 'N/A')}:**")
                for target in data[0]["top_targets"][:5]:
                    if isinstance(target, (list, tuple)) and len(target) >= 2:
                        lines.append(f"- {target[0]}: {target[1]} alerts")
                    elif isinstance(target, dict):
                        lines.append(f"- {target.get('ip', '?')}: {target.get('count', 0)} alerts")
            lines.append("")

        return "\n".join(lines)

    def format_suricata_sig_severity_response(self, data, summary=None, **kwargs):
        """Format Suricata signature severity analysis."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Signature Severity Analysis")

        s = summary or {}
        lines = ["## Suricata Signature Severity Analysis\n"]
        lines.append(f"**Total Signatures:** {s.get('total_signatures', 0)} | **Critical Signatures:** {s.get('critical_signature_count', 0)} | **Total Alerts:** {s.get('total_alerts', 0)}")
        sev_bd = s.get("severity_breakdown", {})
        if sev_bd:
            sev_str = " | ".join(f"{k}: {v}" for k, v in sev_bd.items())
            lines.append(f"**Severity Breakdown:** {sev_str}\n")

        if data:
            lines.append("| Signature | Count | Has Critical? |")
            lines.append("|-----------|-------|--------------|")
            for sig in data[:30]:
                critical = "YES" if sig.get("has_critical") else "No"
                sig_name = str(sig.get("signature", "N/A"))[:80]
                lines.append(f"| {sig_name} | {sig.get('count', 0)} | {critical} |")
            lines.append("")

        return "\n".join(lines)

    def format_suricata_network_threat_response(self, data, summary=None, **kwargs):
        """Format Suricata network threat profile."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Network Threat Profile")

        s = summary or {}
        lines = ["## Network Threat Profile\n"]
        lines.append(f"**Total Alerts:** {s.get('total_alerts', 0)} | **Source IPs:** {s.get('total_source_ips', 0)} | **Dest IPs:** {s.get('total_dest_ips', 0)}")
        cats = s.get("top_categories", [])
        if cats:
            lines.append(f"**Top Categories:** {', '.join(str(c) for c in cats[:5])}")
        services = s.get("top_services", [])
        if services:
            lines.append(f"**Top Services:** {', '.join(str(svc) for svc in services[:5])}")
        lines.append("")

        if data:
            lines.append("| IP | Alert Count | Type |")
            lines.append("|----|------------|------|")
            for ip in data[:30]:
                lines.append(f"| {ip.get('ip', 'N/A')} | {ip.get('alert_count', 0)} | {ip.get('type', 'N/A')} |")
            lines.append("")

        return "\n".join(lines)

    def format_suricata_http_alert_response(self, data, summary=None, **kwargs):
        """Format Suricata alerts with HTTP context."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Suricata HTTP Alert Context")

        s = summary or {}
        lines = ["## Suricata Alerts with HTTP Context\n"]
        lines.append(f"**Total Alerts:** {s.get('total_alerts', data.get('alert_count', 0))} | **Unique Sources:** {s.get('unique_alert_sources', 0)} | **HTTP Events:** {s.get('total_http_events', 0)}\n")

        alerts = data.get("alerts", [])
        if alerts:
            lines.append("### Alert Details\n")
            lines.append("| Time | Signature | Severity | Source IP | Dest IP |")
            lines.append("|------|-----------|----------|----------|---------|")
            for a in alerts[:20]:
                ts = str(a.get("timestamp", "N/A"))[:19]
                sig = str(a.get("signature", a.get("alert", {}).get("signature", "N/A")))[:60]
                sev = a.get("severity", a.get("alert", {}).get("severity", "N/A"))
                lines.append(f"| {ts} | {sig} | {sev} | {a.get('src_ip', 'N/A')} | {a.get('dest_ip', 'N/A')} |")

        http_ctx = data.get("http_context", {})
        if http_ctx:
            top_urls = http_ctx.get("top_urls", [])
            if top_urls:
                lines.append("\n### Top URLs\n")
                for u in top_urls[:10]:
                    if isinstance(u, dict):
                        lines.append(f"- {u.get('key', u.get('url', 'N/A'))}: {u.get('doc_count', u.get('count', 0))} hits")
            top_ua = http_ctx.get("top_user_agents", [])
            if top_ua:
                lines.append("\n### Top User Agents\n")
                for ua in top_ua[:10]:
                    if isinstance(ua, dict):
                        lines.append(f"- {ua.get('key', ua.get('user_agent', 'N/A'))}: {ua.get('doc_count', ua.get('count', 0))} hits")
        lines.append("")

        return "\n".join(lines)

    def format_suricata_tls_threat_response(self, data, summary=None, **kwargs):
        """Format Suricata TLS threat detection results."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "TLS Threat Detection")

        s = summary or {}
        lines = ["## TLS Threat Detection Analysis\n"]
        lines.append(f"**Total TLS Events:** {s.get('total_tls_events', data.get('total_tls_events', 0))} | **Legacy TLS Found:** {s.get('legacy_tls_versions_found', 0)} ({s.get('legacy_tls_event_count', 0)} events)")
        lines.append(f"**Unique JA3 Fingerprints:** {s.get('unique_ja3_fingerprints', 0)} | **Related Alerts:** {s.get('total_alerts', data.get('total_alerts', 0))}\n")

        tls_versions = data.get("tls_versions", [])
        if tls_versions:
            lines.append("### TLS Version Distribution\n")
            lines.append("| Version | Count |")
            lines.append("|---------|-------|")
            for v in tls_versions:
                if isinstance(v, dict):
                    lines.append(f"| {v.get('version', 'N/A')} | {v.get('count', 0)} |")

        legacy = data.get("legacy_tls_found", [])
        if legacy:
            lines.append("\n### Legacy TLS Detected (Security Risk)\n")
            for item in legacy[:10]:
                if isinstance(item, dict):
                    lines.append(f"- **{item.get('version', 'N/A')}**: {item.get('count', 0)} events")

        ja3 = data.get("top_ja3_fingerprints", [])
        if ja3:
            lines.append("\n### Top JA3 Fingerprints\n")
            lines.append("| JA3 Hash | Count |")
            lines.append("|----------|-------|")
            for fp in ja3[:10]:
                if isinstance(fp, dict):
                    lines.append(f"| {fp.get('key', fp.get('hash', 'N/A'))} | {fp.get('doc_count', fp.get('count', 0))} |")
        lines.append("")

        return "\n".join(lines)

    def format_suricata_recon_response(self, data, summary=None, **kwargs):
        """Format Suricata reconnaissance detection results."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Reconnaissance Detection")

        s = summary or {}
        lines = ["## Reconnaissance Detection Results\n"]
        lines.append(f"**Total HTTP Events:** {s.get('total_http_events', 0)} | **Scanner Agents Found:** {s.get('scanner_agents_found', 0)} | **HTTP Errors:** {s.get('http_error_count', 0)}")
        lines.append(f"**Total Alerts:** {s.get('total_alerts', 0)} | **Unique Alert Sources:** {s.get('unique_alert_sources', 0)}\n")

        scanners = data.get("scanner_user_agents", [])
        if scanners:
            lines.append("### Scanner User Agents Detected\n")
            lines.append("| User Agent / Tool | Count |")
            lines.append("|-------------------|-------|")
            for sc in scanners[:15]:
                if isinstance(sc, dict):
                    lines.append(f"| {sc.get('indicator', 'N/A')} | {sc.get('count', 0)} |")

        errors = data.get("http_error_codes", [])
        if errors:
            lines.append("\n### HTTP Error Codes\n")
            lines.append("| Status Code | Count |")
            lines.append("|-------------|-------|")
            for e in errors[:10]:
                if isinstance(e, dict):
                    lines.append(f"| {e.get('status_code', 'N/A')} | {e.get('count', 0)} |")

        sources = data.get("alert_source_ips", [])
        if sources:
            lines.append(f"\n### Top Alert Source IPs\n")
            for ip in sources[:10]:
                lines.append(f"- {ip}")
        lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # Pallas 3.3: Cross-Platform Correlation Formatters (5 methods)
    # ========================================================================

    def format_combined_alert_view_response(self, data, summary=None, **kwargs):
        """Format unified Wazuh + Suricata alert timeline."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Combined Alert View")

        s = summary or {}
        lines = ["## Combined Alert View: Wazuh + Suricata\n"]
        lines.append(f"**Total Combined:** {s.get('total_combined', len(data))} | **Suricata:** {s.get('suricata_count', 0)} | **Wazuh:** {s.get('wazuh_count', 0)}\n")

        if data:
            lines.append("| Source | Time | Severity | Description | Category | Src IP | Dest IP |")
            lines.append("|--------|------|----------|-------------|----------|--------|---------|")
            for a in data[:50]:
                ts = str(a.get("timestamp", "N/A"))[:19]
                desc = str(a.get("description", "N/A"))[:50]
                lines.append(f"| {a.get('source', 'N/A')} | {ts} | {a.get('severity', 'N/A')} | {desc} | {a.get('category', 'N/A')} | {a.get('src_ip', 'N/A')} | {a.get('dest_ip', 'N/A')} |")
            lines.append("")

        return "\n".join(lines)

    def format_agents_suricata_detections_response(self, data, summary=None, **kwargs):
        """Format agent-to-Suricata IDS detection mapping."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Agent-IDS Detection Mapping")

        s = summary or {}
        lines = ["## Agent-to-IDS Detection Mapping\n"]
        lines.append(f"**Total Agents:** {s.get('total_agents', 0)} | **With Detections:** {s.get('agents_with_detections', 0)} | **Total Suricata Alerts:** {s.get('total_suricata_alerts', 0)}\n")

        if data:
            lines.append("| Agent | ID | IP | Status | Suricata Alerts | As Source | As Dest | Unique Sigs |")
            lines.append("|-------|----|-----|--------|----------------|-----------|---------|-------------|")
            for a in data[:30]:
                lines.append(f"| {a.get('agent_name', 'N/A')} | {a.get('agent_id', 'N/A')} | {a.get('agent_ip', 'N/A')} | {a.get('agent_status', 'N/A')} | {a.get('suricata_alerts', 0)} | {a.get('as_source', 0)} | {a.get('as_destination', 0)} | {a.get('unique_signatures', 0)} |")
            lines.append("")

        return "\n".join(lines)

    def format_ioc_suricata_context_response(self, data, summary=None, **kwargs):
        """Format IoC analysis enriched with Suricata network context."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0:
            data = data[0] if isinstance(data[0], dict) else {}
        if not isinstance(data, dict):
            return self._safe_fallback(data, "IoC with Suricata Context")

        s = summary or {}
        lines = ["## IoC Analysis with Suricata Context\n"]
        lines.append(f"**Indicator:** {s.get('indicator', data.get('indicator', 'N/A'))} | **Suricata Matches:** {s.get('suricata_matches', data.get('suricata_detection_count', 0))}\n")

        ioc_data = data.get("ioc_data", {})
        if ioc_data and isinstance(ioc_data, dict):
            lines.append("### IoC Details\n")
            for k, v in ioc_data.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"- **{k}:** {v}")

        detections = data.get("suricata_detections", [])
        if detections:
            lines.append("\n### Matching Suricata Detections\n")
            lines.append("| Time | Signature | Category | Severity | Direction |")
            lines.append("|------|-----------|----------|----------|-----------|")
            for d in detections[:20]:
                ts = str(d.get("timestamp", "N/A"))[:19]
                sig = str(d.get("signature", "N/A"))[:60]
                lines.append(f"| {ts} | {sig} | {d.get('category', 'N/A')} | {d.get('severity', 'N/A')} | {d.get('direction', 'N/A')} |")
        else:
            lines.append("\n**No matching Suricata detections found for this indicator.**")
        lines.append("")

        return "\n".join(lines)

    def format_comprehensive_posture_response(self, data, summary=None, **kwargs):
        """Format comprehensive security posture across all data sources."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if not isinstance(data, list):
            return self._safe_fallback(data, "Comprehensive Security Posture")

        s = summary or {}
        lines = ["## Comprehensive Security Posture\n"]
        lines.append(f"**Total Agents:** {s.get('total_agents', 0)} | **Total Vulns:** {s.get('total_vulnerabilities', 0)} | **Wazuh Alerts:** {s.get('total_wazuh_alerts', 0)} | **Suricata Alerts:** {s.get('total_suricata_alerts', 0)}")
        lines.append(f"**Agents with IDS Detections:** {s.get('agents_with_suricata_detections', 0)}\n")

        if data:
            lines.append("| Agent | ID | IP | Status | OS | Vulns | Wazuh Alerts | Suricata | Risk Score |")
            lines.append("|-------|----|-----|--------|-----|-------|-------------|----------|------------|")
            for a in sorted(data, key=lambda x: x.get("risk_score", 0), reverse=True)[:30]:
                risk = a.get("risk_score", 0)
                risk_label = f"**{risk}**" if risk >= 20 else str(risk)
                lines.append(f"| {a.get('agent_name', 'N/A')} | {a.get('agent_id', 'N/A')} | {a.get('agent_ip', 'N/A')} | {a.get('agent_status', 'N/A')} | {a.get('os', 'N/A')} | {a.get('vulnerabilities', 0)} | {a.get('wazuh_alerts', 0)} | {a.get('suricata_detections', 0)} | {risk_label} |")
            lines.append("")

        return "\n".join(lines)

    def format_unified_mitre_response(self, data, summary=None, **kwargs):
        """Format unified MITRE ATT&CK coverage from Wazuh + Suricata."""
        if isinstance(data, dict) and "correlated_data" in data:
            data = data["correlated_data"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return self._safe_fallback(data, "Unified MITRE ATT&CK Coverage")

        s = summary or {}
        lines = ["## Unified MITRE ATT&CK Coverage\n"]
        lines.append(f"**Wazuh Tactics:** {s.get('wazuh_unique_tactics', 0)} | **Suricata Tactics:** {s.get('suricata_unique_tactics', 0)} | **Unified:** {s.get('total_unified_tactics', 0)} | **Overlapping:** {s.get('overlapping_tactics', 0)}")
        lines.append(f"**Wazuh Techniques:** {s.get('wazuh_unique_techniques', 0)} | **Suricata Techniques:** {s.get('suricata_unique_techniques', 0)} | **Unified:** {s.get('total_unified_techniques', 0)}\n")

        # Tactics comparison
        unified_tactics = data.get("unified_tactics", [])
        overlap_tactics = data.get("overlap_tactics", [])
        wazuh_tactics = set(data.get("wazuh_tactics", []))
        suricata_tactics = set(data.get("suricata_tactics", []))

        if unified_tactics:
            lines.append("### Tactic Coverage\n")
            lines.append("| Tactic | Wazuh | Suricata | Both? |")
            lines.append("|--------|-------|----------|-------|")
            for tactic in unified_tactics:
                in_w = "Yes" if tactic in wazuh_tactics else "No"
                in_s = "Yes" if tactic in suricata_tactics else "No"
                both = "Yes" if tactic in overlap_tactics else "No"
                lines.append(f"| {tactic} | {in_w} | {in_s} | {both} |")
            lines.append("")

        # Technique counts
        wazuh_techs = data.get("wazuh_techniques", [])
        suricata_techs = data.get("suricata_techniques", [])
        overlap_techs = data.get("overlap_techniques", [])
        if overlap_techs:
            lines.append(f"### Overlapping Techniques ({len(overlap_techs)})\n")
            lines.append(", ".join(str(t) for t in overlap_techs[:20]))
            lines.append("")

        wazuh_only = set(wazuh_techs) - set(suricata_techs)
        suricata_only = set(suricata_techs) - set(wazuh_techs)
        if wazuh_only:
            lines.append(f"### Wazuh-Only Techniques ({len(wazuh_only)})\n")
            lines.append(", ".join(str(t) for t in sorted(wazuh_only)[:20]))
            lines.append("")
        if suricata_only:
            lines.append(f"### Suricata-Only Techniques ({len(suricata_only)})\n")
            lines.append(", ".join(str(t) for t in sorted(suricata_only)[:20]))
            lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # Pallas 3.3: New Suricata Tool Formatters (JA4 + Flow)
    # ========================================================================

    def format_suricata_ja4_response(self, raw_text):
        """Format JA4 fingerprint deep analysis."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## JA4 Fingerprint Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## JA4 Fingerprint Analysis\n\n{raw_text}"

        total = data.get("total_tls_events", 0)
        time_range = data.get("time_range", "24h")
        fingerprints = data.get("fingerprints", [])

        lines = [f"## JA4 Fingerprint Analysis ({time_range})\n\n**Total TLS Events:** {total:,}\n"]

        if not fingerprints:
            lines.append("No JA4 fingerprints found.\n")
            return "\n".join(lines)

        lines.append(f"**Unique JA4 Fingerprints:** {len(fingerprints)}\n")

        for i, fp in enumerate(fingerprints[:15], 1):
            ja4 = fp.get("ja4_fingerprint", "N/A")
            count = fp.get("count", 0)
            pct = f"{(count / total * 100):.1f}%" if total > 0 else "N/A"

            lines.append(f"### {i}. `{ja4}`")
            lines.append(f"**Hits:** {count:,} ({pct} of TLS traffic)\n")

            src_ips = fp.get("top_src_ips", [])
            if src_ips:
                lines.append("**Source IPs:** " + ", ".join(
                    f"{s.get('ip')} ({s.get('count')})" for s in src_ips
                ))

            dest_ips = fp.get("top_dest_ips", [])
            if dest_ips:
                lines.append("**Dest IPs:** " + ", ".join(
                    f"{d.get('ip')} ({d.get('count')})" for d in dest_ips
                ))

            services = fp.get("top_services", [])
            if services:
                lines.append("**Services:** " + ", ".join(
                    f"{s.get('service')} ({s.get('count')})" for s in services
                ))

            versions = fp.get("tls_versions", [])
            if versions:
                lines.append("**TLS Versions:** " + ", ".join(
                    f"{v.get('version')} ({v.get('count')})" for v in versions
                ))

            lines.append("")

        return "\n".join(lines)

    def format_suricata_flow_analysis_response(self, raw_text):
        """Format network flow/conversation analysis."""
        try:
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
        except (json.JSONDecodeError, TypeError):
            return f"## Flow Analysis\n\n{raw_text}"

        if not isinstance(data, dict):
            return f"## Flow Analysis\n\n{raw_text}"

        total = data.get("total_alerts", 0)
        time_range = data.get("time_range", "24h")
        conversations = data.get("conversations", [])
        unique = data.get("unique_conversations", len(conversations))

        lines = [f"## Network Flow / Conversation Analysis ({time_range})\n"]
        lines.append(f"**Total Alerts:** {total:,} | **Unique Conversations:** {unique}\n")

        if not conversations:
            lines.append("No flow conversations found.\n")
            return "\n".join(lines)

        lines.append("| Src IP | Dest IP | Alerts | Unique Sigs | Min Severity | Protocols | Dest Ports | Top Signatures |")
        lines.append("|--------|---------|--------|------------|-------------|-----------|-----------|----------------|")

        for conv in conversations[:30]:
            sigs = ", ".join(str(s) for s in conv.get("top_signatures", [])[:2]) or "N/A"
            protocols = ", ".join(str(p) for p in conv.get("protocols", [])) or "N/A"
            ports = ", ".join(str(p) for p in conv.get("dest_ports", [])[:3]) or "N/A"
            lines.append(f"| {conv.get('src_ip', 'N/A')} | {conv.get('dest_ip', 'N/A')} | {conv.get('alert_count', 0)} | {conv.get('unique_signatures', 0)} | {conv.get('min_severity', 'N/A')} | {protocols} | {ports} | {sigs} |")

        lines.append("")
        return "\n".join(lines)


# ============================================================================
# CHANGELOG
# ============================================================================
#
# [Round 1] Date: 2026-02-19
#   - Added format_search_events_response()
#   - Fixed _extract_data() for all 3 payload formats
#   - Added isinstance() guards throughout
#
# [Round 2] Date: 2026-02-19
#   - Added format_raw_json_response() for clean raw text presentation
#   - Updated format_statistics_response() with content-wrapper awareness
#
# [Round 3] Date: 2026-02-19
#   - Rewrote format_agent_status_response() with strict filtering + full details
#   - Added _format_agent_details_table() for ANY status filter
#   - Added _format_agent_overview() showing counts + details (replaces count-only)
#   - Added format_vulnerability_summary_response() — no more raw JSON
#   - Added format_health_response() with stability assessment
#   - Added format_ports_response() with suspicious port detection
#   - Added format_processes_response() with suspicious process detection
#   - Added _deduplicate_vulns() — CVE dedup with affected agent aggregation
#   - Added STATUS_ALIASES for "online"→"active", "offline"→"disconnected" etc.

