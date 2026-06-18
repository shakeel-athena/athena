"""
Query Orchestrator - Hybrid Intelligence System
Combines deterministic rules with LLM fallback for optimal tool selection

ARCHITECTURE:
1. High-confidence regex patterns (95% confidence, 0ms latency)
2. Medium-confidence keyword matching (75% confidence, 0ms latency)
3. LLM fallback for complex queries (variable confidence, 2s latency)
4. Validation layer for all selections
5. Multi-tool correlation with intelligent chaining
6. SOC-grade response formatting

REPLACES: chat_handler.py
INTEGRATES WITH: All 29 MCP tools via tool_registry.py
"""

import os
import re
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

from .tool_registry import ToolRegistry, ToolMetadata, ToolCategory
from .correlation_engine import CorrelationEngine, CorrelationResult
from .response_formatter import ResponseFormatter
from .llm_client import LLMClient, LocalLLMClient, LLMRouter, OllamaClient

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ----------------------------------------------------------------------------
# E1.1 (2026-06-11) Track E modularization: ToolPlan / ExecutionResult /
# QueryContext were extracted verbatim to orchestrator/planning/tool_plan.py.
# We re-export the names here so every existing
#   from wazuh_mcp_server.query_orchestrator import ToolPlan, ...
# call keeps resolving unchanged. No behaviour change.
# ============================================================================

from .orchestrator.planning.tool_plan import ToolPlan, ExecutionResult, QueryContext  # noqa: E402,F401


# ============================================================================
# M5 (2026-06-10) PALLAS_P4_AUTO_AGGREGATE
# Map row-based alert tools to their full-population aggregation counterparts.
# When the primary tool returns a sample (total >> shown) and its mapped
# aggregation tool exists, the orchestrator runs the aggregation in parallel
# and prepends a "Full-Scale Analysis" section so the LLM/analyst sees the
# population-level picture, not just the sample.
# ============================================================================
_M5_AGG_AUGMENT_MAP: Dict[str, str] = {
    "get_wazuh_alerts":       "get_wazuh_alert_summary",
    "get_suricata_alerts":    "get_suricata_alert_summary",
    "search_security_events": "analyze_alert_patterns",
}

_M5_FORMATTER_METHOD_MAP: Dict[str, str] = {
    "get_suricata_alert_summary": "format_suricata_alert_summary_response",
    "get_wazuh_alert_summary":    "format_alert_summary_response",
    "analyze_alert_patterns":     "format_alert_patterns_response",
}


# QueryContext is imported above via the planning facade (E1.1).


# ============================================================================
# HYBRID TOOL SELECTOR - Core Intelligence
# ============================================================================

class HybridToolSelector:
    """
    Hybrid tool selection combining rules and LLM.

    Strategy:
    1. Try high-confidence regex patterns first (fastest, most accurate)
    2. Fall back to keyword matching (fast, good accuracy)
    3. Use LLM for complex/ambiguous queries (slower, handles edge cases)
    4. Validate all selections against tool registry
    5. Learn from usage patterns over time
    """

    def __init__(self, llm_client: LLMClient, suricata_enabled: bool = False):
        self.llm = llm_client
        self.suricata_enabled = suricata_enabled

        # Statistics tracking
        self.stats = {
            "rule_regex": 0,
            "rule_keyword": 0,
            "llm": 0,
            "fallback": 0,
            "total": 0
        }

        # Agent name → ID cache (populated lazily on first name resolution)
        # 5-minute TTL — fresh enough for SOC use, avoids re-fetching on every query.
        self._agent_name_cache: Dict[str, str] = {}
        self._agent_name_cache_ts: float = 0.0
        self._agent_name_cache_ttl: int = 300

        # Tool executor reference — set by QueryOrchestrator after construction.
        # Used by _resolve_agent_name_to_id to fetch the agent list.
        self._tool_executor = None

        # Bounded buffers of recent queries per strategy (for /api/query_stats telemetry)
        # These help identify which queries miss regex/keyword tiers and should be
        # promoted to deterministic patterns. Kept in-memory, reset on server restart.
        from collections import deque
        self.recent_llm_queries = deque(maxlen=50)       # queries that hit LLM fallback
        self.recent_fallback_queries = deque(maxlen=50)  # queries that hit safe-default

        # ================================================================
        # HIGH-CONFIDENCE PATTERNS (Regex-based, 95%+ confidence)
        # ================================================================
        self.HIGH_CONFIDENCE_PATTERNS = {
            "vulnerability_summary_explicit": {
                # FIX: Catches "give me a vulnerability summary", "vulnerability summary",
                # "vuln summary", "vulnerability overview" etc.
                # Must be BEFORE critical_vulnerability_simple to prevent hijacking.
                "patterns": [
                    r"\bvulnerabilit\w*\s+(summary|overview|report|stats|statistics)\b",
                    r"\b(summary|overview|report|stats|statistics)\s+of\s+\bvulnerabilit",
                    r"\bcve\s+(summary|overview|report|stats)\b",
                    r"\bvuln\s+(summary|overview|report|stats)\b",
                ],
                "tool": "get_wazuh_vulnerability_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "Explicit vulnerability summary query"
            },

            "critical_vulnerability_simple": {
                "patterns": [
                    r"^(show|list|get|share|display|give)\s+(me\s+)?critical\s+(vulnerability|vulnerabilities|cve)",
                    r"critical\s+(vulnerability|vulnerabilities|cve)\s*$"
                ],
                "tool": "get_wazuh_critical_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "vulnerability_with_agents",
                "confidence": 0.92,
                "description": "Simple critical vulnerability query"
            },

            "critical_vulnerability_with_agents": {
                "patterns": [
                    r"critical\s+(vulnerability|vulnerabilities|cve).*\b(agent|host|server|machine|system)",
                    r"\b(agent|host|server|machine|system).*critical\s+(vulnerability|vulnerabilities|cve)",
                    r"critical\s+(vulnerability|vulnerabilities|cve).*\b(affected|associated|related)"
                ],
                "tool": "get_wazuh_critical_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "vulnerability_with_agents",
                "confidence": 0.92,
                "description": "Critical vulnerabilities with agent correlation"
            },

            "all_vulnerabilities_with_agents": {
                "patterns": [
                    r"(vulnerability|vulnerabilities|cve).*\b(agent|host|server|machine|affected)",
                    r"\b(agent|host|server|machine).*\b(vulnerability|vulnerabilities|cve)",
                    r"(vulnerability|vulnerabilities|cve).*\b(associated|related)\s+with",
                    r"(vulnerability|vulnerabilities|vuln).*\b(of|for|on)\s+agent\s*\d"
                ],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "vulnerability_with_agents",
                "confidence": 0.88,
                "description": "Vulnerabilities with agent details"
            },

            # OS-scoped vulnerabilities — bumps queries like "share
            # vulnerabilities on ubuntu" / "linux vulnerabilities" from the
            # 0.82 keyword tier into the high-confidence regex tier. Routes
            # to the same tool the keyword tier was already routing to; the
            # OS entity is forwarded by _build_arguments either way. The
            # change is cosmetic (UI shows 96% Regex Match instead of 82%
            # Keyword Match) but reassures the operator the routing is
            # deterministic.
            "vulnerabilities_by_os": {
                "patterns": [
                    # "vulnerabilities (on|for|in|of|across) <os>"
                    r"\b(?:vulnerabilit\w*|cves?|vulns?)\b\s+(?:on|for|in|of|across)\s+\b(?:linux|ubuntu|debian|centos|red\s*hat|rhel|fedora|alpine|kali|amazon\s*linux|amzn|suse|opensuse|oracle|rocky|almalinux|alma|arch|raspbian|gentoo|windows|win|macos|mac\s*os|darwin|osx)\b",
                    # "<os> vulnerabilities"
                    r"\b(?:linux|ubuntu|debian|centos|red\s*hat|rhel|fedora|alpine|kali|amazon\s*linux|amzn|suse|opensuse|oracle|rocky|almalinux|alma|arch|raspbian|gentoo|windows|win|macos|mac\s*os|darwin|osx)\s+(?:vulnerabilit\w*|cves?|vulns?)\b",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": [],
                "confidence": 0.96,
                "description": "Vulnerabilities scoped to a specific OS family"
            },

            "active_agents_only": {
                "patterns": [
                    r"^(show|list|get|share|display|give)\s+(me\s+)?(only\s+)?active\s+agents?",
                    r"^active\s+agents?\s+(only|list)?",
                    r"^(show|list|get|share|display|give)\s+(me\s+)?(running|online|connected)\s+agents?",
                    r"\bonly\s+(active|running|online|connected)\s+agents?\b",
                    r"\b(running|online|connected)\s+agents?\b"
                ],
                "tool": "get_wazuh_running_agents",
                "secondary": [],
                "confidence": 0.95,
                "description": "Active agents only"
            },


            "agent_ports": {
                "patterns": [
                    r"\b(agent|agents)\b.*\b(open\s+ports?|ports?)\b",
                    r"\b(open\s+ports?|ports?)\b.*\b(agent|agents)\b",
                    r"\b(list|show|get|share|display|give)\b.*\bports?\b.*\b(for|on|of)\b.*\bagent\b",
                    r"\bports?\b.*\bagent\s*\d",
                    r"\b(agent|agents)\b.*\bpots?\b",
                    r"\bpots?\b.*\b(agent|agents)\b"
                ],
                "tool": "get_agent_ports",
                "secondary": [],
                "confidence": 0.90,
                "description": "Agent open ports query (requires agent_id)"
            },

            "agent_processes": {
                "patterns": [
                    r"\b(agent|agents)\b.*\b(process|processes|running\s+processes)\b",
                    r"\b(process|processes|running\s+processes)\b.*\b(agent|agents)\b",
                    r"\b(list|show|get|share|display|give)\b.*\bprocess(es)?\b.*\b(for|on|of)\b.*\bagent\b",
                    r"\bprocess(es)?\b.*\bagent\s*\d"
                ],
                "tool": "get_agent_processes",
                "secondary": [],
                "confidence": 0.90,
                "description": "Agent processes query (requires agent_id)"
            },

            "agent_configuration": {
                "patterns": [
                    r"\b(agent|agents)\b.*\b(config|configuration)\b",
                    r"\b(config|configuration)\b.*\b(agent|agents)\b",
                    r"\b(get|show)\b.*\bagent\b.*\bconfiguration\b"
                ],
                "tool": "get_agent_configuration",
                "secondary": [],
                "confidence": 0.96,
                "description": "Agent configuration query (requires agent_id)"
            },

            "agent_health": {
                "patterns": [
                    # FIX: All patterns now require 'agent' explicitly so that
                    # "check cluster health" / "check wazuh health" do NOT match here.
                    r"\b(check|validate|verify)\b.*\bagent\b.*\b(health|status)\b",
                    r"\bagent\b.*\b(check|validate|verify)\b.*\b(health|status)\b",
                    r"\bagent\b.*\b(health|heartbeat|keepalive)\b",
                    r"\b(health|heartbeat)\b.*\bagent\b",
                    # 2026-05-15 fix: handle "agent<digits>" no-space form so
                    # "show agent001 health" / "agent#073 health" route here
                    # instead of falling through regex+keyword to the LLM (which
                    # otherwise picks the generic get_wazuh_agents tool).
                    r"\bagent[#_\-]*\d{1,6}\b.*\b(health|heartbeat|keepalive|status)\b",
                    r"\b(health|status|heartbeat|keepalive)\b.*\bagent[#_\-]*\d{1,6}\b",
                ],
                "tool": "check_agent_health",
                "secondary": [],
                "confidence": 0.90,
                "description": "Agent health query (requires agent_id)"
            },

            "disconnected_agents_only": {
                "patterns": [
                    r"^(show|list|get|share|display|give)\s+(me\s+)?(only\s+)?(disconnected|offline|down|inactive)\s+agents?",
                    r"^(disconnected|offline|down|inactive)\s+agents?\s+(only|list)?",
                    r"\bonly\s+(disconnected|offline|down|inactive)\s+agents?\b",
                    r"\b(disconnected|offline|down)\s+agents?\b"
                ],
                "tool": "get_wazuh_agents",
                "args": {"status": "disconnected"},
                "secondary": [],
                "confidence": 0.95,
                "description": "Disconnected agents only"
            },

            "agent_status_with_vulnerabilities": {
                "patterns": [
                    r"agent\s+status.*\b(vulnerability|vulnerabilities|cve)",
                    r"\b(vulnerability|vulnerabilities|cve).*agent\s+status",
                    r"agents?.*\bwith\b.*(vulnerability|vulnerabilities|their\s+cve)",
                    r"agents?.*\b(vulnerability|vulnerabilities)\s*(count|load|total|exposure)"
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities"],
                "correlation": "agents_with_vulnerabilities",
                "confidence": 0.89,
                "description": "Agent status with vulnerability correlation"
            },

            # FIX R4: search_security_events MUST be BEFORE alert_investigation
            # to prevent "security events" regex in alert_investigation from hijacking.
            # Plan turn-2: "search" is the dominant signal — if the user asked to
            # search/find/look-up/query, route here regardless of whether they
            # said "event" or "alert" or "log" or just a technique name.
            "search_security_events_explicit": {
                "patterns": [
                    # ANY query starting with search/find/look-up/query → this tool
                    r"^\s*(?:please\s+)?(?:search|find|look\s+up|look\s+for|query)\b",
                    # search + event/alert/log/incident/activity/hit (any noun)
                    r"\b(?:search|find|look\s+up|query)\s+(?:for\s+)?(?:security\s+|wazuh\s+)?(?:event|alert|log|incident|activity|hit|threat)s?\b",
                    # search + structured filter (rule N, level N, T-code, mitre)
                    r"\b(?:search|find|query)\b.*\b(?:rule\s+\d+|level\s+\d+|t\d{4}|mitre)\b",
                    # search + "for X matching Y" / "with X"
                    r"\bsearch\b.*\b(?:for|with|matching|containing)\b",
                    # legacy patterns
                    r"\b(virustotal|ossec|rootcheck)\b.*\b(event|events|alert|alerts)\b",
                    r"\b(event|events)\b.*\b(virustotal|ossec|rootcheck)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "confidence": 0.97,
                "description": "Search-first routing — 'search/find/lookup' → search_security_events with structured filters"
            },

            # FIX R4: analyze_alert_patterns - MUST be BEFORE alert_investigation
            "alert_patterns_explicit": {
                "patterns": [
                    r"\balert\s+pattern",
                    r"\bpattern\b.*\balert",
                    r"\b(analyze|analysis|detect|find)\b.*\b(alert|security)\s+pattern",
                    r"\b(recurring|frequent|repeated)\s+alert",
                    r"\balert\s+(trend|trends|frequency|recurring)\b",
                ],
                "tool": "analyze_alert_patterns",
                "secondary": [],
                "confidence": 0.93,
                "description": "Alert pattern analysis"
            },

            # FIX R4+R12: check_ioc_reputation - MUST be BEFORE threat_analysis keywords
            "ioc_reputation_explicit": {
                "patterns": [
                    r"\bioc\s+reputation\b",
                    r"\breputation\b.*\b(ioc|indicator|ip|hash|domain)\b",
                    r"\b(check|verify|lookup|look\s+up|detect|scan|investigate)\b.*\bioc\b",
                    r"\b(check|verify|lookup|look\s+up|detect|scan|investigate)\b.*\breputation\b",
                    r"\bioc\b.*\b(check|verify|lookup|look\s+up|reputation|detect|scan)\b",
                    r"\bioc\b.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                    r"\b(detect|scan)\b.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                ],
                "tool": "check_ioc_reputation",
                "secondary": [],
                "correlation": None,
                "confidence": 0.93,
                "description": "IoC reputation check"
            },

            # FIX R4: generate_security_report - HIGH_CONFIDENCE to beat keyword "port" in "report"
            "security_report_explicit": {
                "patterns": [
                    r"\b(generate|create|build|produce)\b.*\breport\b",
                    r"\b(security|daily|weekly|monthly)\s+report\b",
                    r"\breport\b.*\b(security|daily|weekly|monthly)\b",
                    r"\b(generate|create)\b.*\b(security|soc)\b",
                ],
                "tool": "generate_security_report",
                "secondary": [],
                "confidence": 0.93,
                "description": "Security report generation"
            },

            # FIX R4: get_wazuh_alert_summary - MUST be BEFORE alert_investigation
            "alert_summary_explicit": {
                "patterns": [
                    r"\balert\s+summary\b",
                    r"\bsummary\b.*\balert",
                    r"\bsummariz\w*\s+alert",
                    r"\balert\b.*\b(overview|breakdown|distribution)\b",
                    r"\b(show|get|share|display|give)\b.*\balert\s+summary\b",
                ],
                "tool": "get_wazuh_alert_summary",
                "secondary": [],
                "confidence": 0.93,
                "description": "Alert summary aggregation"
            },
            # === PALLAS_NLTEST_ROUTING === DO NOT REMOVE
            # Added 20260512T095002Z after NL test phase 1 found these queries fell through
            # to the LLM router and got mis-classified.
            "top_rules_explicit_v2": {
                "patterns": [
                    r"\btop\s+(wazuh\s+)?rules?\b",
                    r"\bmost\s+(common|triggered|frequent|fired)\s+rules?\b",
                    r"\brules?\s+summary\b",
                    r"\bshow\s+rules?\s+stat",
                    r"\b(show|list|get|display)\s+top\s+rules?\b",
                ],
                "tool": "get_wazuh_rules_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "Top rules / rules summary (NL-test fix)"
            },
            "sca_results_explicit_v2": {
                "patterns": [
                    r"\b(show|list|get|display|share)\s+sca\s+results?\b",
                    r"\bsca\s+results?\b",
                    r"\bsca\s+findings?\b",
                    r"\bget\s+sca\b",
                ],
                "tool": "get_sca_results",
                "secondary": [],
                "confidence": 0.97,
                "description": "SCA results (NL-test fix)"
            },
            # === END PALLAS_NLTEST_ROUTING ===
            # === PALLAS_NLTEST_ROUTING_V2 === DO NOT REMOVE
            # Added 20260512T122707Z after NL test Phase 2 surfaced these misroutes.

            # Bump WAZUH alert summary above SURICATA alert summary (the suricata
            # pattern was catching plain "alert summary").
            "wazuh_alert_summary_priority": {
                "patterns": [
                    r"^\s*(show\s+|get\s+|give\s+me\s+)?alert\s+summary\s*$",
                    r"^\s*alerts?\s+overview\s*$",
                    r"^\s*alerts?\s+breakdown\s*$",
                    r"\bwazuh\s+alert\s+(summary|overview|breakdown)\b",
                ],
                "tool": "get_wazuh_alert_summary",
                "secondary": [],
                "confidence": 0.98,
                "description": "Plain alert summary defaults to Wazuh (not Suricata)"
            },

            # "inactive agents" / "agents that are not active" / "not-active agents"
            "inactive_agents_explicit": {
                "patterns": [
                    r"\binactive\s+agents?\b",
                    r"\bnon[-\s]?active\s+agents?\b",
                    r"\bnot\s+active\s+agents?\b",
                    r"\bagents?\s+that\s+are\s+(inactive|not\s+active)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "confidence": 0.97,
                "description": "Inactive agents (NL-test fix)"
            },

            # Manager daemon statistics: analysisd, remoted, manager-wide
            "wazuh_manager_stats_explicit": {
                "patterns": [
                    r"\banalysis(d|D)\s+stats?\b",
                    r"\banalysisd\s+(statistics|stats|metrics)\b",
                    r"\bwazuh\s+(manager\s+)?(statistics|stats|metrics)\b",
                    r"\bmanager\s+(statistics|stats|metrics)\b",
                    r"\b(show|get|display)\s+wazuh\s+stats?\b",
                ],
                "tool": "get_wazuh_statistics",
                "secondary": [],
                "confidence": 0.97,
                "description": "Wazuh manager / analysisd statistics (NL-test fix)"
            },
            # === END PALLAS_NLTEST_ROUTING_V2 ===
            # === PALLAS_NLTEST_ROUTING_V3 === DO NOT REMOVE
            # Added 20260512T124914Z: Phase 3 found "suspicious tls fingerprints" misrouting
            # to generic wazuh alerts. Bind TLS/JA3 terminology explicitly.
            "suspicious_tls_ja3_explicit": {
                "patterns": [
                    r"\bsuspicious\s+tls\s+(fingerprints?|certificates?|sessions?|connections?)\b",
                    r"\bsuspicious\s+ja3s?\b",
                    r"\b(unusual|anomalous|odd)\s+tls\b",
                    r"\b(rogue|malicious)\s+(tls|certificate)s?\b",
                ],
                "tool": "get_suricata_tls_analysis",
                "secondary": ["get_suricata_ja3_fingerprints"],
                "correlation": "suricata_tls_anomaly_analysis",
                "confidence": 0.96,
                "description": "Suspicious TLS/JA3 hunts (NL-test fix)"
            },
            # === END PALLAS_NLTEST_ROUTING_V3 ===
            # === PALLAS_CLOUD_MODULES === DO NOT REMOVE
            "office365_summary_explicit": {
                "patterns": [
                    r"\boffice\s*365\s+(summary|overview|breakdown|stats)\b",
                    r"\bo365\s+(summary|overview|breakdown)\b",
                    r"\bmicrosoft\s+365\s+(summary|overview)\b",
                ],
                "tool": "get_office365_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "Office365 summary"
            },
            "office365_events_explicit": {
                "patterns": [
                    r"\b(show|list|get|display)\s+office\s*365\b",
                    r"\boffice\s*365\s+(events?|logs?|activity|audit)\b",
                    r"\bo365\s+(events?|logs?|audit)\b",
                    r"\bmicrosoft\s+365\s+(events?|logs?)\b",
                    r"\b(exchange|sharepoint|azure\s+ad|aad)\s+(events?|logs?|activity)\b",
                    r"\bmailbox\s+(access|audit|events?)\b",
                ],
                "tool": "get_office365_events",
                "secondary": [],
                "confidence": 0.96,
                "description": "Office365 events list"
            },
            "github_summary_explicit": {
                "patterns": [
                    r"\bgithub\s+(summary|overview|breakdown|stats?)\b",
                ],
                "tool": "get_github_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "GitHub summary"
            },
            "github_events_explicit": {
                "patterns": [
                    r"\b(show|list|get|display)\s+github\b",
                    r"\bgithub\s+(events?|logs?|activity|audit|workflow|workflows|clones?)\b",
                ],
                "tool": "get_github_events",
                "secondary": [],
                "confidence": 0.96,
                "description": "GitHub events list"
            },
            "aws_summary_explicit": {
                "patterns": [
                    r"\baws\s+(summary|overview|breakdown|stats?)\b",
                    r"\bamazon\s+(summary|overview)\b",
                ],
                "tool": "get_aws_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "AWS summary"
            },
            "aws_events_explicit": {
                "patterns": [
                    r"\b(show|list|get|display)\s+aws\b",
                    r"\baws\s+(events?|logs?|activity|audit|alerts?)\b",
                    r"\bcloudtrail\s+(events?|logs?)\b",
                    r"\b(amazon\s+)?guardduty\b",
                    r"\baws\s+waf\s+(blocks?|events?|logs?)\b",
                ],
                "tool": "get_aws_events",
                "secondary": [],
                "confidence": 0.96,
                "description": "AWS events list"
            },
            # === END PALLAS_CLOUD_MODULES ===
            # === PALLAS_DEFENDER_INTUNE === DO NOT REMOVE
            # SCOPING: every pattern REQUIRES "defender" or "intune" (or
            # defender-specific terms like "mde") to avoid cross-leak with
            # Wazuh / Suricata queries.

            # Defender: audit log
            "defender_audit_summary_explicit": {
                "patterns": [
                    r"\b(defender|mde)\s+audit\s+(summary|overview|breakdown|stats)\b",
                ],
                "tool": "get_defender_audit_summary",
                "secondary": [],
                "confidence": 0.98,
                "description": "Defender audit summary"
            },
            "defender_audit_events_explicit": {
                "patterns": [
                    r"\b(defender|mde)\s+(audit|action(s)?)\b",
                    r"\bdefender\s+(audit\s+log|audit\s+events?|action\s+history)\b",
                    r"\b(show|list|get|display)\s+defender\s+(audit|actions?)\b",
                ],
                "tool": "get_defender_audit_events",
                "secondary": [],
                "confidence": 0.97,
                "description": "Defender audit events list"
            },
            # Defender: machines
            "defender_machines_explicit": {
                "patterns": [
                    r"\bdefender\s+(machines?|devices?|endpoints?|hosts?)\b",
                    r"\bmde\s+(machines?|devices?|endpoints?)\b",
                    r"\b(show|list|get|display)\s+defender\s+(machines?|devices?|endpoints?)\b",
                ],
                "tool": "get_defender_machines",
                "secondary": [],
                "confidence": 0.97,
                "description": "Defender machines / endpoints list"
            },
            # Defender: vulnerabilities
            "defender_vulnerability_summary_explicit": {
                "patterns": [
                    r"\bdefender\s+(vulnerabilit(y|ies)|cve(s)?|vuln(s)?)\s+(summary|overview|breakdown|stats)\b",
                    r"\bmde\s+(vulnerabilit(y|ies)|cve(s)?)\s+(summary|overview)\b",
                ],
                "tool": "get_defender_vulnerability_summary",
                "secondary": [],
                "confidence": 0.98,
                "description": "Defender vulnerability summary"
            },
            "defender_vulnerabilities_explicit": {
                "patterns": [
                    r"\bdefender\s+(vulnerabilit(y|ies)|cve(s)?|vuln(s)?)\b",
                    r"\bmde\s+(vulnerabilit(y|ies)|cve(s)?)\b",
                    r"\b(show|list|get|display)\s+defender\s+(vulnerabilit(y|ies)|cve(s)?)\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "secondary": [],
                "confidence": 0.97,
                "description": "Defender vulnerabilities list"
            },

            # Intune: audit
            "intune_audit_summary_explicit": {
                "patterns": [
                    r"\bintune\s+(audit|activity)\s+(summary|overview|breakdown|stats)\b",
                    r"\bintune\s+(summary|overview)\b",
                ],
                "tool": "get_intune_audit_summary",
                "secondary": [],
                "confidence": 0.98,
                "description": "Intune audit summary"
            },
            "intune_audit_events_explicit": {
                "patterns": [
                    r"\bintune\s+(audit|audit\s+log|audit\s+events?|events?|activity|activities|logs?)\b",
                    r"\b(show|list|get|display)\s+intune\s+(audit|events?|activity|activities|logs?)\b",
                ],
                "tool": "get_intune_audit_events",
                "secondary": [],
                "confidence": 0.97,
                "description": "Intune audit events list"
            },
            # Intune: inventory (devices/policies/profiles/etc.)
            "intune_inventory_explicit": {
                "patterns": [
                    r"\bintune\s+(devices?|managed\s+devices?)\b",
                    r"\bintune\s+(compliance\s+)?polic(y|ies)\b",
                    r"\bintune\s+(config(uration)?\s+)?profiles?\b",
                    r"\bintune\s+enrollment\b",
                    r"\bintune\s+(mobile\s+)?apps?\b",
                    r"\bintune\s+scripts?\b",
                    r"\b(show|list|get|display)\s+intune\s+(devices?|polic(y|ies)|profiles?|apps?|scripts?|enrollment)\b",
                ],
                "tool": "get_intune_inventory",
                "secondary": [],
                "confidence": 0.97,
                "description": "Intune inventory (devices/policies/profiles/apps/scripts/enrollment)"
            },
            # === END PALLAS_DEFENDER_INTUNE ===
            # === PALLAS_DEFINT_HARDSCOPE === DO NOT REMOVE
            # Added 20260513T082130Z: hard-scoping catch-all so any query mentioning
            # 'defender'/'mde'/'intune' lands in the correct tool family, even
            # when other words are typo'd. Confidence (0.93) beats the LLM
            # fallback (~0.75) but loses to the precise per-intent patterns.

            # Defender catch-all (typo-tolerant) â€” vulnerability-flavored words
            "defender_catchall_vuln": {
                "patterns": [
                    r"\bdefender\b.*?\b(vul[a-z]*|cve[a-z]*|find[a-z]*|expos[a-z]*|risk[a-z]*|exploit[a-z]*|patch[a-z]*)",
                    r"\bmde\b.*?\b(vul[a-z]*|cve[a-z]*|find[a-z]*|exploit[a-z]*)",
                ],
                "tool": "get_defender_vulnerabilities",
                "secondary": [],
                "confidence": 0.93,
                "description": "Defender vuln catch-all (typo-tolerant)"
            },
            # Defender catch-all â€” machine/device-flavored words
            "defender_catchall_machines": {
                "patterns": [
                    r"\bdefender\b.*?\b(mach[a-z]*|devic[a-z]*|endpoint[a-z]*|host[a-z]*|onbo[a-z]*|sensor[a-z]*)",
                    r"\bmde\b.*?\b(mach[a-z]*|devic[a-z]*|endpoint[a-z]*)",
                ],
                "tool": "get_defender_machines",
                "secondary": [],
                "confidence": 0.93,
                "description": "Defender machines catch-all"
            },
            # Defender catch-all â€” audit/action-flavored words
            "defender_catchall_audit": {
                "patterns": [
                    r"\bdefender\b.*?\b(audit[a-z]*|action[a-z]*|activit[a-z]*|histor[a-z]*|log[a-z]*)",
                    r"\bmde\b.*?\b(audit[a-z]*|action[a-z]*|histor[a-z]*)",
                ],
                "tool": "get_defender_audit_events",
                "secondary": [],
                "confidence": 0.93,
                "description": "Defender audit catch-all"
            },
            # Defender LAST-RESORT â€” query mentions 'defender' but nothing else specific.
            # Route to vuln summary as the most informative default for Defender.
            "defender_last_resort": {
                "patterns": [
                    r"\bdefender\b",
                    r"\bmde\b",
                ],
                "tool": "get_defender_vulnerability_summary",
                "secondary": [],
                "confidence": 0.88,
                "description": "Generic Defender mention: default to vuln summary"
            },

            # Intune catch-all (typo-tolerant)
            "intune_catchall_inventory": {
                "patterns": [
                    r"\bintune\b.*?\b(devic[a-z]*|polic[a-z]*|profil[a-z]*|app[a-z]*|enroll[a-z]*|script[a-z]*|complian[a-z]*)",
                ],
                "tool": "get_intune_inventory",
                "secondary": [],
                "confidence": 0.93,
                "description": "Intune inventory catch-all"
            },
            "intune_catchall_audit": {
                "patterns": [
                    r"\bintune\b.*?\b(audit[a-z]*|event[a-z]*|activit[a-z]*|log[a-z]*|histor[a-z]*)",
                ],
                "tool": "get_intune_audit_events",
                "secondary": [],
                "confidence": 0.93,
                "description": "Intune audit catch-all"
            },
            # Intune LAST-RESORT
            "intune_last_resort": {
                "patterns": [
                    r"\bintune\b",
                ],
                "tool": "get_intune_audit_summary",
                "secondary": [],
                "confidence": 0.88,
                "description": "Generic Intune mention: default to audit summary"
            },
            # === END PALLAS_DEFINT_HARDSCOPE ===
            # === PALLAS_DEFENDER_SEVERITY_OVERRIDE === DO NOT REMOVE
            # Added 20260513T110703Z: maximum-priority Defender patterns that win against
            # ANY Wazuh severity / "vulnerability_with_agents" pattern when
            # "defender" or "mde" appears anywhere in the query.
            # Confidence 0.99 â€” only the explicit per-intent Defender patterns
            # (also at 0.99 effectively) can equal it, and they're at the same
            # priority so they fight on regex specificity.

            "defender_critical_severity_lock": {
                "patterns": [
                    r"\b(defender|mde)\b.*?\bcritical\b.*?\b(vul|cve|find)",
                    r"\bcritical\b.*?\b(defender|mde)\b.*?\b(vul|cve)",
                    r"\b(defender|mde)\b.*?\b(vul|cve)[a-z]*.*?\bcritical\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "tool_args": {"severity": "Critical"},
                "secondary": ["get_defender_machines"],
                "confidence": 0.99,
                "description": "Defender critical vulnerabilities (severity-lock override)"
            },
            "defender_high_severity_lock": {
                "patterns": [
                    r"\b(defender|mde)\b.*?\bhigh\b.*?\b(vul|cve|find)",
                    r"\bhigh\b.*?\b(defender|mde)\b.*?\b(vul|cve)",
                    r"\b(defender|mde)\b.*?\b(vul|cve)[a-z]*.*?\bhigh\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "tool_args": {"severity": "High"},
                "secondary": ["get_defender_machines"],
                "confidence": 0.99,
                "description": "Defender high vulnerabilities (severity-lock override)"
            },
            "defender_medium_severity_lock": {
                "patterns": [
                    r"\b(defender|mde)\b.*?\bmedium\b.*?\b(vul|cve|find)",
                    r"\bmedium\b.*?\b(defender|mde)\b.*?\b(vul|cve)",
                    r"\b(defender|mde)\b.*?\b(vul|cve)[a-z]*.*?\bmedium\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "tool_args": {"severity": "Medium"},
                "secondary": ["get_defender_machines"],
                "confidence": 0.99,
                "description": "Defender medium vulnerabilities (severity-lock override)"
            },
            "defender_low_severity_lock": {
                "patterns": [
                    r"\b(defender|mde)\b.*?\blow\b.*?\b(vul|cve|find)",
                    r"\blow\b.*?\b(defender|mde)\b.*?\b(vul|cve)",
                    r"\b(defender|mde)\b.*?\b(vul|cve)[a-z]*.*?\blow\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "tool_args": {"severity": "Low"},
                "secondary": ["get_defender_machines"],
                "confidence": 0.99,
                "description": "Defender low vulnerabilities (severity-lock override)"
            },
            # Also catch "critical/high/medium/low" + "defender agents" / "in defender" etc.
            "defender_critical_with_agents_lock": {
                "patterns": [
                    r"\bcritical\s+(vulnerabilit\w*|cves?)\b.*?\b(defender|mde)\b",
                    r"\bcritical\b.*?\bon\s+(defender|mde)\s+(agents?|machines?|endpoints?)",
                    r"\bcritical\b.*?\bin\s+(defender|mde)\b",
                ],
                "tool": "get_defender_vulnerabilities",
                "tool_args": {"severity": "Critical"},
                "secondary": ["get_defender_machines"],
                "confidence": 0.99,
                "description": "Critical defender vulnerabilities (alt word order)"
            },
            # === END PALLAS_DEFENDER_SEVERITY_OVERRIDE ===

            # === PALLAS_VULN_ROUTING === DO NOT REMOVE
            # Added 20260513T085632Z: improved vulnerability routing per user feedback.
            # 1) Generic "show vulnerabilities" should give a SUMMARY (not a list).
            # 2) "show high/medium/low vulnerabilities" should pick severity-filtered tool.

            # PRIORITY 1: bare/generic vulnerability queries -> summary
            "vulns_generic_to_summary": {
                "patterns": [
                    r"^\s*(please\s+)?(show|list|get|display|view|share)\s+(all\s+)?vulnerabilit(y|ies)\s*$",
                    r"^\s*(all\s+)?vulnerabilit(y|ies)\s*$",
                    r"^\s*(please\s+)?(show|list|get|display|view|share)\s+(all\s+)?vulnerabilit(y|ies)\s+(on|across|for|of)\s+(all\s+)?(agents?|hosts?|machines?|endpoints?|servers?)\s*$",
                    r"^\s*(give\s+me|tell\s+me|provide)\s+(a\s+)?vulnerabilit(y|ies)\s+(report|overview|summary)?\s*$",
                    r"^\s*vulnerabilit(y|ies)\s+across\s+(all\s+)?(agents?|hosts?|machines?)\s*$",
                ],
                "tool": "get_wazuh_vulnerability_summary",
                "secondary": [],
                "confidence": 0.97,
                "description": "Generic vulnerability query -> summary (NOT list)"
            },

            # PRIORITY 2: severity-specific (high/medium/low) -> filtered listing
            # (Critical already handled by existing get_wazuh_critical_vulnerabilities pattern.)
            "vulns_high_severity": {
                "patterns": [
                    r"\b(show|list|get|display|view)\s+high\s+(severity\s+)?vulnerabilit(y|ies)\b",
                    r"\bhigh\s+severity\s+vulnerabilit(y|ies)\b",
                    r"\bhigh\s+vulnerabilit(y|ies)\b",
                    r"\bhigh\s+(severity\s+)?cves?\b",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "tool_args": {"severity": "High"},
                "secondary": [],
                "confidence": 0.96,
                "description": "High-severity vulnerabilities (filtered listing)"
            },
            "vulns_medium_severity": {
                "patterns": [
                    r"\b(show|list|get|display|view)\s+medium\s+(severity\s+)?vulnerabilit(y|ies)\b",
                    r"\bmedium\s+severity\s+vulnerabilit(y|ies)\b",
                    r"\bmedium\s+vulnerabilit(y|ies)\b",
                    r"\bmedium\s+(severity\s+)?cves?\b",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "tool_args": {"severity": "Medium"},
                "secondary": [],
                "confidence": 0.96,
                "description": "Medium-severity vulnerabilities (filtered listing)"
            },
            "vulns_low_severity": {
                "patterns": [
                    r"\b(show|list|get|display|view)\s+low\s+(severity\s+)?vulnerabilit(y|ies)\b",
                    r"\blow\s+severity\s+vulnerabilit(y|ies)\b",
                    r"\blow\s+vulnerabilit(y|ies)\b",
                    r"\blow\s+(severity\s+)?cves?\b",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "tool_args": {"severity": "Low"},
                "secondary": [],
                "confidence": 0.96,
                "description": "Low-severity vulnerabilities (filtered listing)"
            },
            # === END PALLAS_VULN_ROUTING ===








            # FIX R4: analyze_security_threat
            "threat_analysis_explicit": {
                "patterns": [
                    r"\b(analyze|analyse|analysis)\b.*\b(threat|security\s+threat)\b",
                    r"\bthreat\b.*\b(analyze|analyse|analysis|investigate)\b",
                    r"\b(analyze|analyse)\b.*\b(indicator|ioc)\b(?!.*reputation)",
                    r"\bthreat\s+(analysis|investigation)\b",
                ],
                "tool": "analyze_security_threat",
                "secondary": [],
                "confidence": 0.93,
                "description": "Security threat indicator analysis"
            },

            # FIX R5: Correlation query patterns — must be BEFORE alert_investigation
            "top_vulnerable_agents": {
                "patterns": [
                    r"\b(most|top|highest|worst)\s+\d*\s*\w*\s*(vulnerable|vuln|cve|risk)\s*(agent|agents|host|endpoint)",
                    r"\b(agent|agents|host|endpoint)\b.*\b(most|highest|top)\b.*\b(vuln|vulnerable|cve|risk)",
                    r"\bwhich\s+agent\b.*\b(most|highest)\b.*\b(vuln|vulnerable|cve)",
                    r"\b(rank|sort)\b.*\bagent\b.*\b(vuln|vulnerable|cve|risk)",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities"],
                "correlation": "top_agents_by_vuln_count",
                "confidence": 0.92,
                "description": "Top agents by vulnerability count"
            },

            "disconnected_with_critical_vulns": {
                "patterns": [
                    r"\bdisconnected\b.*\b(critical|cve|vuln)",
                    r"\b(critical|cve|vuln).*\bdisconnected\b",
                    r"\b(offline|down)\s+agent.*\b(critical|vuln|cve)",
                ],
                "tool": "get_wazuh_agents",
                "args": {"status": "disconnected"},
                "secondary": ["get_wazuh_vulnerabilities"],
                "correlation": "disconnected_agents_with_critical_vulns",
                "confidence": 0.93,
                "description": "Disconnected agents with critical vulnerabilities"
            },

            "active_agents_high_vulns": {
                "patterns": [
                    r"\bactive\s+(agent|agents).*\b(high|critical)\s*(vuln|vulnerabilit|cve)",
                    r"\b(high|critical)\s*(vuln|vulnerabilit|cve).*\bactive\s+(agent|agents)",
                ],
                "tool": "get_wazuh_agents",
                "args": {"status": "active"},
                "secondary": ["get_wazuh_vulnerabilities"],
                "correlation": "active_agents_with_high_vulns",
                "confidence": 0.93,
                "description": "Active agents with high/critical vulnerabilities"
            },

            "compare_vulns_by_status": {
                "patterns": [
                    r"\bcompare\b.*\b(vuln|vulnerabilit|cve).*\b(active|disconnected)",
                    r"\b(vuln|vulnerabilit|cve).*\b(active\s+vs|vs\s+disconnected|active.*disconnected)",
                    r"\b(agent|agents)\b.*\bmore\s+than\s+\d+\s+(vuln|vulnerabilit|cve)",
                    r"\b(exceed|over|above|threshold)\b.*\d+\s*(vuln|vulnerabilit|cve)",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities"],
                "correlation": "compare_vulns_active_vs_disconnected",
                "confidence": 0.94,
                "description": "Compare vulnerability load by agent status"
            },

            "top_agents_by_alerts": {
                "patterns": [
                    r"\bwhich\s+agent\b.*\b(most|highest)\b.*\b(alert|critical\s+alert)",
                    r"\b(most|highest|top)\b.*\b(alert|critical\s+alert).*\b(agent|agents)",
                    r"\b(agent|agents)\b.*\b(most|highest)\b.*\b(alert|critical)",
                    r"\btop\s+(agent|agents)\b.*\balert",
                    r"\b(agent|agents)\b.*\b(alert)\s+(count|volume|frequency)",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_agents"],
                "correlation": "alerts_with_agents",
                "confidence": 0.92,
                "description": "Agents with most/highest alerts"
            },

            "agent_posture_deep_dive": {
                "patterns": [
                    # 2026-05-17: tightened — bare "posture" alone was stealing fim/port/full-investigation queries.
                    # Require explicit deep-dive / comprehensive / full-assessment intent.
                    r"\b(deep\s+dive|full\s+assessment|comprehensive\s+(assessment|review|analysis|posture))\b.*\bagent\b",
                    r"\bagent\b.*\b(deep\s+dive|full\s+assessment|security\s+posture)\b",
                    r"\b(overall|complete|full)\s+(security|risk)\b.*\bagent\s+\d",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_agent_ports", "get_agent_processes"],
                "correlation": "agent_posture_deep_dive",
                "confidence": 0.92,
                "description": "Full agent security posture deep-dive"
            },

            # Compound: alerts + vulnerabilities cross-correlation
            "alerts_with_vulns_compound": {
                "patterns": [
                    r"\b(alert|alerts)\b.*\b(and|with)\b.*\b(vulnerability|vulnerabilities|cve)",
                    r"\b(vulnerability|vulnerabilities|cve)\b.*\b(and|with)\b.*\b(alert|alerts)\b",
                    r"\bagents?\b.*\bboth\b.*\b(alert|alerts)\b.*\b(vulnerability|vulnerabilities)",
                    r"\b(combined|cross).*(alert|alerts).*(vuln|vulnerabilit|cve)",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_agents"],
                "correlation": "alerts_with_vulnerabilities",
                "confidence": 0.91,
                "description": "Alerts cross-referenced with vulnerabilities"
            },

            # Compound: FIM + agent context
            "fim_with_agent_context": {
                "patterns": [
                    r"\b(fim|file\s+integrity)\b.*\b(agent|agents)\b.*\b(context|status|posture|detail)",
                    r"\b(fim|file\s+integrity)\b.*\bwith\b.*\b(agent|agents)\b",
                    r"\b(fim|file\s+integrity)\b.*\b(and|with)\b.*\b(process|processes)\b",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_wazuh_agents", "get_agent_processes"],
                "correlation": "fim_with_agent_posture",
                "confidence": 0.91,
                "description": "FIM events with agent and process context"
            },

            # Compound: MITRE + agent mapping
            "mitre_with_agent_mapping": {
                "patterns": [
                    r"\bmitre\b.*\b(map|mapped|mapping)\b.*\bagents?\b",
                    r"\bmitre\b.*\b(with|per|by)\s+agent",
                    r"\b(att&ck|attack)\b.*\bagent\b.*\b(coverage|mapping)\b",
                ],
                "tool": "get_mitre_coverage",
                "secondary": ["get_wazuh_agents"],
                "correlation": "rule_mitre_with_agents",
                "confidence": 0.91,
                "description": "MITRE ATT&CK coverage mapped to agents"
            },

            "overall_security_posture": {
                "patterns": [
                    r"\b(overall|our|environment)\b.*\b(security|risk)\s+(posture|status|health)",
                    r"\bhow\s+many\s+agents?\b.*\b(active|disconnected|vs)\b",
                    r"\b(security|risk)\s+(posture|overview|status)\b",
                ],
                "tool": "generate_security_report",
                "secondary": [],
                "confidence": 0.88,
                "description": "Overall environment security posture"
            },

            "security_posture_button": {
                # Anchored ^...$ patterns are eligible for the M14a regex
                # shortcut, which fires BEFORE the Qwen Tier-1 router. Without
                # this block, Qwen sees "comprehensive security posture" and
                # routes it to get_wazuh_agents, producing an agent list
                # instead of a real multi-domain SOC report.
                "patterns": [
                    r"^comprehensive\s+security\s+posture\s*$",
                    r"^(?:overall|environment|our)?\s*security\s+posture\s*$",
                    r"^security\s+posture\s+(?:report|overview|summary)\s*$",
                    r"^generate\s+security\s+report\s*$",
                ],
                "tool": "generate_security_report",
                "secondary": [],
                "confidence": 0.97,
                "description": "Security Posture button - anchored shortcut beats Qwen routing"
            },

            "alerts_for_agent": {
                "patterns": [
                    r"\balerts?\b.*\b(for|from|on)\b.*\bagent\s*\d",
                    r"\bagent\s*\d.*\balerts?\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_agents"],
                "correlation": "alerts_with_agents",
                "confidence": 0.93,
                "description": "Alerts for a specific agent"
            },

            # FIX R6: Enterprise SOC Analysis patterns
            "rule_trigger_analysis": {
                "patterns": [
                    r"\b(which|what|top|show)\b.*\brules?\b.*\b(trigger|fire|firing|frequency)",
                    r"\brule\s+(trigger|frequency|firing)\s+(analysis|report|breakdown)",
                    r"\b(noisy|nois)\s+rules?\b",
                    r"\btop\s+\d*\s*(firing|triggered|active)\s+rules?\b",
                    r"\brule\s+(analysis|analytics)\b",
                ],
                "tool": "get_rule_trigger_analysis",
                "secondary": [],
                "confidence": 0.93,
                "description": "Rule trigger frequency analysis"
            },

            "mitre_coverage": {
                "patterns": [
                    r"\bmitre\b.*\b(coverage|mapping|techniques?|tactics?|heatmap|att.?ck|attack)\b",
                    r"\b(att.?ck|attack)\b.*\b(coverage|techniques?|mapping|matrix|tactics?)\b",
                    r"\bdetection\s+(coverage|gap|blind\s+spot)",
                    r"\bwhich\s+mitre\b.*\b(techniques?|tactics?)\b",
                    r"\b(tactics?|techniques?)\b.*\b(mapping|distribution|breakdown|coverage)\b",
                    r"^\s*(show\s+|list\s+|get\s+)?mitre(\s+att.?ck)?\s*(coverage|mapping|rules?|techniques?|tactics?)?\s*$",
                ],
                "tool": "get_mitre_coverage",
                "secondary": [],
                "confidence": 0.93,
                "description": "MITRE ATT&CK coverage mapping (Wazuh alerts) — covers generic phrasing; use 'suricata mitre' for NIDS-only view"
            },

            "alert_timeline": {
                "patterns": [
                    r"\balert\s+(timeline|trend|time\s*series|distribution)",
                    r"\b(hourly|daily)\s+alert\b",
                    r"\balert\s+(spike|spikes|anomal)",
                    r"\balert\s+(activity|volume)\s+(over\s+time|trend)",
                    r"\b(show|get|display)\b.*\balert\b.*\b(timeline|trend)",
                ],
                "tool": "get_alert_timeline",
                "secondary": [],
                "confidence": 0.92,
                "description": "Alert time-series with spike detection"
            },

            "log_source_health": {
                "patterns": [
                    r"\blog\s+(source|ingestion)\s+(health|status|check)",
                    r"\b(silent|quiet|inactive)\s+(agent|source|log)",
                    r"\b(are|which)\b.*\b(agent|source).*\bsending\b",
                    r"\bingestion\s+(health|status|validation|gap)",
                    r"\bsource\s+(health|coverage|status)",
                ],
                "tool": "get_log_source_health",
                "secondary": [],
                "confidence": 0.92,
                "description": "Log ingestion health validation"
            },

            "decoder_analysis": {
                "patterns": [
                    r"\b(decoder|decoders)\s+(analysis|coverage|list|active|status)",
                    r"\b(show|list|get|display)\b.*\b(decoder|decoders)\b",
                    r"\b(active|custom)\s+decoders?\b",
                    r"\bparsing\s+(coverage|analysis|status)",
                ],
                "tool": "get_decoder_analysis",
                "secondary": [],
                "confidence": 0.90,
                "description": "Decoder coverage analysis"
            },

            "fim_events": {
                "patterns": [
                    r"^(show|list|get|display)\s+(me\s+)?fim\b",
                    r"\b(fim|file\s+integrity)\s+(event|events|monitor|check|alert)",
                    r"\bsyscheck\s+(event|events|alert|result)",
                    r"\bfile\s+(change|changes|modified|integrity)\b",
                    r"\b(which|what)\s+files?\b.*\b(changed|modified|added|deleted)",
                ],
                "tool": "get_fim_events",
                "secondary": [],
                "confidence": 0.95,
                "description": "File Integrity Monitoring events"
            },

            "sca_results": {
                "patterns": [
                    r"\bsca\s+(result|results|score|check|assessment)",
                    r"\b(security\s+)?configuration\s+assessment\b",
                    r"\bcis\s+benchmark\b.*\bagent\b",
                    r"\bhardening\s+(status|result|check|score)\b",
                    r"\bsca\b.*\bagent\b",
                ],
                "tool": "get_sca_results",
                "secondary": [],
                "confidence": 0.92,
                "description": "Security Configuration Assessment"
            },

            "agent_inventory": {
                "patterns": [
                    r"\b(system\s+)?inventory\b.*\bagent\b",
                    r"\bagent\b.*\b(inventory|packages|hardware|software)\b",
                    r"\b(installed|what)\s+packages?\b.*\bagent\b",
                    r"\bnetwork\s+interfaces?\b.*\bagent\b",
                    r"\bhardware\s+(info|details|inventory)\b.*\bagent\b",
                ],
                "tool": "get_agent_inventory",
                "secondary": [],
                "confidence": 0.90,
                "description": "Full agent system inventory"
            },

            "rootcheck_results": {
                "patterns": [
                    r"\brootcheck\b.*\b(result|results|scan|agent)\b",
                    r"\b(rootkit|malware)\s+(scan|check|detection|result)",
                    r"\bcheck\b.*\bagent\b.*\b(rootkit|malware)\b",
                    r"\bagent\b.*\brootcheck\b",
                ],
                "tool": "get_rootcheck_results",
                "secondary": [],
                "confidence": 0.92,
                "description": "Rootkit/malware scan results"
            },

            "alert_vuln_correlation": {
                "patterns": [
                    r"\bcorrelate\b.*\b(alert|alerts)\b.*\b(vuln|vulnerabilit)",
                    r"\b(alert|alerts)\b.*\b(vuln|vulnerabilit).*\b(overlap|correlation|cross)",
                    r"\b(which|what)\s+agent\b.*\b(alert|alerts)\b.*\b(vuln|vulnerabilit)",
                    r"\b(high\s+risk|at\s+risk)\s+agents?\b.*\b(active|threat)",
                ],
                "tool": "correlate_alerts_vulnerabilities",
                "secondary": [],
                "confidence": 0.92,
                "description": "Alert-vulnerability cross-correlation"
            },

            "behavioral_baseline": {
                "patterns": [
                    r"\b(behavioral|behaviour)\s+baseline\b",
                    r"\b(is|are)\b.*\balert\s+(activity|level|rate)\b.*\bnormal\b",
                    r"\b(alert|activity)\s+(deviation|anomaly|baseline)\b",
                    r"\b(anomaly|anomalous)\s+(detection|check|analysis)\b",
                    r"\bbaseline\s+(comparison|check|analysis)\b",
                ],
                "tool": "get_behavioral_baseline",
                "secondary": [],
                "confidence": 0.90,
                "description": "Behavioral baseline deviation analysis"
            },

            "decoder_generation": {
                "patterns": [
                    r"\b(generate|create|build|write|make)\s+(a\s+)?(wazuh\s+)?decoder\b",
                    r"\bdecoder\s+(for|from)\s+",
                    r"\b(parse|decode)\s+(this\s+)?(log|line)\b",
                    r"\bcustom\s+decoder\b",
                    r"\bgenerate\s+decoder\b",
                    r"\bdecode\s+(this|the)\s+log\b",
                ],
                "tool": "generate_wazuh_decoder",
                "secondary": [],
                "confidence": 0.93,
                "description": "Generate Wazuh decoder from raw log"
            },

            "hids_explicit": {
                "patterns": [
                    r"\bhids\b",
                    r"\bhost\s+intrusion\s+(detection|prevention)(\s+system)?\b",
                    r"\bhost[\s\-]based\s+(ids|ips|detection|prevention)\b",
                    r"\bhost\s+(ids|ips)\b",
                    r"\bendpoint\s+(?:intrusion\s+(?:detection|prevention)|detection\s+(?:and\s+response|system|engine))\b",
                    r"\b(host|endpoint)\s+(alert|event|detection)s?\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_agents"],
                "correlation": "alerts_with_agents",
                "confidence": 0.95,
                "description": "Host Intrusion Detection (HIDS) -> Wazuh endpoint alerts"
            },

            "alert_investigation": {
                "patterns": [
                    r"^(show|list|get|share|display|give)\s+(me\s+)?(recent\s+)?alerts?",
                    r"^alerts?\s+from\s+(last|past)",
                    r"security\s+(alerts?|incidents?)",
                    r"\b(recent|latest|last)\s+alerts?\b"
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_agents"],
                "correlation": "alerts_with_agents",
                "confidence": 0.90,
                "description": "Security alert investigation"
            },

            "manager_errors_regex": {
                "patterns": [
                    # Typo-tolerant: catches maanger, manger, maneger, managr, manager, etc.
                    r"\bma{1,3}n+[ae]*g?e?r?\b.*\berr",   # maanger/manger/manager/maneger + error
                    r"\berr.*\bma{1,3}n+[ae]*g?e?r?\b",   # error + manager variants
                    r"\bwazuh\b.*\berr",                   # wazuh errors/wazuh error logs
                ],
                "tool": "get_wazuh_manager_error_logs",
                "secondary": [],
                "confidence": 0.88,
                "description": "Manager error logs (typo-tolerant regex)"
            },

            "manager_logs_regex": {
                "patterns": [
                    r"\b(search|find|show|get|list)\b.*\bma+na+ge?r?\b.*\blog",
                    r"\bma+na+ge?r?\b.*\blog.*\b(search|find|show|get|list)\b",
                ],
                "tool": "search_wazuh_manager_logs",
                "secondary": [],
                "confidence": 0.85,
                "description": "Search manager logs (typo-tolerant regex)"
            },

            "cluster_health_regex": {
                "patterns": [
                    r"\bcluster\s+(health|status|state|info|overview)",
                    r"\b(check|show|get|display)\b.*\bcluster\b",
                    r"\bcluster\s+node",
                ],
                "tool": "get_wazuh_cluster_health",
                "secondary": [],
                "confidence": 0.90,
                "description": "Cluster health and status"
            },

            "compliance_check": {
                "patterns": [
                    r"\b(pci|pci-dss)\s+(compliance|check|audit)",
                    r"\b(hipaa)\s+(compliance|check|audit)",
                    r"\b(gdpr)\s+(compliance|check|audit)",
                    r"\b(nist)\s+(compliance|check|audit)",
                    r"run\s+(compliance|regulatory)\s+check",
                    r"\b(share|show|give|get|display|run)\b.*\b(compliance|complaince|complience)\b",
                    r"\b(compliance|complaince|complience)\b.*\b(on|for)\s+agent\b"
                ],
                "tool": "run_compliance_check",
                "secondary": [],
                "confidence": 0.92,
                "description": "Compliance framework check"
            },

            # FIX R5: validate_wazuh_connection
            "validate_connection": {
                "patterns": [
                    r"\b(validate|test|verify)\b.*\b(wazuh\s+)?(connection|connectivity)\b",
                    r"\b(wazuh|connection)\b.*\b(validate|test|verify|check)\b.*\b(connection|status)\b",
                ],
                "tool": "validate_wazuh_connection",
                "secondary": [],
                "confidence": 0.93,
                "description": "Validate Wazuh server connection"
            },

            "system_statistics": {
                "patterns": [
                    r"^(show|get|display|share|list|give)\s+(me\s+)?(wazuh\s+)?(system\s+)?(statistics|stats|metrics)",
                    r"^(overall|system|wazuh)\s+(status|health|overview|statistics|stats)",
                    r"wazuh\s+(system\s+)?(statistics|stats|metrics)",
                    r"(system|wazuh)\s+(statistics|stats|metrics|overview)"
                ],
                "tool": "get_wazuh_statistics",
                "secondary": [],
                "confidence": 0.88,
                "description": "System statistics"
            }
        }

        # ================================================================
        # SURICATA HIGH-CONFIDENCE PATTERNS (only when Suricata is enabled)
        # ================================================================
        if self.suricata_enabled:
            self.HIGH_CONFIDENCE_PATTERNS.update({
                "network_alerts_generic": {
                    "patterns": [
                        r"\bnetwork\s+alerts?\b",
                        r"\bnetwork\s+(event|detection|incident)s?\b",
                        r"\b(nids|ids|ips|nips)\s+(alert|event|detection)s?\b",
                        r"\b(intrusion|signature)\s+(alert|detection|event)s?\b",
                        r"\bnetwork\b.*\balert.*\b(last|past|recent|today|hour|day|week)",
                        r"\b(last|past|recent)\b.*\bnetwork\s+alerts?\b",
                        r"\bnetwork\s+intrusion\s+(detection|prevention)(\s+system)?\b",
                        r"\b(nids|nips)\b",
                        r"\balerts?\s+(are\s+)?network[\s\-]based\b",
                    ],
                    "tool": "get_suricata_alerts",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Network / NIDS / IPS / intrusion alerts -> Suricata"
                },
                "network_traffic_generic": {
                    "patterns": [
                        r"\bnetwork\s+traffic\b",
                        r"\bnetwork\s+(activity|behaviou?r|flow|connection)s?\b",
                        r"\b(packet|flow)\s+(analysis|capture|inspection)\b",
                        r"\btraffic\s+(pattern|anomal\w*|spike|surge)s?\b",
                        r"\b(egress|ingress)\s+traffic\b",
                    ],
                    "tool": "get_suricata_network_analysis",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Network traffic / flow / packet queries -> Suricata network analysis"
                },
                "ids_ips_signature_generic": {
                    "patterns": [
                        r"\b(ids|ips|nids|nips)\s+(signature|rule|detection)s?\b",
                        r"\b(signature|rule)\s+(hit|match|trigger|firing)s?\b",
                        r"\btop\s+(signature|ids\s+rule)s?\b",
                    ],
                    "tool": "get_suricata_top_signatures",
                    "secondary": [],
                    "confidence": 0.94,
                    "description": "IDS/IPS signature queries -> Suricata top signatures"
                },
                # =========================================================
                # NETWORK / IDS SEVERITY SENTINEL
                # ---------------------------------------------------------
                # SINGLE SOURCE OF TRUTH for routing severity-qualified
                # network alerts to the right Suricata bucket. Replaces the
                # old per-phrasing patchwork that left gaps for some word
                # orders and fell through to LLM/safe-fallback.
                #
                # Universal noun list (every term the user might use for
                # the NIDS side of the platform):
                #     network, network intrusion detection,
                #     intrusion detection, nids, ids, ips, nips,
                #     suricata, security data, units
                #
                # Universal severity-word lists (dashboard-aligned):
                #     critical          -> severity 2
                #     high  | alert     -> severity 1
                #     medium | warning  -> severity 3
                #     low   | other     -> severity 4+
                #
                # Each block has TWO universal patterns: noun-then-severity
                # and severity-then-noun, with arbitrary non-sentence
                # filler between them. Confidence 0.99 (above the 0.97
                # severity tier and the 0.95 generic) so this layer ALWAYS
                # wins when both a noun and a severity word are present.
                #
                # The legacy explicit patterns are kept as belt-and-braces
                # in case someone phrases something the sentinel misses.
                # =========================================================
                "network_critical_alerts": {
                    "patterns": [
                        # Universal sentinel — noun ↔ severity in any order
                        r"\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b[^.?!]*?\bcritical\b",
                        r"\bcritical\b[^.?!]*?\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b",
                        # Legacy explicit patterns (back-compat)
                        r"\bcritical\s+network\s+alerts?\b",
                        r"\bnetwork\s+critical\s+alerts?\b",
                        r"\bnetwork\s+alerts?\s+(?:critical|with\s+critical\s+severity)\b",
                        r"\bcritical\s+(nids|ids|ips|nips)\s+alerts?\b",
                        r"\b(nids|ids|ips|nips)\s+critical\s+alerts?\b",
                        r"\bsuricata\s+critical\b",
                        r"\bcritical\s+suricata\b",
                    ],
                    "tool": "get_suricata_critical_alerts",
                    "secondary": [],
                    "confidence": 0.99,
                    "description": "Critical network/IDS alerts -> Suricata critical (severity 2 per dashboard)"
                },
                "network_high_alerts": {
                    "patterns": [
                        # Universal sentinel — noun ↔ severity in any order
                        r"\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b[^.?!]*?\bhigh\b",
                        r"\bhigh\b[^.?!]*?\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b",
                        # Legacy explicit patterns (back-compat)
                        r"\bhigh\s+severity\s+network\s+alerts?\b",
                        r"\bhigh\s+network\s+alerts?\b",
                        r"\bnetwork\s+high\s+alerts?\b",
                        r"\bnetwork\s+alerts?\s+(?:high|with\s+high\s+severity)\b",
                        r"\b(nids|ids|ips|nips)\s+high\s+alerts?\b",
                        r"\bsuricata\s+high\b",
                        r"\bhigh\s+suricata\b",
                        # Dashboard label "Alert" tier (severity=1) — only with disambiguator
                        r"\b(?:alert|alerts)\s+(?:tier|level\s+1)\s+(?:network|nids|ids|ips|suricata)\b",
                    ],
                    "tool": "get_suricata_high_alerts",
                    "secondary": [],
                    "confidence": 0.99,
                    "description": "High severity / 'Alert' tier network/IDS alerts -> Suricata high (severity 1 per dashboard)"
                },
                "network_medium_alerts": {
                    "patterns": [
                        # Universal sentinel — medium OR dashboard-label "warning"
                        r"\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b[^.?!]*?\b(?:medium|warning)\b",
                        r"\b(?:medium|warning)\b[^.?!]*?\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b",
                        # Legacy explicit patterns (back-compat)
                        r"\bmedium\s+severity\s+network\s+alerts?\b",
                        r"\bmedium\s+network\s+alerts?\b",
                        r"\bnetwork\s+medium\s+alerts?\b",
                        r"\bnetwork\s+alerts?\s+(?:medium|with\s+medium\s+severity)\b",
                        r"\b(nids|ids|ips|nips)\s+medium\s+alerts?\b",
                        r"\bsuricata\s+medium\b",
                        r"\bmedium\s+suricata\b",
                        r"\bsuricata\s+warning\b",
                    ],
                    "tool": "get_suricata_medium_alerts",
                    "secondary": [],
                    "confidence": 0.99,
                    "description": "Medium severity / 'Warning' tier network/IDS alerts -> Suricata medium (severity 3 per dashboard)"
                },
                "network_low_alerts": {
                    "patterns": [
                        # Universal sentinel — low OR dashboard-label "other"
                        r"\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b[^.?!]*?\b(?:low|other)\b",
                        r"\b(?:low|other)\b[^.?!]*?\b(?:network(?:\s+intrusion\s+detection)?|intrusion\s+detection|nids|ids|ips|nips|suricata|security\s+data|units)\b",
                        # Legacy explicit patterns (back-compat)
                        r"\blow\s+severity\s+network\s+alerts?\b",
                        r"\blow\s+network\s+alerts?\b",
                        r"\bnetwork\s+low\s+alerts?\b",
                        r"\bnetwork\s+alerts?\s+(?:low|with\s+low\s+severity)\b",
                        r"\b(nids|ids|ips|nips)\s+low\s+alerts?\b",
                        r"\bsuricata\s+low\b",
                        r"\blow\s+suricata\b",
                        r"\bsuricata\s+other\b",
                    ],
                    "tool": "get_suricata_low_alerts",
                    "secondary": [],
                    "confidence": 0.99,
                    "description": "Low severity / 'Other' tier network/IDS alerts -> Suricata low (severity >= 4 per dashboard)"
                },
                "suricata_alerts_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\balerts?\b",
                        r"\bids\s+alerts?\b",
                        r"\bsuricata\b.*\b(detection|event)",
                        r"\bnids\b.*\balerts?\b",
                    ],
                    "tool": "get_suricata_alerts",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Suricata IDS alerts"
                },
                "suricata_alert_summary_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\b(summary|overview|breakdown)\b",
                        r"\bsuricata\b.*\bstatistic",
                        r"\b(ids|alert)\b.*\bsummary\b",
                    ],
                    "tool": "get_suricata_alert_summary",
                    "secondary": [],
                    "confidence": 0.96,
                    "description": "Suricata alert summary"
                },
                "suricata_critical_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bcritical\b",
                        r"\bcritical\b.*\bsuricata\b",
                        r"\bcritical\b.*\bids\b",
                    ],
                    "tool": "get_suricata_critical_alerts",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Critical Suricata alerts"
                },
                "suricata_high_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bhigh\b.*\b(alert|severity)\b",
                        r"\bhigh\b.*\bsuricata\b.*\balert",
                        r"\bhigh\s+severity\b.*\bsuricata\b",
                    ],
                    "tool": "get_suricata_high_alerts",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "High severity Suricata alerts"
                },
                "suricata_network_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bnetwork\b",
                        r"\bsuricata\b.*\b(top\s+talker|traffic|connection)",
                        r"\bnetwork\s+analysis\b",
                        r"\b(top\s+talker|traffic\s+analysis)\b",
                    ],
                    "tool": "get_suricata_network_analysis",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Suricata network analysis"
                },
                "suricata_search_explicit": {
                    "patterns": [
                        r"\bsearch\b.*\bsuricata\b",
                        r"\bsuricata\b.*\bsearch\b",
                        r"\b(find|look\s+for)\b.*\bsuricata\b",
                    ],
                    "tool": "search_suricata_alerts",
                    "secondary": [],
                    "confidence": 0.92,
                    "description": "Search Suricata alerts"
                },
                "suricata_signatures_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bsignature",
                        r"\bids\s+signature",
                        r"\b(top|most)\b.*\bsignature",
                        r"\bsignature\b.*\b(top|most|frequent)\b",
                    ],
                    "tool": "get_suricata_top_signatures",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Top Suricata signatures"
                },
                "suricata_attackers_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\b(attacker|source\s+ip|top\s+source)",
                        r"\btop\s+attacker",
                        r"\b(attacker|attacking)\b.*\bip\b",
                    ],
                    "tool": "get_suricata_top_attackers",
                    "secondary": [],
                    "confidence": 0.96,
                    "description": "Top Suricata attackers"
                },
                "suricata_health_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\b(health|status)\b",
                        r"\bids\s+(health|status)\b",
                    ],
                    "tool": "get_suricata_health",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Suricata health check"
                },
                "suricata_categories_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bcategor",
                        r"\bids\b.*\bcategor",
                        r"\bsuricata\b.*\b(type|class)\b.*\balert",
                    ],
                    "tool": "get_suricata_category_breakdown",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Suricata category breakdown"
                },
                # Cross-platform correlation patterns
                "combined_alerts_explicit": {
                    "patterns": [
                        r"\bcombined\b.*\balert",
                        r"\b(all|both|unified)\b.*\b(platform|system)\b.*\balert",
                        r"\bwazuh\b.*\bsuricata\b.*\balert",
                        r"\bsuricata\b.*\bwazuh\b.*\balert",
                        r"\bcorrelat\w*\b.*\bwazuh\b.*\bsuricata\b",
                        r"\bwazuh\b.*\bsuricata\b.*\b(detection|correlat)",
                    ],
                    "tool": "get_suricata_alerts",
                    "secondary": ["get_wazuh_alerts"],
                    "correlation": "combined_alert_view",
                    "confidence": 0.93,
                    "description": "Combined Wazuh + Suricata alerts"
                },
                "comprehensive_posture_with_suricata": {
                    "patterns": [
                        r"\b(comprehensive|full|complete)\b.*\b(security|posture)\b.*\b(environment|all)\b",
                        r"\benvironment\b.*\b(security|posture|overview)\b",
                    ],
                    "tool": "get_wazuh_agents",
                    "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts", "get_suricata_alerts"],
                    "correlation": "comprehensive_security_posture",
                    "confidence": 0.92,
                    "description": "Comprehensive security posture with Suricata"
                },
                # Suricata Deep Visibility patterns (HTTP, TLS, MITRE, JA3)
                "suricata_http_analysis_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bhttp\b",
                        r"\bhttp\b.*\b(traffic|analysis)\b",
                        r"\b(web|http)\b.*\b(traffic|request|analysis)\b",
                        r"\b(show|get)\b.*\bhttp\b.*\b(traffic|event|analysis)\b",
                    ],
                    "tool": "get_suricata_http_analysis",
                    "secondary": [],
                    "confidence": 0.94,
                    "description": "Suricata HTTP traffic analysis"
                },
                "suricata_http_search_explicit": {
                    "patterns": [
                        r"\bsearch\b.*\bhttp\b",
                        r"\b(find|look)\b.*\b(url|http)\b",
                        r"\bhttp\b.*\bsearch\b",
                        r"\b(search|find)\b.*\b(url|web\s+request)\b",
                    ],
                    "tool": "search_suricata_http",
                    "secondary": [],
                    "confidence": 0.92,
                    "description": "Search Suricata HTTP events"
                },
                "suricata_tls_analysis_explicit": {
                    "patterns": [
                        r"\b(tls|ssl)\b.*\b(analysis|version|fingerprint|certificate)\b",
                        r"\bsuricata\b.*\b(tls|ssl)\b",
                        r"\b(show|get|analyze)\b.*\b(tls|ssl)\b",
                    ],
                    "tool": "get_suricata_tls_analysis",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Suricata TLS/SSL analysis"
                },
                "suricata_mitre_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bmitre\b",
                        r"\bids\b.*\bmitre\b",
                        r"\bnids\b.*\bmitre\b",
                        r"\bmitre\b.*\b(suricata|ids|nids)\b",
                    ],
                    "tool": "get_suricata_mitre_mapping",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Suricata MITRE ATT&CK mapping (requires explicit Suricata/IDS/NIDS keyword; generic 'mitre coverage' goes to Wazuh)"
                },
                "suricata_ja3_explicit": {
                    "patterns": [
                        r"\bja3\b",
                        r"\bja3s\b",
                        r"\bja4\b",
                        r"\btls\s+fingerprint",
                        r"\b(client|server)\s+fingerprint\b",
                    ],
                    "tool": "get_suricata_ja3_fingerprints",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "JA3/JA3S/JA4 fingerprint analysis"
                },
                "suricata_suspicious_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\bsuspicious\b",
                        r"\b(scanner|scanning|recon)\b.*\b(detect|activity)\b",
                        r"\bsuspicious\b.*\b(http|traffic|activity|user.agent)\b",
                        r"\b(detect|find)\b.*\b(scanner|suspicious|anomal)\b",
                    ],
                    "tool": "get_suricata_suspicious_activity",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "Suspicious activity detection"
                },
                "suricata_traffic_overview_explicit": {
                    "patterns": [
                        r"\bsuricata\b.*\b(traffic|event)\b.*\b(overview|distribution|breakdown)\b",
                        r"\b(traffic|event\s+type)\b.*\b(overview|distribution)\b",
                        r"\bsuricata\b.*\b(event\s+type|protocol)\b.*\b(breakdown|distribution)\b",
                    ],
                    "tool": "get_suricata_traffic_overview",
                    "secondary": [],
                    "confidence": 0.92,
                    "description": "Suricata traffic overview"
                },
                # Cross-platform MITRE correlation
                "wazuh_suricata_mitre_unified_explicit": {
                    "patterns": [
                        r"\b(unified|combined|all)\b.*\bmitre\b",
                        r"\bmitre\b.*\b(both|all|combined|unified)\b",
                        r"\b(wazuh|all)\b.*\bsuricata\b.*\bmitre\b",
                    ],
                    "tool": "get_suricata_mitre_mapping",
                    "secondary": ["get_wazuh_alerts"],
                    "correlation": "wazuh_suricata_mitre_unified",
                    "confidence": 0.93,
                    "description": "Unified MITRE ATT&CK from Wazuh + Suricata"
                },
                "suricata_recon_detect_explicit": {
                    "patterns": [
                        r"\b(reconnaissance|recon|scanning)\b.*\bdetect",
                        r"\bdetect\b.*\b(scanner|recon)\b",
                        r"\b(network|web)\b.*\b(recon|scanning|enumeration)\b",
                    ],
                    "tool": "get_suricata_http_analysis",
                    "secondary": ["get_suricata_alerts", "get_suricata_network_analysis"],
                    "correlation": "suricata_recon_detection",
                    "confidence": 0.92,
                    "description": "Reconnaissance detection"
                },
                "dns_tunneling_explicit": {
                    "patterns": [
                        r"\bdns\s+(tunnel|tunneling|exfil|beacon)\w*",
                        r"\b(tunnel|beacon)\w*\s+.*\bdns\b",
                        r"\bdns\s+(anomal|suspicious|unusual)\w*",
                        r"\bcovert\s+dns\b",
                    ],
                    "tool": "get_suricata_suspicious_activity",
                    "secondary": [],
                    "confidence": 0.94,
                    "description": "DNS tunneling / exfiltration / beaconing -> Suricata"
                },
                "port_scan_explicit": {
                    "patterns": [
                        r"\bport\s+scan(ning|ner)?s?\b",
                        r"\bscan(ning)?\s+(for\s+)?(open\s+)?ports?\b",
                        r"\b(nmap|masscan|zmap|unicornscan)\s+(scan|activity|detect)",
                        r"\b(host|network)\s+(recon|enumeration|discovery)\b",
                        r"\bsweep\s+(scan|activity)\b",
                    ],
                    "tool": "get_suricata_suspicious_activity",
                    "secondary": ["get_suricata_top_attackers"],
                    "confidence": 0.94,
                    "description": "Port scan / network reconnaissance -> Suricata"
                },
                # Pallas 3.2: Universal Query Coverage patterns
                "cross_platform_ip_explicit": {
                    "patterns": [
                        r"\bsame\b.*\bip\b.*\b(wazuh|suricata|both)\b",
                        r"\bcross.platform\b.*\bip\b",
                        r"\bip\b.*\b(both|all)\b.*\bplatform\b",
                        r"\bcorrelat.*\bip\b.*\b(across|between)\b",
                    ],
                    "tool": "get_suricata_top_attackers",
                    "secondary": ["get_wazuh_alerts", "get_wazuh_agents"],
                    "correlation": "cross_platform_ip_correlation",
                    "confidence": 0.93,
                    "description": "Cross-platform IP correlation"
                },
                "unified_scan_detection_explicit": {
                    "patterns": [
                        r"\bwho\b.*\bscanning\b",
                        r"\ball\b.*\bscan.*\bactivity\b",
                        r"\b(detect|find)\b.*\b(all|every)\b.*\bscan",
                        r"\bscanner\b.*\b(all|unified|both)\b",
                    ],
                    "tool": "get_suricata_suspicious_activity",
                    "secondary": ["get_suricata_top_attackers", "get_wazuh_alerts"],
                    "correlation": "unified_scanning_detection",
                    "confidence": 0.93,
                    "description": "Unified scanning detection"
                },
                "attack_chain_explicit": {
                    "patterns": [
                        r"\b(attack|kill)\s*(chain|progression)\b",
                        # 2026-05-17: removed "sequence" — it belongs to temporal_attack_sequence.
                        r"\battack\b.*\b(stages|path|chain)\b",
                        r"\bkill\s*chain\b",
                        r"\battack\b.*\bprogression\b",
                    ],
                    "tool": "get_suricata_mitre_mapping",
                    "secondary": ["get_wazuh_alerts", "get_suricata_alerts"],
                    "correlation": "attack_chain_analysis",
                    "confidence": 0.94,
                    "description": "Kill chain / attack progression analysis"
                },
                "exploit_correlation_explicit": {
                    "patterns": [
                        r"\b(vuln|vulnerabilit)\b.*\b(exploit|target|attack)\b",
                        r"\bactively\b.*\b(exploit|target)\b",
                        r"\b(exploit|attack)\b.*\b(vuln|cve)\b",
                        r"\btarget\b.*\b(vulnerabilit|cve)\b",
                    ],
                    "tool": "get_wazuh_vulnerabilities",
                    "secondary": ["get_suricata_alerts", "get_wazuh_alerts"],
                    "correlation": "vulnerability_exploit_correlation",
                    "confidence": 0.93,
                    "description": "Vulnerability exploit correlation"
                },
                "detection_gap_explicit": {
                    "patterns": [
                        r"\b(detection|coverage)\b.*\b(gap|blind|missing)\b",
                        r"\bblind\s*spot",
                        r"\b(mitre|att.ck)\b.*\bgap\b",
                        r"\bwhat\b.*\b(not|missing)\b.*\bdetect",
                    ],
                    "tool": "get_mitre_coverage",
                    "secondary": ["get_suricata_mitre_mapping"],
                    "correlation": "detection_coverage_gap",
                    "confidence": 0.93,
                    "description": "MITRE detection coverage gap analysis"
                },
                "port_risk_explicit": {
                    "patterns": [
                        r"\b(exposed|open)\b.*\b(port|service)\b.*\b(risk|target|attack)\b",
                        r"\bport\b.*\b(risk|exposure|target)\b",
                        r"\bwhich\b.*\bport\b.*\b(target|attack|risk)\b",
                        r"\bservice\b.*\b(exposed|at.risk|target)\b",
                    ],
                    "tool": "get_agent_ports",
                    "secondary": ["get_suricata_alerts", "get_wazuh_vulnerabilities"],
                    "correlation": "port_exposure_risk",
                    "confidence": 0.92,
                    "description": "Port exposure risk assessment"
                },
                # --- Pallas 3.3: New Suricata correlation patterns ---
                "suricata_attacker_target_map_explicit": {
                    "patterns": [
                        r"\b(attacker|source)\b.*\b(target|victim|destination)\b.*\b(map|mapping)\b",
                        r"\b(map|show)\b.*\b(attacker|source)\b.*\b(target|dest)\b",
                        r"\bwho\b.*\b(attacking|targeting)\b.*\b(whom|who|what)\b",
                        r"\battacker\b.*\btarget\b",
                    ],
                    "tool": "get_suricata_alerts",
                    "secondary": ["get_suricata_network_analysis"],
                    "correlation": "suricata_attacker_target_map",
                    "confidence": 0.93,
                    "description": "Suricata attacker-to-target mapping"
                },
                "suricata_sig_severity_explicit": {
                    "patterns": [
                        r"\bsignature\b.*\bseverity\b",
                        r"\bseverity\b.*\bsignature\b",
                        r"\bwhich\b.*\bsignature\b.*\b(critical|dangerous)\b",
                        r"\bsignature\b.*\b(ranking|rank|severity)\b",
                    ],
                    "tool": "get_suricata_alert_summary",
                    "secondary": ["get_suricata_critical_alerts"],
                    "correlation": "suricata_signature_severity_analysis",
                    "confidence": 0.92,
                    "description": "Signature severity cross-reference"
                },
                "suricata_network_threat_explicit": {
                    "patterns": [
                        r"\bnetwork\b.*\bthreat\b.*\bprofile\b",
                        r"\bthreat\b.*\bnetwork\b.*\bprofile\b",
                        r"\bnetwork\b.*\b(threat|risk)\b.*\b(analysis|overview)\b",
                    ],
                    "tool": "get_suricata_network_analysis",
                    "secondary": ["get_suricata_alert_summary"],
                    "correlation": "suricata_network_threat_profile",
                    "confidence": 0.92,
                    "description": "Network threat profile"
                },
                "agents_suricata_detections_explicit": {
                    "patterns": [
                        r"\bagent\b.*\bsuricata\b.*\b(detection|alert)\b",
                        r"\bsuricata\b.*\b(detection|alert)\b.*\bagent\b",
                        r"\bwhich\b.*\bagent\b.*\b(ids|suricata)\b.*\bdetect",
                        r"\blink\b.*\bagent\b.*\b(ids|suricata|network)\b",
                        r"\bagent\b.*\b(ids|network)\b.*\bdetection\b",
                    ],
                    "tool": "get_wazuh_agents",
                    "secondary": ["get_suricata_alerts"],
                    "correlation": "wazuh_agents_suricata_detections",
                    "confidence": 0.93,
                    "description": "Agent-to-Suricata detection mapping"
                },
                # --- Pallas 3.3: JA4 and Flow tools ---
                "suricata_ja4_explicit": {
                    "patterns": [
                        r"\bja4\b.*\b(analysis|deep|detail|fingerprint)\b",
                        r"\bja4\b.*\b(lookup|investigat)\b",
                        r"\b(analyze|investigat)\b.*\bja4\b",
                        r"\bshow\b.*\bja4\b",
                    ],
                    "tool": "get_suricata_ja4_analysis",
                    "secondary": [],
                    "confidence": 0.93,
                    "description": "JA4 fingerprint deep analysis"
                },
                "suricata_flow_explicit": {
                    "patterns": [
                        r"\b(flow|conversation)\b.*\b(analysis|pattern)\b",
                        r"\b(ip\s+pair|communication)\b.*\b(pattern|analysis)\b",
                        r"\b(network|alert)\b.*\bconversation\b",
                        r"\b(who|which)\b.*\b(talk|communicat)\w*\b.*\b(with|to)\b",
                        r"\bshow\b.*\b(flow|conversation)\b",
                    ],
                    "tool": "get_suricata_flow_analysis",
                    "secondary": [],
                    "confidence": 0.92,
                    "description": "Network flow/conversation analysis"
                },
            })

        # Pallas 2.4.1: Cloudflare patterns (always active)
        self.HIGH_CONFIDENCE_PATTERNS.update({
            "cloudflare_http_summary": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+http\s+(?:summary|overview|traffic)\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:summary|overview|traffic)\s*$",
                ],
                "tool": "get_cloudflare_http_summary",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare HTTP traffic summary",
            },
            "cloudflare_http_errors": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:http\s+)?(?:errors|4xx|5xx|failures)\s*$",
                ],
                "tool": "get_cloudflare_http_errors",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare HTTP errors",
            },
            "cloudflare_firewall_events": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:firewall|waf)(?:\s+events)?\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+blocks?\s*$",
                ],
                "tool": "get_cloudflare_firewall_events",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare firewall / WAF events",
            },
            "cloudflare_firewall_top_attackers": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+attackers\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:top\s+)?(?:attacker|blocked)\s+ips\s*$",
                ],
                "tool": "get_cloudflare_firewall_top_attackers",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top attackers",
            },
            "cloudflare_firewall_top_rules": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+(?:firewall\s+)?rules\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:waf|firewall)\s+rules\s*$",
                ],
                "tool": "get_cloudflare_firewall_top_rules",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top firewall rules",
            },
            "cloudflare_security_summary": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+security(?:\s+summary)?\s*$",
                ],
                "tool": "get_cloudflare_security_summary",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare security summary",
            },
            "cloudflare_http_top_paths": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+paths\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+urls\s*$",
                ],
                "tool": "get_cloudflare_http_top_paths",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top HTTP paths",
            },
            "cloudflare_http_top_clients": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+(?:clients|countries|sources)\s*$",
                ],
                "tool": "get_cloudflare_http_top_clients",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top HTTP clients",
            },
            "cloudflare_cache_performance": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+cache(?:\s+performance|\s+ratio|\s+stats)?\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+hit\s+ratio\s*$",
                ],
                "tool": "get_cloudflare_cache_performance",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare cache performance",
            },
            "cloudflare_cache_top_misses": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+(?:top\s+)?cache\s+misses?\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+bypass(?:es)?\s*$",
                ],
                "tool": "get_cloudflare_cache_top_misses",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top cache misses",
            },
            "cloudflare_dns_summary": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+dns(?:\s+summary|\s+overview)?\s*$",
                ],
                "tool": "get_cloudflare_dns_summary",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare DNS summary",
            },
            "cloudflare_dns_top_queries": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+dns(?:\s+(?:queries|domains))?\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+top\s+domains\s*$",
                ],
                "tool": "get_cloudflare_dns_top_queries",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare top DNS queries",
            },
            "cloudflare_dns_errors": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+dns\s+errors\s*$",
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+nxdomain\s*$",
                ],
                "tool": "get_cloudflare_dns_errors",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare DNS errors",
            },
            "cloudflare_workers_summary": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+workers(?:\s+summary|\s+overview)?\s*$",
                ],
                "tool": "get_cloudflare_workers_summary",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare Workers summary",
            },
            "cloudflare_workers_errors": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+workers?\s+errors\s*$",
                ],
                "tool": "get_cloudflare_workers_errors",
                "secondary": [],
                "confidence": 0.96,
                "description": "Cloudflare Workers errors",
            },
            # Generic-phrasing catch-alls — land on the security summary (most
            # comprehensive single dashboard) and pull in HTTP/cache/DNS as
            # secondaries so a single ask like "analyze cloudflare" produces a
            # multi-source briefing.
            "cloudflare_generic_status": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)(?:\s+(?:status|overview|health|state))?\s*$",
                    r"^(?:analy[sz]e|assess|investigate|check)\s+(?:cloudflare|cf)\s*$",
                    r"^(?:how\s+is|whats?(?:\'s)?\s+(?:happening\s+(?:on|with)|going\s+on\s+(?:with|on))?\s+)?(?:cloudflare|cf)(?:\s+doing|\s+looking)?\s*\??$",
                    r"^(?:tell|give)\s+me\s+(?:about|the)\s+(?:cloudflare|cf)(?:\s+(?:status|overview|situation))?\s*$",
                ],
                "tool": "get_cloudflare_security_summary",
                "secondary": [
                    "get_cloudflare_http_summary",
                    "get_cloudflare_firewall_events",
                    "get_cloudflare_cache_performance",
                ],
                "correlation": None,
                "confidence": 0.95,
                "description": "Generic Cloudflare status / analysis → security summary + secondaries",
            },
            "cloudflare_traffic_generic": {
                "patterns": [
                    r"^(?:show\s+)?(?:cloudflare|cf)\s+traffic\s*$",
                    r"^(?:cloudflare|cf)\s+http\s*$",
                    r"^(?:cloudflare|cf)\s+requests?\s*$",
                ],
                "tool": "get_cloudflare_http_summary",
                "secondary": [],
                "confidence": 0.96,
                "description": "Generic Cloudflare HTTP traffic ask",
            },
        })

        # Pallas 3.2: Non-Suricata universal patterns (always active)
        self.HIGH_CONFIDENCE_PATTERNS.update({
            "ip_investigation_explicit": {
                "patterns": [
                    r"\binvestigat.*\bip\b",
                    r"\bwhat\b.*\bknow\b.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                    r"\blookup\b.*\bip\b",
                    r"\bip\b.*\b(investigation|pivot|lookup)\b",
                    r"\binvestigat.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts", "search_suricata_http", "get_wazuh_agents"] if self.suricata_enabled else ["get_wazuh_agents"],
                "correlation": "ip_investigation_pivot",
                "confidence": 0.93,
                "description": "Cross-source IP investigation"
            },
            "full_agent_investigation_explicit": {
                "patterns": [
                    r"\b(full|complete|thorough)\b.*\b(investigation|assessment|analysis)\b.*\bagent\b",
                    r"\binvestigat.*\bagent\b.*\b(fully|completely|thorough)\b",
                    r"\bagent\b.*\b(full|complete|thorough)\b.*\b(investigation|assessment|analysis)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_agent_ports", "get_agent_processes",
                               "get_wazuh_alerts", "get_fim_events"],
                "correlation": "full_agent_investigation",
                "confidence": 0.93,
                "description": "Full agent security investigation"
            },
            "threat_priority_explicit": {
                "patterns": [
                    r"\bwhat\b.*\b(worry|priorit|urgent)\b",
                    r"\b(biggest|top|main)\b.*\b(risk|threat|concern)\b",
                    r"\bprioritize\b.*\b(threat|risk|alert)\b",
                    r"\bwhat\b.*\bshould\b.*\b(focus|priorit|worry)\b",
                    r"\btop\b.*\bpriorities\b",
                ],
                "tool": "get_wazuh_alert_summary",
                "secondary": (["get_suricata_alert_summary", "get_wazuh_vulnerability_summary",
                                "get_log_source_health"] if self.suricata_enabled else
                               ["get_wazuh_vulnerability_summary", "get_log_source_health"]),
                "correlation": "unified_threat_summary",
                "confidence": 0.92,
                "description": "Prioritized threat summary"
            },
            "top_risk_agents_explicit": {
                "patterns": [
                    r"\b(most|highest)\b.*\b(risk|danger|critical)\b.*\bagent\b",
                    r"\bagent\b.*\b(risk|danger)\b.*\b(rank|score|rating)\b",
                    r"\bwhich\b.*\bagent\b.*\b(need|requir)\b.*\battention\b",
                    r"\b(riskiest|most.vulnerable)\b.*\bagent\b",
                    r"\btop\b.*\brisk\b.*\bagent",
                    r"\b(identify|find|show)\b.*\b(top|highest)\b.*\brisk\b.*\bagent",
                ],
                "tool": "get_wazuh_agents",
                "secondary": (["get_wazuh_vulnerabilities", "get_wazuh_alerts", "get_suricata_alerts"]
                               if self.suricata_enabled else
                               ["get_wazuh_vulnerabilities", "get_wazuh_alerts"]),
                "correlation": "top_risk_agents_composite",
                "confidence": 0.93,
                "description": "Agent risk ranking"
            },
            "fim_alert_correlation_explicit": {
                "patterns": [
                    r"\b(file|fim)\b.*\bchang.*\b(why|alert|cause)\b",
                    r"\bfile\b.*\bmodif.*\balert\b",
                    r"\b(syscheck|fim)\b.*\b(correlat|relat).*\balert\b",
                    r"\bwhat\b.*\bfile\b.*\bchang",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents"],
                "correlation": "fim_alert_correlation",
                "confidence": 0.92,
                "description": "FIM-alert correlation"
            },
            "alert_enrichment_explicit": {
                "patterns": [
                    r"\bexplain\b.*\balert\b",
                    r"\balert\b.*\b(context|detail|enrich|deep)\b",
                    r"\binvestigat.*\balert\b",
                    r"\balert\b.*\b(investigation|enrichment)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_wazuh_agents", "get_wazuh_vulnerabilities", "get_suricata_alerts"]
                               if self.suricata_enabled else
                               ["get_wazuh_agents", "get_wazuh_vulnerabilities"]),
                "correlation": "alert_context_enrichment",
                "confidence": 0.92,
                "description": "Alert context enrichment"
            },
            "brute_force_explicit": {
                "patterns": [
                    r"\bbrute\s*force\b",
                    r"\bpassword\b.*\b(spray|guess|crack)\b",
                    r"\b(login|auth)\b.*\b(fail|attempt|brute)\b",
                    r"\bcredential\b.*\b(stuff|brute|attack)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "confidence": 0.92,
                "description": "Brute force / credential attack detection"
            },
            "login_failure_explicit": {
                "patterns": [
                    r"\b(login|logon|sign.in)\b.*\b(fail|failure|denied)\b",
                    r"\bfailed\b.*\b(login|auth|logon)\b",
                    r"\baccess\b.*\bdenied\b",
                    r"\bunauthorized\b.*\b(access|login|attempt)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "login failed OR authentication failure"},
                "confidence": 0.91,
                "description": "Login failure detection"
            },
            # --- Pallas 3.4: Windows Security Event patterns ---
            "privilege_escalation_explicit": {
                "patterns": [
                    r"\bprivilege\s*(escalat|elevat)",
                    r"\b(priv\s*esc|privesc)\b",
                    r"\bevent[\.\s]*id[\s:]*4672\b",
                    r"\bevent[\.\s]*id[\s:]*4673\b",
                    r"\b(token|impersonat).*\b(manipulat|escalat|elevat)",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "privilege escalation OR event.id:4672 OR event.id:4673 OR SeDebugPrivilege"},
                "confidence": 0.92,
                "description": "Privilege escalation detection"
            },
            "powershell_suspicious_explicit": {
                "patterns": [
                    r"\bpowershell\b.*\b(suspicious|malicious|obfuscat|attack|unusual)\b",
                    r"\b(suspicious|malicious|obfuscat|attack)\b.*\bpowershell\b",
                    r"\bpowershell\s+execut\b",
                    r"\bpowershell\s+activit\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "powershell OR rule.groups:powershell"},
                "confidence": 0.92,
                "description": "Suspicious PowerShell activity detection"
            },
            "encoded_powershell_explicit": {
                "patterns": [
                    r"\bencoded\b.*\b(powershell|command|script)\b",
                    r"\bpowershell\b.*\b(encoded|base64|obfuscat)\b",
                    r"\b(base64|b64)\b.*\b(command|script|execution)\b",
                    r"\bpowershell\b.*\b-enc\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "encoded command OR base64 OR powershell -enc OR powershell -encodedcommand"},
                "confidence": 0.93,
                "description": "Encoded PowerShell / base64 command detection"
            },
            "scheduled_task_explicit": {
                "patterns": [
                    r"\bscheduled\s+task\b",
                    r"\bschtasks\b",
                    r"\bevent[\.\s]*id[\s:]*4698\b",
                    r"\b(task|cron)\s*(creat|persist|suspicious)\b",
                    r"\b(suspicious|malicious|unauthorized)\b.*\b(task|cron|job)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "scheduled task OR schtasks OR event.id:4698 OR event.id:4699"},
                "confidence": 0.92,
                "description": "Suspicious scheduled task detection"
            },
            "service_creation_explicit": {
                "patterns": [
                    r"\b(new|abnormal|suspicious|unauthorized)\s+service\b",
                    r"\bservice\s*(creat|install|modif)\b",
                    r"\bevent[\.\s]*id[\s:]*7045\b",
                    r"\bevent[\.\s]*id[\s:]*4697\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "new service OR service install OR event.id:7045 OR event.id:4697"},
                "confidence": 0.92,
                "description": "Abnormal service creation/installation"
            },
            "command_execution_explicit": {
                "patterns": [
                    r"\bcommand\s+(execution|line)\b.*\b(suspicious|alert|unusual|detect)\b",
                    r"\b(suspicious|unusual|malicious)\b.*\bcommand\s*(execution|line)\b",
                    r"\bcmd\.exe\b.*\b(suspicious|alert|unusual)\b",
                    r"\b(wmic|mshta|regsvr32|certutil)\b.*\b(execution|alert|suspicious)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "command execution OR cmd.exe OR rule.groups:command_monitoring OR wmic OR mshta"},
                "confidence": 0.91,
                "description": "Suspicious command execution detection"
            },
            "malware_detection_explicit": {
                "patterns": [
                    r"\bmalware\b.*\b(detect|alert|found|event)\b",
                    r"\b(virus|trojan|worm|rootkit)\b.*\b(detect|alert|found)\b",
                    r"\b(detect|alert|found)\b.*\b(malware|virus|trojan|worm|rootkit)\b",
                    r"\b(antivirus|av|endpoint\s+protection)\b.*\balert\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "malware OR virus OR trojan OR ransomware OR rootkit"},
                "confidence": 0.92,
                "description": "Malware detection alerts"
            },
            "lateral_movement_explicit": {
                "patterns": [
                    r"\blateral\s+movement\b",
                    r"\bpass\s+the\s+(hash|ticket)\b",
                    r"\b(remote|wmi|psexec)\s*(execution|command|lateral)\b",
                    r"\b(pivot|spread)\b.*\b(network|system|host)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "lateral movement OR pass the hash OR remote execution OR psexec OR wmi remote"},
                "confidence": 0.92,
                "description": "Lateral movement detection"
            },
            "suspicious_login_location_explicit": {
                "patterns": [
                    r"\b(suspicious|unusual|abnormal|anomal)\b.*\b(login|logon|sign.?in)\b.*\b(location|source|origin)\b",
                    r"\b(new|unknown|unusual)\b.*\b(logon|login)\b.*\b(type|source)\b",
                    r"\blogon\s+type\s+10\b",
                    r"\bremote\s+(login|logon|desktop)\b.*\b(suspicious|unusual|new|unexpected)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "new logon OR logon type 10 OR remote login OR remote desktop"},
                "confidence": 0.91,
                "description": "Suspicious login location/type detection"
            },
            "multi_alert_host_explicit": {
                "patterns": [
                    r"\b(multiple|many|several|repeat)\b.*\balert\b.*\b(host|agent|endpoint|system)\b",
                    r"\b(host|agent|endpoint)\b.*\b(multiple|many|several)\b.*\balert\b",
                    r"\bhigh\s+alert\s*(volume|count|frequency)\b.*\b(agent|host)\b",
                    r"\b(noisy|chatty|alert.heavy)\b.*\b(agent|host|endpoint)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "security event"},
                "confidence": 0.90,
                "description": "Hosts with multiple alert types"
            },
            # --- Pallas 3.4: Cloud Security patterns ---
            "aws_security_explicit": {
                "patterns": [
                    r"\baws\b.*\b(security|alert|event|login|console)\b",
                    r"\bcloudtrail\b",
                    r"\b(iam|s3\s+bucket)\b.*\b(change|policy|access|alert|event)\b",
                    r"\baws\b.*\b(iam|s3|ec2|lambda)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "aws OR cloudtrail OR rule.groups:aws"},
                "confidence": 0.92,
                "description": "AWS/CloudTrail security event detection"
            },
            "o365_security_explicit": {
                "patterns": [
                    r"\b(o365|office\s*365|microsoft\s*365)\b",
                    r"\bmailbox\b.*\b(access|forward|delegation|suspicious)\b",
                    r"\b(sharepoint|onedrive|teams)\b.*\b(alert|suspicious|access)\b",
                    r"\bexchange\s+online\b.*\b(alert|suspicious)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "office365 OR o365 OR rule.groups:office365"},
                "confidence": 0.92,
                "description": "Office 365 / Microsoft 365 security events"
            },
            "azure_security_explicit": {
                "patterns": [
                    r"\bazure\b.*\b(security|alert|event|ad|active\s+directory)\b",
                    r"\bazure\s+ad\b",
                    r"\b(entra|azure\s+identity)\b",
                    r"\b(conditional\s+access|azure\s+sentinel)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "azure OR azure ad OR rule.groups:azure"},
                "confidence": 0.92,
                "description": "Azure / Azure AD security events"
            },
            "github_security_explicit": {
                "patterns": [
                    r"\bgithub\b.*\b(security|alert|event|push|commit|access)\b",
                    r"\b(repository|repo)\b.*\b(security|alert|suspicious|access)\b",
                    r"\bgithub\b.*\b(audit|log)\b",
                    r"\b(secret|credential)\b.*\b(leak|expos|commit|push|github)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "github OR rule.groups:github"},
                "confidence": 0.92,
                "description": "GitHub security events"
            },
            "mfa_security_explicit": {
                "patterns": [
                    r"\bmfa\b.*\b(fail|bypass|suspicious|alert|event)\b",
                    r"\b(multi.factor|two.factor|2fa)\b.*\b(fail|bypass|suspicious)\b",
                    r"\b(fail|bypass|suspicious)\b.*\b(mfa|multi.factor|2fa)\b",
                    r"\bmfa\b.*\b(fatigue|bomb|push\s+spam)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "mfa OR multi-factor OR 2fa OR two-factor authentication"},
                "confidence": 0.92,
                "description": "MFA failure / bypass detection"
            },
            "oauth_token_explicit": {
                "patterns": [
                    r"\boauth\b.*\b(token|suspicious|alert|abuse)\b",
                    r"\b(api\s+key|api\s+token)\b.*\b(leak|expos|suspicious|misus|compromis)\b",
                    r"\btoken\b.*\b(theft|steal|compromis|suspicious|exfiltrat)\b",
                    r"\b(bearer|access)\s+token\b.*\b(alert|suspicious|expos)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "oauth OR api key OR token theft OR api token"},
                "confidence": 0.91,
                "description": "OAuth / API token abuse detection"
            },
            # --- Pallas 3.4: User & Access Monitoring patterns ---
            "admin_group_change_explicit": {
                "patterns": [
                    r"\badmin\b.*\bgroup\b.*\b(add|chang|modif|member)\b",
                    r"\bgroup\s+membership\b.*\b(chang|modif|add)\b",
                    r"\b(user|account)\b.*\badd\b.*\b(admin|privileged|domain\s+admin)\b",
                    r"\bevent[\.\s]*id[\s:]*4728\b",
                    r"\bevent[\.\s]*id[\s:]*4732\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "admin group OR group membership change OR event.id:4728 OR event.id:4732"},
                "confidence": 0.92,
                "description": "Admin group membership change detection"
            },
            "disabled_account_explicit": {
                "patterns": [
                    r"\bdisabled\b.*\baccount\b.*\b(login|logon|access|attempt|activit)\b",
                    r"\baccount\b.*\bdisabled\b.*\b(login|logon|access|attempt)\b",
                    r"\b(login|logon|access)\b.*\bdisabled\b.*\baccount\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "disabled account login OR disabled account access"},
                "confidence": 0.91,
                "description": "Disabled account login attempt detection"
            },
            "account_lockout_explicit": {
                "patterns": [
                    r"\baccount\s+lock\s*out\b",
                    r"\blocked\s+out\b.*\b(account|user)\b",
                    r"\bevent[\.\s]*id[\s:]*4740\b",
                    r"\b(user|account)\b.*\blocked\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "account lockout OR locked out OR event.id:4740"},
                "confidence": 0.92,
                "description": "Account lockout detection"
            },
            "password_policy_explicit": {
                "patterns": [
                    r"\bpassword\s+(policy|chang|reset|expir)\b",
                    r"\b(policy)\b.*\bpassword\b",
                    r"\bpassword\b.*\b(reset|expir|violation)\b",
                    r"\bevent[\.\s]*id[\s:]*4723\b",
                    r"\bevent[\.\s]*id[\s:]*4724\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "password policy OR password change OR password reset OR event.id:4723"},
                "confidence": 0.91,
                "description": "Password policy / change events"
            },
            "kerberos_anomaly_explicit": {
                "patterns": [
                    r"\bkerberos\b.*\b(anomal|suspicious|attack|ticket|fail)\b",
                    r"\b(golden\s+ticket|silver\s+ticket|kerberoast)\b",
                    r"\bevent[\.\s]*id[\s:]*4768\b",
                    r"\bevent[\.\s]*id[\s:]*4769\b",
                    r"\btgt\b.*\b(anomal|suspicious|attack)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "kerberos OR TGT OR ticket OR event.id:4768 OR event.id:4769 OR kerberoast"},
                "confidence": 0.92,
                "description": "Kerberos anomaly / ticket attack detection"
            },
            "ntlm_auth_explicit": {
                "patterns": [
                    r"\bntlm\b.*\b(auth|spike|relay|anomal|suspicious|attack)\b",
                    r"\b(ntlm\s+relay|ntlm\s+downgrade)\b",
                    r"\bntlm\b.*\b(v1|version\s*1)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "ntlm OR ntlm authentication OR ntlm relay"},
                "confidence": 0.91,
                "description": "NTLM authentication anomaly detection"
            },
            "usb_activity_explicit": {
                "patterns": [
                    r"\busb\b.*\b(activit|device|connect|mount|insert|detect)\b",
                    r"\b(removable|external)\s*(media|device|drive|storage)\b",
                    r"\b(mass\s+storage|thumb\s+drive|flash\s+drive)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "usb OR removable media OR removable device OR mass storage"},
                "confidence": 0.92,
                "description": "USB / removable media activity detection"
            },
            "file_access_explicit": {
                "patterns": [
                    r"\bfile\s+access\b.*\b(suspicious|unauthorized|unusual|audit)\b",
                    r"\b(suspicious|unauthorized|unusual)\b.*\bfile\s+access\b",
                    r"\bobject\s+access\b.*\b(audit|alert|event)\b",
                    r"\bevent[\.\s]*id[\s:]*4663\b",
                    r"\bevent[\.\s]*id[\s:]*4656\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "file access OR object access OR event.id:4663 OR event.id:4656"},
                "confidence": 0.91,
                "description": "Suspicious file / object access detection"
            },
            "remote_login_explicit": {
                "patterns": [
                    r"\bremote\s+(login|logon|access)\b.*\b(event|alert|monitor|unusual)\b",
                    r"\brdp\s+(login|logon|session|connect)\b",
                    r"\bssh\s+(login|logon|session|connect|brute)\b",
                    r"\blogon\s+type\s+(3|10)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "remote login OR rdp logon OR ssh login OR logon type 10 OR logon type 3"},
                "confidence": 0.91,
                "description": "Remote login event detection"
            },
            "service_account_explicit": {
                "patterns": [
                    r"\bservice\s+account\b.*\b(interactive|suspicious|anomal|login|logon|misus)\b",
                    r"\b(interactive|suspicious)\b.*\bservice\s+account\b",
                    r"\bservice\s+account\b.*\b(logon\s+type|elevat|priv)\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "service account OR interactive logon service OR service logon type"},
                "confidence": 0.91,
                "description": "Service account misuse detection"
            },
            # --- Pallas 3.4: Agent Health Enhancement patterns ---
            "agent_os_summary_explicit": {
                "patterns": [
                    r"\bagent\b.*\b(by|per|group)\b.*\b(os|operating\s+system|platform)\b",
                    r"\b(os|operating\s+system|platform)\b.*\b(breakdown|summary|distribution|group)\b.*\bagent\b",
                    r"\bagent\b.*\b(os|platform)\s+(distribution|breakdown|summary)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "correlation": "agent_health_overview",
                "confidence": 0.92,
                "description": "Agent summary grouped by OS"
            },
            "agent_version_mismatch_explicit": {
                "patterns": [
                    r"\bagent\b.*\bversion\b.*\b(mismatch|differ|inconsisten|mixed|outdated)\b",
                    r"\b(version|upgrade)\b.*\b(mismatch|differ|inconsisten|outdated)\b.*\bagent\b",
                    r"\bagent\b.*\b(need|require)\b.*\b(updat|upgrade|patch)\b",
                    r"\b(outdated|old)\b.*\bagent\b.*\bversion\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "correlation": "agent_health_overview",
                "confidence": 0.92,
                "description": "Agent version mismatch detection"
            },
            "stale_agents_explicit": {
                "patterns": [
                    r"\b(stale|unresponsive|not\s+report)\b.*\bagent\b",
                    r"\bagent\b.*\b(not\s+report|stale|unresponsive|silent)\b",
                    r"\bagent\b.*\b(last\s+seen|not\s+check|no\s+heartbeat)\b.*\b(\d+\s+hour|\d+\s+day)\b",
                    r"\bagent\b.*\b(missing|lost\s+contact|gone\s+dark)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "correlation": "agent_health_overview",
                "confidence": 0.91,
                "description": "Stale / unresponsive agent detection"
            },
            "newly_enrolled_explicit": {
                "patterns": [
                    r"\b(new|recent|latest)\b.*\benroll\b.*\bagent\b",
                    r"\bagent\b.*\b(new|recent)\b.*\benroll\b",
                    r"\b(recently|newly)\s+(added|registered|deployed|enrolled)\b.*\bagent\b",
                    r"\bagent\b.*\b(recently|newly)\s+(added|registered|deployed|enrolled)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "correlation": "agent_health_overview",
                "confidence": 0.91,
                "description": "Newly enrolled agent detection"
            },
            "agent_event_volume_explicit": {
                "patterns": [
                    r"\b(abnormal|unusual|high|low|anomal)\b.*\bevent\s+volume\b",
                    r"\bevent\s+volume\b.*\b(abnormal|unusual|anomal)\b",
                    r"\bagent\b.*\b(event\s+volume|event\s+count|log\s+volume)\b",
                    r"\b(noisy|quiet|silent)\b.*\bagent\b.*\bevent\b",
                ],
                "tool": "get_log_source_health",
                "secondary": ["get_wazuh_agents"],
                "correlation": "agent_event_volume",
                "confidence": 0.91,
                "description": "Agent event volume anomaly detection"
            },
            # --- Pallas 3.5: Multi-Asset Campaign Detection patterns ---
            "attack_graph_explicit": {
                "patterns": [
                    r"\battack\s+graph\b",
                    r"\bbuild\b.*\battack\s+graph\b",
                    r"\b48\s*hour\b.*\battack\s+graph\b",
                    r"\b(show|build|create|generate)\b.*\battack\s+graph\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_wazuh_agents"],
                "correlation": "multi_asset_campaign_detection",
                "confidence": 0.94,
                "description": "Build attack graph"
            },
            "coordinated_campaign_explicit": {
                "patterns": [
                    r"\bcoordinat\w*\s+(attack|campaign)\b",
                    r"\bmulti.asset\s+campaign\b",
                    r"\bcampaign\s+detect\b",
                    r"\b(detect|find|show)\b.*\bcoordinat\w*\b.*\b(attack|campaign)\b",
                    r"\b(same|shared)\b.*\b(attack|indicator)\b.*\b(multiple|several)\b.*\b(agent|host)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_wazuh_agents"],
                "correlation": "multi_asset_campaign_detection",
                "confidence": 0.93,
                "description": "Coordinated multi-asset campaign detection"
            },
            "anomalous_encryption_explicit": {
                "patterns": [
                    r"\banomal\w*\s+encrypt\b",
                    r"\bunusual\s+encrypt\b",
                    r"\bsuspicious\s+tls\b",
                    r"\b(legacy|old|weak)\b.*\b(tls|ssl|encrypt)\b",
                    r"\b(self.signed|unusual)\b.*\b(cert|tls|ssl)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_wazuh_agents"],
                "correlation": "multi_asset_campaign_detection",
                "confidence": 0.91,
                "description": "Anomalous encryption / suspicious TLS"
            },
            "shadow_it_explicit": {
                "patterns": [
                    r"\bshadow\s+it\b",
                    r"\bunknown\s+(system|device|host)\b",
                    r"\bunmanag\w*\s+(device|system|host|asset)\b",
                    r"\b(rogue|unauthorized)\b.*\b(device|system|host)\b",
                    r"\b(find|detect|show)\b.*\bunknown\b.*\b(internal|device|system)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_wazuh_agents"],
                "correlation": "multi_asset_campaign_detection",
                "confidence": 0.92,
                "description": "Shadow IT / unmanaged device detection"
            },
            # --- Pallas 3.5: Alert Noise Analysis patterns ---
            "false_positive_explicit": {
                "patterns": [
                    r"\bfalse\s+positive\b",
                    r"\blikely\s+false\s+positive\b",
                    r"\bfp\s+analy\b",
                    r"\b(which|what)\b.*\balert\b.*\bfalse\s+positive\b",
                    r"\b(identify|find|detect)\b.*\bfalse\s+positive\b",
                ],
                "tool": "get_rule_trigger_analysis",
                "secondary": ["get_alert_timeline", "get_wazuh_agents"],
                "correlation": "alert_noise_analysis",
                "confidence": 0.93,
                "description": "False positive identification"
            },
            "rule_tuning_explicit": {
                "patterns": [
                    r"\brule\s+tun\b",
                    r"\bnoisy\s+rule\b",
                    r"\btune\b.*\bnoisy\b.*\balert\b",
                    r"\b(suggest|recommend)\b.*\brule\b.*\b(tun|exclud|suppress)\b",
                    r"\balert\b.*\bnois\b.*\b(reduc|tun|fix)\b",
                ],
                "tool": "get_rule_trigger_analysis",
                "secondary": ["get_alert_timeline", "get_wazuh_agents"],
                "correlation": "alert_noise_analysis",
                "confidence": 0.93,
                "description": "Rule tuning recommendations for noisy alerts"
            },
            "low_severity_chain_explicit": {
                "patterns": [
                    r"\blow.severity\b.*\bchain\b",
                    r"\bstealth\b.*\battack\b",
                    r"\blow.severity\b.*\balert\b.*\bchain\b",
                    r"\bstealth\b.*\bthrough\b.*\blow\b",
                    r"\b(hidden|subtle)\b.*\battack\b.*\blow\b.*\bseverity\b",
                ],
                "tool": "get_rule_trigger_analysis",
                "secondary": ["get_alert_timeline", "get_wazuh_agents"],
                "correlation": "alert_noise_analysis",
                "confidence": 0.92,
                "description": "Low-severity alert chaining (stealth attacks)"
            },
            # --- Pallas 3.5: Off-Hours Anomaly Detection patterns ---
            "off_hours_powershell_explicit": {
                "patterns": [
                    r"\bpowershell\b.*\b(outside|off|non).*(hour|business|maintenan)\b",
                    r"\bpowershell\b.*\boff.hours\b",
                    r"\b(outside|off|after)\b.*\b(hour|business)\b.*\bpowershell\b",
                    r"\b(unusual|abnormal|anomal)\b.*\bpowershell\b.*\b(time|hour|night)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "off_hours_anomaly_detection",
                "args": {"query": "powershell OR rule.groups:powershell", "limit": 200},
                "confidence": 0.92,
                "description": "PowerShell activity outside business hours"
            },
            "off_hours_outbound_explicit": {
                "patterns": [
                    r"\boutbound\b.*\b(non.business|off.hours|after.hours)\b",
                    r"\b(off|after|non).hours\b.*\b(traffic|outbound|network)\b",
                    r"\b(unusual|abnormal)\b.*\b(outbound|traffic)\b.*\b(non.business|night|off)\b",
                    r"\bnight\s*time\b.*\b(traffic|outbound|connection)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "off_hours_anomaly_detection",
                "args": {"query": "outbound OR connection OR network OR traffic", "limit": 200},
                "confidence": 0.91,
                "description": "Unusual outbound traffic during non-business hours"
            },
            "unusual_port_multi_host_explicit": {
                "patterns": [
                    r"\bunusual\s+port\b.*\b(across|multiple)\b.*\bhost\b",
                    r"\b(unexpected|unusual|abnormal)\b.*\bport\b.*\b(multiple|several|different)\b",
                    r"\bport\b.*\busage\b.*\b(across|multiple)\b",
                    r"\bsame\s+port\b.*\b(multiple|different)\b.*\bhost\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "off_hours_anomaly_detection",
                "args": {"query": "port OR connection OR listen OR bind", "limit": 200},
                "confidence": 0.91,
                "description": "Unusual port usage across multiple hosts"
            },
            # --- Pallas 3.5: Data Exfiltration Detection patterns ---
            "exfiltration_volume_explicit": {
                "patterns": [
                    r"\b(data\s+)?exfil\b",
                    r"\bdata\s+exfiltration\b",
                    r"\boutbound\s+volume\s+anomal\b",
                    r"\b(detect|find|show)\b.*\bexfiltration\b",
                    r"\bdata\s+leak\b",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_agent_processes"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "data_exfiltration_detection",
                "confidence": 0.93,
                "description": "Data exfiltration detection"
            },
            "bulk_download_explicit": {
                "patterns": [
                    r"\bbulk\b.*\bfile\b.*\b(download|access)\b",
                    r"\bmass\s+file\s+access\b",
                    r"\bsensitive\s+director\b.*\b(access|download|read)\b",
                    r"\b(large|bulk|mass)\b.*\b(file|document)\b.*\b(access|transfer|copy)\b",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_agent_processes"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "data_exfiltration_detection",
                "confidence": 0.91,
                "description": "Bulk file download / mass file access"
            },
            "data_compression_explicit": {
                "patterns": [
                    r"\bcompression\b.*\b(before|outbound|transfer)\b",
                    r"\bzip\b.*\b(before|outbound|transfer|exfil)\b",
                    r"\b(rar|7z|tar|gzip)\b.*\b(before|outbound|suspicious)\b",
                    r"\bcompress\b.*\b(before|then)\b.*\b(transfer|send|upload)\b",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_agent_processes"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "data_exfiltration_detection",
                "confidence": 0.91,
                "description": "Compression before outbound transfer"
            },
            "data_staging_explicit": {
                "patterns": [
                    r"\bdata\s+staging\b",
                    r"\bstaging\b.*\b(before|exfil|transfer)\b",
                    r"\b(collect|gather|stage)\b.*\bdata\b.*\b(exfil|transfer)\b",
                    r"\bpre.exfil\b.*\bstag\b",
                ],
                "tool": "get_fim_events",
                "secondary": ["get_agent_processes"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "data_exfiltration_detection",
                "confidence": 0.91,
                "description": "Data staging before exfiltration"
            },
            # --- Pallas 3.5: Cross-Platform Threat Correlation patterns ---
            "c2_endpoint_correlation_explicit": {
                "patterns": [
                    r"\bc2\b.*\b(endpoint|process|wazuh)\b",
                    r"\bcommand.and.control\b.*\b(endpoint|process)\b",
                    r"\b(suricata|ids)\b.*\bc2\b.*\b(endpoint|correlat)\b",
                    r"\b(correlat|combin)\b.*\b(suricata|ids|network)\b.*\b(endpoint|wazuh)\b",
                ],
                "tool": "get_suricata_alerts",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents", "search_suricata_http"],
                "correlation": "cross_platform_threat_correlation",
                "confidence": 0.93,
                "description": "C2 alerts correlated with endpoint events"
            },
            "c2_beaconing_explicit": {
                "patterns": [
                    r"\bc2\s+beacon\b",
                    r"\bbeacon\w*\s+(pattern|detect|analy)\b",
                    r"\bperiodic\s+traffic\b",
                    r"\bbeacon\b.*\b(c2|c&c|command)\b",
                    r"\b(detect|find|show)\b.*\bbeacon\b",
                ],
                "tool": "get_suricata_alerts",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents", "search_suricata_http"],
                "correlation": "cross_platform_threat_correlation",
                "confidence": 0.94,
                "description": "C2 beaconing / periodic traffic detection"
            },
            "cross_platform_hash_explicit": {
                "patterns": [
                    r"\bmalware\s+hash\b",
                    r"\bknown\s+hash\b.*\bexecut\b",
                    r"\bhash\b.*\b(check|match|lookup|ioc)\b",
                    r"\b(md5|sha256|sha1)\b.*\b(match|check|execute)\b",
                ],
                "tool": "get_suricata_alerts",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents", "search_suricata_http"],
                "correlation": "cross_platform_threat_correlation",
                "confidence": 0.91,
                "description": "Known malware hash correlation"
            },
            "suspicious_ip_departments_explicit": {
                "patterns": [
                    r"\bsuspicious\s+ip\b.*\bacross\b",
                    r"\bip\b.*\brepeated\b.*\bacross\b",
                    r"\bsame\s+ip\b.*\b(multiple|different|several)\b.*\b(agent|host|department)\b",
                    r"\b(common|shared)\b.*\battack\b.*\bip\b",
                ],
                "tool": "get_suricata_alerts",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents", "search_suricata_http"],
                "correlation": "cross_platform_threat_correlation",
                "confidence": 0.91,
                "description": "Suspicious IP repeated across departments"
            },
            "cloud_endpoint_correlation_explicit": {
                "patterns": [
                    r"\bgithub\b.*\btoken\b.*\b(endpoint|alert)\b",
                    r"\bcloud\b.*\btoken\b.*\b(endpoint|alert)\b",
                    r"\b(github|cloud)\b.*\b(plus|and|\+)\b.*\b(endpoint|alert)\b",
                    r"\btoken\b.*\b(misuse|abuse)\b.*\b(endpoint|correlat)\b",
                ],
                "tool": "get_suricata_alerts",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents", "search_suricata_http"],
                "correlation": "cross_platform_threat_correlation",
                "confidence": 0.91,
                "description": "Cloud/GitHub token + endpoint alert correlation"
            },
            # --- Pallas 3.5: Dynamic Risk Scoring patterns ---
            "dynamic_risk_score_explicit": {
                "patterns": [
                    r"\bdynamic\b.*\brisk\s+scor\b",
                    r"\bcalculate\b.*\brisk\s+scor\b",
                    r"\bcomposite\b.*\brisk\b",
                    r"\brisk\s+scor\b.*\b(all|every|each)\b.*\bagent\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.93,
                "description": "Dynamic composite risk scoring"
            },
            "top_compromised_explicit": {
                "patterns": [
                    r"\b(most|top)\b.*\b(likely|probably)\b.*\bcompromis",
                    r"\btop\b.*\bcompromis.*\basset\b",
                    r"\b(highest|most)\b.*\brisk\b.*\b(asset|host)\b",
                    r"\bmost\b.*\b(at.risk|vulnerable|compromis)",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.93,
                "description": "Top most likely compromised assets"
            },
            "exposed_exploitable_explicit": {
                "patterns": [
                    r"\bexposed\b.*\b(extern|internet)\b.*\b(exploit|vuln)\b",
                    r"\bcritical\b.*\bserver\b.*\bexposed\b",
                    r"\bexploit\b.*\bvuln\b.*\bexpos\b",
                    r"\binternet.facing\b.*\b(vuln|exploit)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.91,
                "description": "Exposed servers with exploitable vulnerabilities"
            },
            "weak_compliance_explicit": {
                "patterns": [
                    r"\bweak\b.*\bcomplia\b",
                    r"\bhigh.value\b.*\bweak\b.*\bcomplia\b",
                    r"\bcomplia\b.*\b(poor|weak|fail)\b",
                    r"\b(fail|poor)\b.*\bcomplia\b.*\b(posture|score)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.91,
                "description": "High-value assets with weak compliance"
            },
            "lateral_exposure_explicit": {
                "patterns": [
                    r"\blateral\s+movement\s+exposure\b",
                    r"\bblast\s+radius\b",
                    r"\blateral\b.*\b(exposure|risk|spread)\b.*\b(asset|agent)\b",
                    r"\b(spread|pivot)\b.*\brisk\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.91,
                "description": "Lateral movement exposure risk"
            },
            "repeated_incidents_explicit": {
                "patterns": [
                    r"\brepeated\b.*\b(security\s+)?incident\b",
                    r"\brecurr\b.*\bincident\b",
                    r"\b(frequent|repeat)\b.*\b(alert|incident|attack)\b.*\b(asset|agent|host)\b",
                    r"\basset\b.*\b(repeat|recurr|frequent)\b.*\bincident\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.91,
                "description": "Assets with repeated security incidents"
            },
            "unpatched_suspicious_explicit": {
                "patterns": [
                    r"\bunpatch\b.*\b(and|with|\+)\b.*\bsuspicious\b",
                    r"\bvulnerable\b.*\b(with|and|\+)\b.*\balert\b",
                    r"\bunpatch\b.*\bsuspicious\s+behav\b",
                    r"\b(vuln|cve)\b.*\b(and|plus|\+)\b.*\b(alert|suspicious|anomal)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.92,
                "description": "Unpatched vulns AND suspicious behavior"
            },
            "containment_priority_explicit": {
                "patterns": [
                    r"\b(require|need)\b.*\bimmediate\b.*\bcontainment\b",
                    r"\bcontainment\s+priorit\b",
                    r"\b(immediate|urgent)\b.*\b(contain|isolat|quarantin)\b",
                    r"\bprioritize\b.*\b(containment|response)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"] + (["get_suricata_alerts"] if self.suricata_enabled else []) + ["get_agent_ports"],
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.92,
                "description": "Alerts requiring immediate containment prioritization"
            },
            # --- Pallas 3.5: Temporal Attack Sequence patterns ---
            "attack_sequence_auth_explicit": {
                "patterns": [
                    r"\bfailed\s+login\b.*\bfollow",
                    r"\bbrute\s*force\b.*\bthen\b.*\b(success|login)\b",
                    r"\b(failed|failure)\b.*\b(then|follow|before)\b.*\b(success|access)\b",
                    r"\blogin\b.*\bfailure\b.*\bsuccess\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "authentication OR login OR logon OR failed login OR successful login", "limit": 200},
                "confidence": 0.93,
                "description": "Auth-based attack sequence: failed logins → success"
            },
            "attack_sequence_lateral_explicit": {
                "patterns": [
                    r"\blateral\s+movement\s+chain\b",
                    r"\bmovement\b.*\bacross\b.*\bhost\b",
                    r"\blateral\b.*\b(chain|sequen|pivot|spread)\b",
                    r"\bhost\s+to\s+host\b.*\b(attack|spread|movement)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "lateral movement OR psexec OR wmi OR remote execution OR smb OR winrm", "limit": 200},
                "confidence": 0.93,
                "description": "Lateral movement chain detection"
            },
            "attack_sequence_malware_outbound_explicit": {
                "patterns": [
                    r"\bmalware\b.*\b(follow|then|outbound)\b",
                    r"\bmalware\b.*\bconnection\b",
                    r"\b(virus|trojan|malware)\b.*\b(c2|c&c|callback|beacon)\b",
                    r"\binfection\b.*\b(outbound|external|command.and.control)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "malware OR virus OR trojan OR outbound OR c2 OR command and control", "limit": 200},
                "confidence": 0.92,
                "description": "Malware → outbound C2 sequence"
            },
            "attack_sequence_general_explicit": {
                "patterns": [
                    r"\bmulti.stage\s+attack\b",
                    r"\battack\s+sequen\b",
                    r"\bchained\s+attack\b",
                    r"\bkill\s*chain\b.*\b(detect|analy|show)\b",
                    r"\b(detect|show|find)\b.*\bmulti.phase\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "*", "limit": 200},
                "confidence": 0.93,
                "description": "General multi-stage attack sequence detection"
            },
            "attack_sequence_credential_explicit": {
                "patterns": [
                    r"\bcredential\s+(theft|dump|steal)\b.*\bthen\b",
                    r"\bafter\s+credential\b",
                    r"\bpass.the.hash\b.*\b(then|follow|lateral)\b",
                    r"\bcredential\b.*\b(follow|then|sequence)\b.*\b(lateral|escalat|access)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "credential OR pass the hash OR mimikatz OR lsass OR ntlm OR kerberos", "limit": 200},
                "confidence": 0.92,
                "description": "Credential theft → follow-up attack sequence"
            },
            "attack_sequence_cloud_explicit": {
                "patterns": [
                    r"\bcloud\b.*\b(resource|bucket|instance)\b.*\b(creat|delet|then)\b",
                    r"\bs3\b.*\bpublic\b.*\b(then|access|follow)\b",
                    r"\bcloud\b.*\b(then|follow|sequence|chain)\b",
                    r"\b(aws|azure|gcp)\b.*\b(creat|delet)\b.*\b(then|follow)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "aws OR azure OR office365 OR github OR cloud OR s3 OR bucket", "limit": 200},
                "confidence": 0.91,
                "description": "Cloud resource attack sequence"
            },
            "attack_sequence_firewall_endpoint_explicit": {
                "patterns": [
                    r"\bfirewall\b.*\b(deny|block)\b.*\b(spike|endpoint|alert)\b",
                    r"\b(deny|block)\s+spike\b.*\balert\b",
                    r"\bfirewall\b.*\b(plus|and|\+)\b.*\bendpoint\b",
                    r"\bnetwork\b.*\bblock\b.*\b(then|follow)\b.*\b(endpoint|host)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "firewall OR deny OR block OR drop OR reject", "limit": 200},
                "confidence": 0.91,
                "description": "Firewall deny spikes + endpoint alert correlation"
            },
            "attack_sequence_cloud_endpoint_explicit": {
                "patterns": [
                    r"\bcloud\b.*\b(login|anomal)\b.*\b(malware|endpoint)\b",
                    r"\bcloud\b.*\b(plus|and|\+)\b.*\bendpoint\b",
                    r"\b(office365|aws|azure)\b.*\b(anomal|suspicious)\b.*\b(endpoint|malware)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "office365 OR aws OR azure OR cloud login OR anomaly OR malware", "limit": 200},
                "confidence": 0.91,
                "description": "Cloud login anomaly + endpoint malware sequence"
            },
            "attack_sequence_priv_esc_explicit": {
                "patterns": [
                    r"\blogin\b.*\bthen\b.*\bprivilege\b",
                    r"\b(escalat|elevat).*\bafter\b.*\blogin\b",
                    r"\b(success|auth)\b.*\b(then|follow)\b.*\b(priv|escalat|elevat)",
                    r"\baccess\b.*\bthen\b.*\bescalat",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "authentication OR login OR privilege escalation OR SeDebugPrivilege OR 4672", "limit": 200},
                "confidence": 0.92,
                "description": "Login → privilege escalation sequence"
            },
            "attack_sequence_timeline_explicit": {
                "patterns": [
                    r"\battack\s+timeline\b",
                    r"\bshow\b.*\battack\b.*\bsequen",
                    r"\btemporal\b.*\b(attack|threat|sequen)",
                    r"\btimeline\b.*\b(attack|threat|incident)\b",
                ],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "*", "limit": 200},
                "confidence": 0.92,
                "description": "Attack timeline / temporal sequence analysis"
            },
            # --- Pallas 3.3: New always-active patterns ---
            "vulnerability_trend_explicit": {
                "patterns": [
                    r"\bvulnerabilit\w*\s+(trend|distribution|breakdown)\b.*\bseverity\b",
                    r"\bseverity\b.*\b(trend|distribution|breakdown)\b.*\bvulnerabilit",
                    r"\bvuln\w*\b.*\bby\s+severity\b",
                    r"\bseverity\b.*\bbreakdown\b.*\b(vuln|cve)\b",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "vulnerability_trend_by_severity",
                "confidence": 0.92,
                "description": "Vulnerability distribution by severity with agent context"
            },
            "combined_alerts_all_explicit": {
                "patterns": [
                    r"\b(all|combined|unified|every)\b.*\balert\b.*\b(together|combined|both|unified)\b",
                    r"\b(endpoint|wazuh)\b.*\band\b.*\b(network|suricata)\b.*\balert\b",
                    r"\b(show|list)\b.*\b(all|every)\b.*\balert\b.*\b(source|platform)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts"] if self.suricata_enabled else [],
                "correlation": "combined_alert_view" if self.suricata_enabled else None,
                "confidence": 0.91,
                "description": "Combined alert view from all sources"
            },
            # ============================================================
            # M7 (2026-06-10) - Combined endpoint+network alerts WITH
            # severity filter. The single-source severity patterns
            # (network_critical_alerts etc. at conf 0.99) would otherwise
            # win against combined_alerts_all_explicit (conf 0.91) and the
            # response would only show one source. Conf 1.00 here beats
            # the 0.99 single-source patterns AND the +0.05 compound bonus.
            # ============================================================
            "combined_alerts_with_severity_explicit": {
                "patterns": [
                    # severity + endpoint + network (any order)
                    r"\b(critical|high|medium|low)\b[^.?!]*\b(endpoint|wazuh|hids)\b[^.?!]*\b(network|suricata|nids|ids|ips)\b",
                    r"\b(critical|high|medium|low)\b[^.?!]*\b(network|suricata|nids|ids|ips)\b[^.?!]*\b(endpoint|wazuh|hids)\b",
                    r"\b(endpoint|wazuh|hids)\b[^.?!]*\b(network|suricata|nids|ids|ips)\b[^.?!]*\b(critical|high|medium|low)\b",
                    r"\b(network|suricata|nids|ids|ips)\b[^.?!]*\b(endpoint|wazuh|hids)\b[^.?!]*\b(critical|high|medium|low)\b",
                ],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts"] if self.suricata_enabled else [],
                "correlation": "combined_alert_view" if self.suricata_enabled else None,
                "confidence": 1.00,
                "description": "Combined endpoint+network alerts filtered by severity (critical/high/medium/low)"
            },
            "compare_vulns_status_explicit": {
                "patterns": [
                    r"\bcompare\b.*\b(active|disconnected)\b.*\bvulnerabilit",
                    r"\bvulnerabilit\w*\b.*\bactive\b.*\bvs\b.*\bdisconnected\b",
                    r"\b(active|disconnected)\b.*\bvs\b.*\b(active|disconnected)\b.*\bvuln",
                ],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "compare_vulns_active_vs_disconnected",
                "confidence": 0.92,
                "description": "Compare vulnerabilities: active vs disconnected agents"
            },
            "security_posture_explicit": {
                "patterns": [
                    r"\b(overall|complete|full|environment)\b.*\b(security|risk)\s+(posture|status|health|overview)\b",
                    r"\b(security|risk)\s+(posture|overview|status)\b.*\b(overall|full|complete)\b",
                    r"\bhow\s+secure\b.*\b(environment|infrastructure)\b",
                ],
                "tool": "get_wazuh_agents",
                "secondary": (["get_wazuh_vulnerabilities", "get_wazuh_alerts", "get_suricata_alerts"]
                              if self.suricata_enabled else ["get_wazuh_vulnerabilities", "get_wazuh_alerts"]),
                "correlation": "comprehensive_security_posture" if self.suricata_enabled else "agents_with_vulnerabilities",
                "confidence": 0.90,
                "description": "Overall environment security posture"
            },
        })

        # ================================================================
        # MEDIUM-CONFIDENCE PATTERNS (Keyword-based, 70-80% confidence)
        # ----------------------------------------------------------------
        # E1.6a (2026-06-11): the ~930-line static pattern data lives in
        # orchestrator/planning/keyword_patterns.py now. Same entries,
        # same conditional Suricata branches, same field shapes. The
        # factory call below returns the merged dict so every existing
        # use (self.KEYWORD_PATTERNS.items(), .values(), update by other
        # __init__ blocks) keeps working unchanged.
        # ================================================================
        from .orchestrator.planning.keyword_patterns import build_keyword_patterns
        self.KEYWORD_PATTERNS = build_keyword_patterns(self.suricata_enabled)

        # ================================================================
        # ENTITY EXTRACTION PATTERNS
        # ================================================================
        self.ENTITY_PATTERNS = {
            "severity": r"\b(critical|high|medium|low)\b",
            "agent_id": r"(?:\bagent[_\-\s]*|\bagt[_\-\s]+|\bid\s+|\#)(\d{1,6})\b",
            # Capture agent NAME (alphanumeric with -, _, .) when prefixed with the
            # context word "agent" / "host" / "for" / "on". Excludes pure numeric
            # ids (caught by agent_id) and short stop-words.
            "agent_name": r"\b(?:agent|host|machine|endpoint|server|for|on)\s+([a-zA-Z][a-zA-Z0-9_\-.]{2,60})\b",
            "cve_id": r"\bCVE[\s\-]?(\d{4})[\s\-]?(\d{4,7})\b",
            "status": r"\b(active|running|connected|online|alive|healthy|operational|responding|live|up|disconnected|offline|inactive|down|dead|unreachable|lost|stale|never_connected|never connected|pending|unenrolled|unregistered)\b",
            "time_range": r"\b(?:last|past|in\s+the\s+last)\s+(\d+)\s*(minute|min|mins|hour|hr|hrs|day|d|week|wk|wks)s?\b",
            "framework": r"\b(PCI-DSS|PCI|HIPAA|GDPR|NIST|SOX)\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "level": r"\blevel\s+(\d+)\b",
            "rule_id": r"\brule\s*(?:id)?\s*[:#\-]?\s*(\d{4,6})\b",
            "mitre_id": r"\b(T\d{4}(?:\.\d{3})?)\b",
            "mitre_tactic": r"\b(initial access|execution|persistence|privilege escalation|defense evasion|credential access|discovery|lateral movement|collection|command and control|exfiltration|impact|reconnaissance|resource development)\b",
            # MITRE technique NAMES (top ~25 commonly seen on Wazuh dashboards).
            # Captured names normalize to title-case so they match `rule.mitre.technique`.
            "mitre_technique": (
                r"\b("
                r"powershell|local\s+accounts?|valid\s+accounts?|"
                r"stored\s+data\s+manipulation|"
                r"email\s+forwarding\s+rule|modify\s+registry|"
                r"application\s+layer\s+protocol|ingress\s+tool\s+transfer|"
                r"software\s+deployment\s+tools?|file\s+deletion|"
                r"brute\s+force|credential\s+dumping|process\s+injection|"
                r"command\s+and\s+scripting\s+interpreter|"
                r"scheduled\s+task|service\s+execution|"
                r"masquerading|obfuscat(?:ed|ion)\s+files?|"
                r"system\s+information\s+discovery|"
                r"remote\s+services?|lateral\s+tool\s+transfer|"
                r"data\s+staged|data\s+from\s+local\s+system|"
                r"exfiltration\s+over\s+c2|"
                r"account\s+manipulation|web\s+shell|"
                r"impair\s+defenses|indicator\s+removal"
                r")\b"
            ),
            # Tier 1 improvements: protocol / port / hostname entity capture
            "protocol": r"\b(TCP|UDP|ICMP|HTTP|HTTPS|SSH|RDP|DNS|SMTP|FTP|SFTP|LDAP|SMB|TLS|SSL)\b",
            "port": r"\bport\s*[:#]?\s*(\d{2,5})\b",
            "hostname": r"\b([a-zA-Z][\w-]{1,30}\.[a-zA-Z][\w.-]{1,60})\b",
            # OS filter — captures common platform names + distros. Normalized to
            # canonical platform value (linux/windows/darwin) downstream.
            "os": r"\b(linux|windows|win\s*(?:server|10|11)?|macos|mac\s*os|darwin|osx|ubuntu|debian|centos|redhat|rhel|fedora|alpine|kali|amazon\s*linux|amzn|suse)\b",
        }

        # Phase 5.2: Validate that every routing pattern references a real tool.
        # Catches typos/orphans at startup instead of silent fallback at query time.
        try:
            self._validate_pattern_tool_references()
        except Exception as _e:
            logger.error(f"[STARTUP] Pattern validator crashed: {_e}")

    def _record_routing_metrics(self, plan) -> None:
        """Phase 6.2: bump NL routing counters and audit low-confidence."""
        try:
            from wazuh_mcp_server.monitoring import (
                NL_ROUTING_STRATEGY, NL_ROUTING_CONFIDENCE, NL_ROUTING_LOW_CONF,
            )
            _method = getattr(plan, "selection_method", None) or "unknown"
            NL_ROUTING_STRATEGY.labels(method=_method).inc()
            _conf = float(getattr(plan, "confidence", 0.0) or 0.0)
            _bucket = f"{int(_conf * 10) * 10:02d}"
            NL_ROUTING_CONFIDENCE.labels(bucket=_bucket).inc()
            if _conf < 0.30:
                NL_ROUTING_LOW_CONF.inc()
                _q = getattr(plan, "user_query", "") or ""
                logger.warning(
                    f"[AUDIT_LOW_CONF] method={_method} conf={_conf:.2f} "
                    f"primary={getattr(plan, 'primary_tool', '?')} query={_q[:200]!r}"
                )
        except Exception as _e:
            logger.debug(f"_record_routing_metrics: {_e}")

    def _validate_pattern_tool_references(self) -> None:
        """Phase 5.2: assert every pattern.tool / pattern.secondary[*] is a real
        ToolRegistry entry. Logs orphans loudly but does not raise — orphans
        degrade routing but should not block startup."""
        try:
            from wazuh_mcp_server.tool_registry import ToolRegistry
            known = {t.name for t in ToolRegistry.get_all_tools()}
        except Exception as _e:
            logger.warning(f"[STARTUP] Pattern validator: ToolRegistry unavailable ({_e})")
            return
        orphans = []
        combined = {}
        if hasattr(self, "HIGH_CONFIDENCE_PATTERNS"):
            combined.update(self.HIGH_CONFIDENCE_PATTERNS)
        if hasattr(self, "KEYWORD_PATTERNS"):
            combined.update(self.KEYWORD_PATTERNS)
        for pname, pconf in combined.items():
            if not isinstance(pconf, dict):
                continue
            prim = pconf.get("tool")
            if prim and prim not in known:
                orphans.append((pname, "primary", prim))
            for sec in pconf.get("secondary", []) or []:
                if sec not in known:
                    orphans.append((pname, "secondary", sec))
        if orphans:
            logger.error(f"[STARTUP] {len(orphans)} pattern->tool reference(s) are orphaned:")
            for pname, role, tname in orphans[:20]:
                logger.error(f"   pattern={pname!r} {role}_tool={tname!r} (not in registry)")
        else:
            logger.info(
                f"[STARTUP] OK: {len(combined)} routing patterns validated, "
                f"all tool references resolve in ToolRegistry"
            )

    # ================================================================
    # Phase 1 — Small-LLM router fleet/OS snapshot helpers
    # ================================================================
    async def _router_get_fleet_snapshot(self):
        """Fetch full agent list (id, name, os, status) for router context.

        Cached for 1 hour to avoid hammering the Wazuh Manager. Returns a
        list of dicts; empty list on failure (router degrades gracefully).
        """
        import time as _time
        if not hasattr(self, "_router_fleet_cache"):
            self._router_fleet_cache = None
            self._router_fleet_cache_ts = 0.0
        now = _time.time()
        if self._router_fleet_cache and (now - self._router_fleet_cache_ts < 3600):
            return self._router_fleet_cache
        if self._tool_executor is None:
            return []
        try:
            resp = await self._tool_executor.execute("get_wazuh_agents", {"limit": 500})
            items = []
            if hasattr(resp, "data"):
                items = (resp.data or {}).get("data", {}).get("affected_items", []) or []
            elif isinstance(resp, dict):
                items = resp.get("data", {}).get("affected_items", []) or []
            self._router_fleet_cache = items
            self._router_fleet_cache_ts = now
            logger.info(f"[ROUTER_LLM] fleet snapshot refreshed: {len(items)} agents")
            return items
        except Exception as e:
            logger.warning(f"[ROUTER_LLM] fleet snapshot failed: {e}")
            return self._router_fleet_cache or []

    def _router_get_os_values(self, agents):
        """Extract distinct OS strings from a fleet list."""
        os_set = set()
        for a in agents or []:
            if not isinstance(a, dict):
                continue
            os_obj = a.get("os", {})
            if isinstance(os_obj, dict):
                full = (os_obj.get("name") or "")
                version = os_obj.get("version") or ""
                if version:
                    full = (full + " " + version).strip()
                if full:
                    os_set.add(full)
            elif isinstance(os_obj, str) and os_obj:
                os_set.add(os_obj)
        return sorted(os_set)

    async def select_tools(self, query: str, context: Optional[QueryContext] = None) -> ToolPlan:
        """
        Main entry point for hybrid tool selection.

        Execution flow:
        1. Normalize query
        2. Extract entities
        3. Try high-confidence patterns (regex)
        4. Try medium-confidence patterns (keywords)
        5. Fall back to LLM for complex queries
        6. Validate and enrich plan
        7. Return execution plan

        Args:
            query: User's natural language query
            context: Optional context (user, session, etc.)

        Returns:
            ToolPlan with selected tools and arguments
        """

        start_time = time.time()
        query_lower = query.lower().strip()

        # Query canonicalization (convert question forms to statement forms for better regex hit rate)
        query_lower = self._canonicalize_query(query_lower)

        # Synonym normalization (expand common SOC acronyms so regex/keyword layers catch more)
        query_lower = self._normalize_synonyms(query_lower)

        # Extract entities (used by all strategies)
        entities = self._extract_entities(query)

        # Merge carried entities from conversation history (follow-up support).
        #
        # M14b (2026-06-11) PALLAS_P4_PIVOT_ENTITY_DROP: only carry an entity
        # forward when the new query plausibly references it. Without this,
        # a topic pivot like "show vulnerabilities" after "show processes
        # for agent 072" silently inherits agent_id=072 and the vuln tool
        # gets scoped to one agent (returns 0) instead of running fleet-wide.
        # Severity / time_range are still always carried because they're
        # safe defaults; agent_id is the one that derails routing.
        # E1.2 (2026-06-11): M14b predicate moved to
        # orchestrator/execution/arg_guards.should_carry_agent_id. Same
        # env flag, same regex, same behaviour - just one import away.
        from .orchestrator.execution.arg_guards import should_carry_agent_id
        self._m14_dropped_carries = []  # surfaced in metadata.memory
        _q_low_for_carry = (query or "").lower()
        _agent_carry_ok = should_carry_agent_id(_q_low_for_carry)
        if context and context.carried_entities:
            for key, value in context.carried_entities.items():
                if key in entities:
                    continue
                if key == "agent_id" and not _agent_carry_ok:
                    self._m14_dropped_carries.append({
                        "key": key,
                        "value": value,
                        "reason": "query did not reference an agent (topic pivot)",
                    })
                    logger.info(
                        f"[M14b] Dropped carry {key}={value} - query has no agent reference"
                    )
                    continue
                entities[key] = value
                logger.info(f"[CONTEXT] Carried entity from previous turn: {key}={value}")

        # Resolve agent_name → agent_id when the user typed a name (e.g., "test-sensor")
        # rather than a numeric ID. Uses a 5-minute TTL cache to avoid hammering
        # the agents endpoint on every query. Skipped silently if the resolver
        # isn't wired up or the name isn't known.
        if entities.get("agent_name") and not entities.get("agent_id"):
            resolved = await self._resolve_agent_name_to_id(entities["agent_name"])
            if resolved:
                entities["agent_id"] = resolved
                logger.info(
                    f"[ENTITY] Resolved agent_name='{entities['agent_name']}' → "
                    f"agent_id='{resolved}'"
                )
            else:
                logger.warning(
                    f"[ENTITY] agent_name='{entities['agent_name']}' not found "
                    f"in fleet; downstream tools will run without agent filter"
                )

        # Store entities for process_query() access (conversation memory)
        self._last_entities = entities

        logger.info(f"Extracted entities: {entities}")

        # NLP Hunt-query guard: complex multi-indicator hunt queries bypass the
        # deterministic Strategy 0.4 / 0.45 keyword tiers and go straight to
        # the LLM router so the model can interpret all the signals together.
        if self._is_complex_hunt_query(query):
            logger.info("[ROUTING] complex hunt query detected -> routing directly to LLM (Strategy 0.5/3)")
            # E1.5a (2026-06-11): Qwen call now goes through the shared
            # helper in orchestrator/planning/qwen_router.py. Same env
            # gate, same prompt build, same validation + confidence
            # gate. Falls through to legacy Strategy 3 on None.
            try:
                from .orchestrator.planning.qwen_router import try_route_with_qwen
                _fleet = await self._router_get_fleet_snapshot()
                _os_vals = self._router_get_os_values(_fleet)
                _hq_result = await try_route_with_qwen(
                    query, context, self.llm, _fleet, _os_vals,
                    selection_method="llm_router_hunt",
                    reasoning_prefix="Hunt-query guard",
                )
                if _hq_result is not None:
                    _hq_plan, _ = _hq_result
                    self.stats["llm_router"] = self.stats.get("llm_router", 0) + 1
                    self.stats["total"] = self.stats.get("total", 0) + 1
                    return _hq_plan
            except Exception as _hq_exc:
                logger.warning(f"[ROUTING] hunt-guard LLM route failed: {_hq_exc}; falling through to legacy LLM (Strategy 3)")
            # Fallback to legacy Strategy 3 LLM selection if Qwen path didn't work
            try:
                _llm_plan = await self._llm_tool_selection(query, entities)
                _validated = self._validate_and_enrich_plan(_llm_plan, query, entities)
                self.stats["llm"] = self.stats.get("llm", 0) + 1
                self.stats["total"] = self.stats.get("total", 0) + 1
                _validated.selection_method = "llm_hunt_fallback"
                return _validated
            except Exception as _hq2_exc:
                logger.warning(f"[ROUTING] hunt-guard legacy LLM also failed: {_hq2_exc}; falling through to normal cascade")

        # === PALLAS_P3_SMART_HYBRID_BYPASS === Phase 3 Smart Hybrid
        # When PALLAS_SMART_HYBRID=true (default), skip Strategies 0.4
        # (keyword overrides) and 0.45 (correlation preempt) and go straight
        # to Strategy 0.5 (Qwen LLM router). Domain knowledge from 0.4 has
        # been migrated into the Qwen system prompt (llm_router_prompt.py).
        # Toggle to 'false' to restore the legacy cascade for rollback.
        if os.getenv("PALLAS_SMART_HYBRID", "true").lower() == "true":
            logger.info("[ROUTING][P3] smart hybrid ON - bypassing Strategy 0.4 + 0.45, going to Qwen")

            # E1.5b (2026-06-12): M14a regex-shortcut block moved to
            # orchestrator/planning/post_corrections.try_regex_shortcut.
            # Same env gate, same anchored-pattern + confidence>=0.95
            # filter, same arg backfill + corrections + metric stamping.
            from .orchestrator.planning.post_corrections import try_regex_shortcut
            _m14a_plan = try_regex_shortcut(
                query, query_lower, entities, context,
                self.HIGH_CONFIDENCE_PATTERNS,
                self._build_plan_from_config,
                self._build_search_events_args,
                self._build_indicator_args,
                self._apply_post_selection_corrections,
                self._record_routing_metrics,
                self.stats,
            )
            if _m14a_plan is not None:
                return _m14a_plan

            # E1.5a (2026-06-11): Qwen call delegated to the shared
            # try_route_with_qwen helper. The helper returns
            # (plan, complex_flag) or None. We patch selection_method
            # to llm_router_complex when the flag is true (was inlined
            # before) and stash the flag on the plan attr so the WS
            # layer can still read it.
            try:
                from .orchestrator.planning.qwen_router import try_route_with_qwen
                _fleet = await self._router_get_fleet_snapshot()
                _os_vals = self._router_get_os_values(_fleet)
                _p3_result = await try_route_with_qwen(
                    query, context, self.llm, _fleet, _os_vals,
                    selection_method="llm_router",
                    reasoning_prefix="Router LLM (P3 smart hybrid)",
                )
                if _p3_result is not None:
                    _plan_obj, _complex = _p3_result
                    if _complex:
                        logger.info("[ROUTING][P3] Qwen flagged complex=true; downstream will escalate to agent_loop")
                        _plan_obj.selection_method = "llm_router_complex"
                    self.stats["llm_router"] = self.stats.get("llm_router", 0) + 1
                    self.stats["total"] = self.stats.get("total", 0) + 1
                    try:
                        _plan_obj.complex = _complex  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    return _plan_obj
                else:
                    logger.info("[ROUTING][P3] Qwen returned no usable plan; will escalate via legacy fallback")
            except Exception as _p3_exc:
                logger.warning(f"[ROUTING][P3] Qwen path failed: {_p3_exc}; falling through to legacy cascade")
            # If we get here, Qwen didn't return a usable plan — fall through to legacy Strategy 1-3.
            # Jump past Strategy 0.4 and 0.45 by setting a flag the code below honors.
            _p3_skip_04_045 = True
        else:
            _p3_skip_04_045 = False
        # === END PALLAS_P3_SMART_HYBRID_BYPASS ===

        # === PALLAS_P3_STRAT_GUARD === skip Strategy 0.4 + 0.45 when Smart Hybrid is on.
        # When Qwen returns a usable plan above, the bypass already returned; we get here
        # only if Qwen failed/low-confidence. Under Smart Hybrid (_p3_skip_04_045=True)
        # we jump straight to Strategy 1-3 instead of falling back through 0.4 + 0.45.
        if not _p3_skip_04_045:

            # ================================================================
            # STRATEGY 0.4 — Keyword overrides (deterministic, runs BEFORE LLM router)
            # ================================================================
            # E1.6c (2026-06-12): lifted to
            # orchestrator/planning/keyword_overrides.try_keyword_overrides.
            # Same 8 intent branches (health, compliance, top-risk, NIDS,
            # MITRE tactic/T-code, SCA, Defender), same log line prefixes
            # ([OVERRIDE]), same stats counters (keyword_override, total).
            from .orchestrator.planning.keyword_overrides import try_keyword_overrides
            _override_plan = await try_keyword_overrides(
                query, entities,
                getattr(self, "suricata_enabled", False),
                self._router_get_fleet_snapshot,
                self.stats,
            )
            if _override_plan is not None:
                return _override_plan
            # ================================================================
            # STRATEGY 0.45 — Correlation-bearing regex/keyword PREEMPT
            # E1.6b (2026-06-11): logic lifted to
            # orchestrator/planning/correlation_preempt.try_correlation_preempt.
            # Same scoring, same log line prefixes, same stats counters,
            # same build_plan_from_config call signature.
            # ================================================================
            from .orchestrator.planning.correlation_preempt import try_correlation_preempt
            _corr_plan = try_correlation_preempt(
                query, entities,
                getattr(self, "HIGH_CONFIDENCE_PATTERNS", None),
                getattr(self, "KEYWORD_PATTERNS", None),
                self._build_plan_from_config,
                self.stats,
            )
            if _corr_plan is not None:
                return _corr_plan

        # ================================================================
        # STRATEGY 0.5 — Small-LLM Router (Phase 1 of NL routing v2)
        # ================================================================
        # When ROUTING_LLM_ENABLED=true and the underlying LLMRouter exposes
        # route_query (i.e. Qwen2.5-7B is wired up), let the small model
        # decide tool + args BEFORE the regex/keyword tier runs. If the
        # router returns low confidence (<0.5), invalid JSON, or any error,
        # we fall through to the existing tiers — they remain the safety net.
        if os.getenv("ROUTING_LLM_ENABLED", "false").lower() == "true" and hasattr(self.llm, "route_query"):
            try:
                from .llm_router_prompt import build_router_prompt, validate_against_registry
                _fleet = await self._router_get_fleet_snapshot()
                _os_vals = self._router_get_os_values(_fleet)
                _all_tools = ToolRegistry.get_all_tools()
                _recent_ctx = ""
                try:
                    if context is not None and getattr(context, "conversation_summary", None):
                        _recent_ctx = context.conversation_summary or ""
                except Exception:
                    _recent_ctx = ""
                _sys_prompt = build_router_prompt(_all_tools, _fleet, _os_vals, _recent_ctx)
                _plan_dict = await self.llm.route_query(query, _sys_prompt)
                if _plan_dict:
                    _known = {t.name for t in _all_tools}
                    if validate_against_registry(_plan_dict, _known):
                        _conf = float(_plan_dict.get("confidence", 0.5) or 0.5)
                        if _conf >= 0.5:
                            logger.info(
                                f"✅ ROUTER_LLM MATCH: {_plan_dict['primary_tool']} "
                                f"conf={_conf:.2f} secondary={len(_plan_dict.get('secondary_tools', []))} "
                                f"reasoning={(_plan_dict.get('reasoning') or '')!r}"
                            )
                            _plan = ToolPlan(
                                primary_tool=_plan_dict["primary_tool"],
                                secondary_tools=list(_plan_dict.get("secondary_tools", [])),
                                primary_args=dict(_plan_dict.get("primary_args", {}) or {}),
                                secondary_args=list(_plan_dict.get("secondary_args", []) or []),
                                confidence=_conf,
                                selection_method="llm_router",
                                reasoning=f"Router LLM: {_plan_dict.get('reasoning', '')}",
                                correlation_strategy=_plan_dict.get("correlation_strategy"),
                            )
                            # M8b (2026-06-10) - ROUTER_LLM previously returned
                            # the plan without running _infer_correlation_strategy.
                            # That left requires_correlation=False even for
                            # known multi-source combos like
                            # wazuh_alerts + suricata_*_alerts, so the
                            # combined_alert_view correlation never fired and
                            # Suricata data was silently dropped from the
                            # response. Now we infer the strategy from the
                            # tool combination + flip requires_correlation so
                            # the downstream pipeline knows to merge.
                            if _plan.secondary_tools:
                                _valid_strategies = (
                                    set(CorrelationEngine.STRATEGIES.keys())
                                    if hasattr(CorrelationEngine, "STRATEGIES") else set()
                                )
                                _llm_strategy_valid = (
                                    _plan.correlation_strategy is not None
                                    and _plan.correlation_strategy in _valid_strategies
                                )
                                if _llm_strategy_valid:
                                    _plan.requires_correlation = True
                                    logger.info(
                                        f"[ROUTER_LLM] keeping LLM correlation_strategy: "
                                        f"{_plan.correlation_strategy}"
                                    )
                                else:
                                    _inferred = self._infer_correlation_strategy(
                                        _plan.primary_tool,
                                        _plan.secondary_tools
                                    )
                                    if _inferred:
                                        _plan.correlation_strategy = _inferred
                                        _plan.requires_correlation = True
                                        logger.info(
                                            f"[ROUTER_LLM] inferred correlation_strategy: "
                                            f"{_inferred} (primary={_plan.primary_tool}, "
                                            f"secondary={_plan.secondary_tools})"
                                        )
                                    elif _plan.correlation_strategy:
                                        logger.warning(
                                            f"[ROUTER_LLM] unknown LLM strategy "
                                            f"'{_plan.correlation_strategy}', clearing"
                                        )
                                        _plan.correlation_strategy = None
                            self.stats["llm_router"] = self.stats.get("llm_router", 0) + 1
                            self.stats["total"] = self.stats.get("total", 0) + 1
                            return _plan
                        else:
                            logger.info(
                                f"[ROUTER_LLM] low confidence ({_conf:.2f}); "
                                f"falling through to rules tier"
                            )
                    else:
                        logger.warning(
                            "[ROUTER_LLM] tool validation failed; falling through to rules"
                        )
                else:
                    logger.warning("[ROUTER_LLM] returned no plan; falling through to rules")
            except Exception as _e:
                logger.warning(
                    f"[ROUTER_LLM] error: {_e}; falling through to rules tier"
                )

        # ================================================================
        # COMPLEXITY DETECTOR - Route complex queries to LLM
        # ================================================================
        complexity_patterns = [
            r'\b(most|top|best|worst|highest|lowest)\b.*\b(vulnerability|vulnerabilities|agent|agents)\b',
            r'\b(rank|sort|order|compare)\b',
            r'\band\b.*\b(are they|their|status|health)\b',
            r'\b(which|what).*\b(have|with)\b.*\band\b',
            r'\b(disconnected|active)\b.*\b(but|and|with)\b.*\b(critical|high|vulnerability)\b',
            # Multi-tool correlation patterns
            r'\b(agents?|hosts?)\b.*\b(alerts?|vulnerabilit)\b.*\b(and|with|also)\b',
            r'\b(correlate|cross.?reference|combine|overlay)\b',
            r'\b(alerts?\s+and\s+vulnerabilit|vulnerabilit\w+\s+and\s+alerts?)\b',
            r'\b(posture|risk\s+profile|security\s+status)\b.*\bagent\b',
            r'\b(fim|file\s+integrity)\b.*\b(and|with)\b.*\b(process|agent)\b',
            r'\b(which\s+agents?\s+have\s+both)\b',
            r'\b(overview|summary)\b.*\b(everything|all|complete|full)\b',
            r'\b(complete|full)\b.*\b(overview|summary|posture|status)\b',
        ]

        is_complex_query = any(re.search(p, query_lower) for p in complexity_patterns)

        if is_complex_query:
            logger.info("🧠 COMPLEX QUERY DETECTED → Routing to LLM")
            # Skip to LLM section (STRATEGY 3)
            try:
                llm_plan = await self._llm_tool_selection(query, entities)
                validated = self._validate_and_enrich_plan(llm_plan, query, entities)
                self.stats["llm"] += 1
                self.stats["total"] += 1
                self._record_llm_query(query, validated.primary_tool, "complex_query_detector")
                logger.info(f"Tool selection: LLM ({time.time() - start_time:.2f}s)")
                return validated
            except Exception as e:
                logger.error(f"LLM failed: {e}, falling back to rules")
                # Continue to rule-based matching below

        # ================================================================
        # STRATEGY 1: High-Confidence Regex Patterns
        # Collect ALL matches, then pick the most specific one.
        # Compound patterns (with secondary tools) are preferred over
        # simple patterns when both match — this prevents "disconnected
        # agents with critical vulns" from matching the simple
        # "disconnected_agents_only" instead of the compound
        # "disconnected_with_critical_vulns" pattern.
        # ================================================================
        all_regex_matches = []
        for pattern_name, config in self.HIGH_CONFIDENCE_PATTERNS.items():
            for pattern in config.get("patterns", []):
                if re.search(pattern, query_lower):
                    all_regex_matches.append((pattern_name, config))
                    break  # don't match multiple patterns from same config

        if all_regex_matches:
            # 2026-05-15 fix: compound patterns (with secondary tools) used to
            # get ABSOLUTE priority over simple patterns regardless of
            # confidence. That made queries like "Show me all network critical
            # security incidents..." pick the broad alert_investigation pattern
            # (compound, conf 0.9) over the specific network_critical_alerts
            # pattern (simple, conf 0.99). Now compound gets a small +0.05
            # confidence bonus instead — preserves the original intent for
            # ties while letting clear simple-pattern wins through.
            COMPOUND_BONUS = 0.05
            def _effective_conf(name_config):
                _name, _conf = name_config
                _base = _conf.get("confidence", 0.9)
                _is_compound = bool(_conf.get("secondary"))
                return _base + (COMPOUND_BONUS if _is_compound else 0)

            best_name, best_config = max(all_regex_matches, key=_effective_conf)

            logger.info(
                f"✅ HIGH-CONFIDENCE MATCH: {best_name} "
                f"(conf={best_config.get('confidence', 0.9)}, "
                f"compound={bool(best_config.get('secondary'))}; "
                f"{len(all_regex_matches)} candidates)"
            )

            plan = self._build_plan_from_config(
                config=best_config,
                entities=entities,
                selection_method="rule_regex",
                reasoning=f"Matched regex pattern: {best_name}",
                user_query=query
            )

            # FIX: For search_security_events, inject search query from user query
            # Pallas 3.4: Prefer pattern-injected args; only use NL extraction as fallback
            if plan.primary_tool == "search_security_events":
                if not plan.primary_args.get("query"):
                    plan.primary_args = self._build_search_events_args(query, entities)
                else:
                    # Pattern already provides optimal query; just ensure time_range/limit
                    if "time_range" not in plan.primary_args:
                        plan.primary_args["time_range"] = entities.get("time_range", "24h")
                    if "limit" not in plan.primary_args:
                        plan.primary_args["limit"] = 100

            # FIX R4: For threat/IoC tools, extract indicator from query
            if plan.primary_tool in ("analyze_security_threat", "check_ioc_reputation"):
                plan.primary_args = self._build_indicator_args(query, entities)

            # Post-selection corrections (agent-specific vulns, multi-intent)
            plan = self._apply_post_selection_corrections(plan, entities, query_lower, query, context=context)

            self.stats["rule_regex"] += 1
            self.stats["total"] += 1
            self._record_routing_metrics(plan)

            selection_time = time.time() - start_time
            logger.info(f"Tool selection completed in {selection_time*1000:.1f}ms (regex)")
            logger.info(f"[DEBUG] primary_tool={plan.primary_tool}, primary_args={plan.primary_args}")
            logger.info(f"[DEBUG] secondary_tools={plan.secondary_tools}")

            return plan

        # ================================================================
        # STRATEGY 2: Medium-Confidence Keyword Patterns
        # ================================================================
        for pattern_name, config in self.KEYWORD_PATTERNS.items():
            # Check if keywords match
            keywords = config.get("keywords", [])
            exclude_keywords = config.get("exclude_keywords", [])

            has_keyword = any(kw in query_lower for kw in keywords)
            has_exclusion = any(ekw in query_lower for ekw in exclude_keywords)

            if has_keyword and not has_exclusion:
                logger.info(f"⚡ KEYWORD MATCH: {pattern_name}")

                plan = self._build_plan_from_config(
                    config=config,
                    entities=entities,
                    selection_method="rule_keyword",
                    reasoning=f"Matched keywords: {pattern_name}",
                    user_query=query
                )

                # FIX: For search_security_events, inject search query from user query
                # Pallas 3.4: Prefer pattern-injected args; only use NL extraction as fallback
                if plan.primary_tool == "search_security_events":
                    if not plan.primary_args.get("query"):
                        plan.primary_args = self._build_search_events_args(query, entities)
                    else:
                        if "time_range" not in plan.primary_args:
                            plan.primary_args["time_range"] = entities.get("time_range", "24h")
                        if "limit" not in plan.primary_args:
                            plan.primary_args["limit"] = 100

                # FIX R4: For threat/IoC tools, extract indicator from query
                if plan.primary_tool in ("analyze_security_threat", "check_ioc_reputation"):
                    plan.primary_args = self._build_indicator_args(query, entities)

                # Post-selection corrections (agent-specific vulns, multi-intent)
                plan = self._apply_post_selection_corrections(plan, entities, query_lower, query, context=context)

                self.stats["rule_keyword"] += 1
                self.stats["total"] += 1
                self._record_routing_metrics(plan)

                selection_time = time.time() - start_time
                logger.info(f"Tool selection completed in {selection_time*1000:.1f}ms (keyword)")
                logger.info(f"[DEBUG] primary_tool={plan.primary_tool}, primary_args={plan.primary_args}")
                logger.info(f"[DEBUG] secondary_tools={plan.secondary_tools}")

                return plan

        # ================================================================
        # STRATEGY 3: LLM Fallback (Complex/Ambiguous Queries)
        # ================================================================
        logger.info("🤔 NO RULE MATCH - Consulting LLM for complex query")

        try:
            llm_plan = await self._llm_tool_selection(query, entities)

            # Validate LLM output
            validated_plan = self._validate_and_enrich_plan(llm_plan, query, entities)

            # FIX: For search_security_events selected by LLM, inject search args
            # Pallas 3.4: Prefer pattern-injected args; only use NL extraction as fallback
            if validated_plan.primary_tool == "search_security_events":
                if not validated_plan.primary_args.get("query"):
                    validated_plan.primary_args = self._build_search_events_args(query, entities)
                else:
                    if "time_range" not in validated_plan.primary_args:
                        validated_plan.primary_args["time_range"] = entities.get("time_range", "24h")
                    if "limit" not in validated_plan.primary_args:
                        validated_plan.primary_args["limit"] = 100

            # Post-selection corrections (agent-specific vulns, multi-intent)
            validated_plan = self._apply_post_selection_corrections(validated_plan, entities, query_lower, query, context=context)

            self.stats["llm"] += 1
            self.stats["total"] += 1
            self._record_llm_query(query, validated_plan.primary_tool, "no_rule_match")

            selection_time = time.time() - start_time
            logger.info(f"Tool selection completed in {selection_time*1000:.1f}ms (LLM)")
            logger.info(f"[DEBUG] primary_tool={validated_plan.primary_tool}, primary_args={validated_plan.primary_args}")

            return validated_plan

        except Exception as e:
            logger.error(f"LLM tool selection failed: {e}")

            # ================================================================
            # STRATEGY 4: Safe Fallback with Intelligent Guidance
            # ================================================================
            logger.warning("⚠️ FALLBACK - Using safe default with guidance")

            # Fallback confidence is set to 0.2 (below 0.3 guardrail) so
            # process_query() returns the guidance message WITHOUT executing any tool.
            # The primary_tool below is never executed — it's only retained for plan shape.
            fallback_plan = ToolPlan(
                primary_tool="get_wazuh_agents",
                secondary_tools=[],
                primary_args={"limit": 50},
                confidence=0.2,
                selection_method="fallback",
                reasoning=f"All strategies failed. Returning guidance-only. Error: {str(e)}",
                fallback_guidance=self._build_fallback_guidance(query, entities=entities, error=str(e))
            )

            self.stats["fallback"] += 1
            self.stats["total"] += 1
            self._record_fallback_query(query, str(e))
            self._record_routing_metrics(fallback_plan)

            return fallback_plan

    def _record_llm_query(self, query: str, tool_chosen: str, trigger: str) -> None:
        """Record a query that hit the LLM fallback (for /api/query_stats telemetry)."""
        try:
            self.recent_llm_queries.append({
                "query": query[:200],
                "tool_chosen": tool_chosen,
                "trigger": trigger,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass  # never let telemetry fail the request

    def _record_fallback_query(self, query: str, error: str = "") -> None:
        """Record a query that hit safe-default fallback (for /api/query_stats telemetry)."""
        try:
            self.recent_fallback_queries.append({
                "query": query[:200],
                "error": error[:200],
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass

    def _apply_post_selection_corrections(
        self, plan: ToolPlan, entities: Dict[str, Any], query_lower: str, query: str,
        context: Optional["QueryContext"] = None,
    ) -> ToolPlan:
        """
        Post-selection corrections applied AFTER any strategy picks a tool.
        Fixes structural mismatches between pattern routing and tool capabilities.

        Correction 1 — Agent-specific critical vulnerabilities:
          get_wazuh_critical_vulnerabilities does NOT support agent_id filtering.
          When agent_id is present, swap to get_wazuh_vulnerabilities with severity=critical.

        Correction 2 — Multi-intent queries (vulns AND alerts):
          When query contains BOTH vulnerability and alert intents with a conjunction,
          ensure both tools fire as primary + secondary with proper correlation.
        """

        # ----------------------------------------------------------------
        # Correction 0 (2026-05-18): IP investigation propagation
        # When user asks "investigate IP X.X.X.X" the orchestrator picks
        # ip_investigation_pivot correlation but the per-tool args were
        # left empty, so each tool fetched unfiltered data and the
        # correlation found nothing. Propagate the IP entity into each
        # tool's args using the right field name per tool.
        # ----------------------------------------------------------------
        _inv_ip = entities.get("ip_address") if isinstance(entities, dict) else None
        if _inv_ip and plan.correlation_strategy == "ip_investigation_pivot":
            logger.info(f"[CORRECTION] propagating IP {_inv_ip} to all ip_investigation_pivot tools")
            # Suricata tools accept src_ip / dest_ip (OR'd via should clauses)
            _suricata_tools = (
                "get_suricata_alerts", "get_suricata_critical_alerts",
                "get_suricata_high_alerts", "get_suricata_medium_alerts",
                "get_suricata_low_alerts", "get_suricata_network_analysis",
                "search_suricata_alerts", "search_suricata_http",
                "get_suricata_http_analysis", "get_suricata_tls_analysis",
            )
            # Tools that accept a free-text "query" string we can stuff the IP into
            _query_tools = ("search_security_events", "search_suricata_alerts", "search_suricata_http")

            def _inject_ip(tool, args):
                args = dict(args or {})
                if tool in _suricata_tools:
                    args.setdefault("src_ip", _inv_ip)
                    args.setdefault("dest_ip", _inv_ip)
                if tool in _query_tools:
                    args.setdefault("query", _inv_ip)
                return args

            plan.primary_args = _inject_ip(plan.primary_tool, plan.primary_args)
            new_sec_args = []
            for _t, _a in zip(plan.secondary_tools, plan.secondary_args):
                new_sec_args.append(_inject_ip(_t, _a))
            plan.secondary_args = new_sec_args
            plan.reasoning += f" [CORRECTED: propagated IP {_inv_ip} to all tools]"

        # ----------------------------------------------------------------
        # Correction 1: Agent-specific critical vulnerability filtering
        # ----------------------------------------------------------------
        if plan.primary_tool == "get_wazuh_critical_vulnerabilities" and entities.get("agent_id"):
            logger.info(
                f"[CORRECTION] agent_id={entities['agent_id']} detected with "
                f"get_wazuh_critical_vulnerabilities → swapping to get_wazuh_vulnerabilities "
                f"with severity=critical for agent-specific filtering"
            )
            plan.primary_tool = "get_wazuh_vulnerabilities"
            plan.primary_args["severity"] = "critical"
            plan.primary_args["agent_id"] = entities["agent_id"]
            plan.reasoning += f" [CORRECTED: agent {entities['agent_id']} → get_wazuh_vulnerabilities(severity=critical)]"

        # ----------------------------------------------------------------
        # Correction 2.5 (Phase 2): Multi-status agent fan-out
        # When user asks for multiple agent statuses ("active AND inactive",
        # "active, disconnected and pending"), add a secondary get_wazuh_agents
        # call for each canonical status so the response covers all buckets.
        # ----------------------------------------------------------------
        _STATUS_SYNS = {
            "active": "active", "running": "active", "connected": "active",
            "online": "active", "alive": "active", "healthy": "active",
            "operational": "active", "responding": "active", "live": "active",
            "up": "active",
            "disconnected": "disconnected", "offline": "disconnected",
            "inactive": "disconnected", "down": "disconnected",
            "dead": "disconnected", "unreachable": "disconnected",
            "lost": "disconnected", "stale": "disconnected",
            "never connected": "never_connected", "never_connected": "never_connected",
            "pending": "never_connected", "unenrolled": "never_connected",
            "unregistered": "never_connected",
        }
        _statuses_in_query = set()
        for _syn, _canon in _STATUS_SYNS.items():
            if re.search(r"\b" + re.escape(_syn) + r"\b", query_lower):
                _statuses_in_query.add(_canon)
        _has_status_conj = bool(re.search(
            r"\b(and|both|plus|as\s+well\s+as|along\s+with)\b", query_lower
        )) or "," in query_lower
        if (
            len(_statuses_in_query) >= 2
            and _has_status_conj
            and any(_w in query_lower for _w in ("agent", "host", "endpoint", "machine", "server", "node", "device"))
            and plan.primary_tool in ("get_wazuh_agents", "get_wazuh_running_agents")
        ):
            _primary_status = plan.primary_args.get("status")
            if plan.primary_tool == "get_wazuh_running_agents":
                _primary_status = "active"
            for _st in sorted(_statuses_in_query):
                if _st == _primary_status:
                    continue
                _already_secondary = any(
                    _t == "get_wazuh_agents" and _a.get("status") == _st
                    for _t, _a in zip(plan.secondary_tools, plan.secondary_args)
                )
                if _already_secondary:
                    continue
                _sec_args = {_k: _v for _k, _v in plan.primary_args.items() if _k != "status"}
                _sec_args["status"] = _st
                plan.secondary_tools.append("get_wazuh_agents")
                plan.secondary_args.append(_sec_args)
            logger.info(
                f"[CORRECTION] Multi-status fan-out: statuses={sorted(_statuses_in_query)} "
                f"primary_status={_primary_status} added_secondaries="
                f"{[_a.get('status') for _a in plan.secondary_args[-3:]]}"
            )
            plan.reasoning += (
                f" [CORRECTED: multi-status fan-out -> {sorted(_statuses_in_query)}]"
            )

        # ----------------------------------------------------------------
        # Correction 2.6 (Phase 3.1): Multi-severity fan-out for alerts/vulnerabilities
        # ----------------------------------------------------------------
        _SEVERITIES = ("critical", "high", "medium", "low")
        _ALERT_TOOLS_3 = ("get_wazuh_alerts", "get_wazuh_alert_summary")
        _VULN_TOOLS_3 = ("get_wazuh_vulnerabilities", "get_wazuh_critical_vulnerabilities", "get_wazuh_vulnerability_summary")
        _SEV_DOMAIN_TOOLS = _ALERT_TOOLS_3 + _VULN_TOOLS_3
        _SEV_TOOL_GENERIC = {
            "get_wazuh_critical_vulnerabilities": "get_wazuh_vulnerabilities",
        }
        _sev_in_query = [_s for _s in _SEVERITIES if re.search(r"\b" + _s + r"\b", query_lower)]
        _has_sev_conj = bool(re.search(
            r"\b(and|both|plus|as\s+well\s+as|along\s+with)\b", query_lower
        )) or "," in query_lower
        _has_alert_or_vuln_noun = bool(re.search(
            r"\b(alert|alerts|vulnerabilit\w+|vuln|vulns|cve)\b", query_lower
        ))
        if (
            len(_sev_in_query) >= 2
            and _has_sev_conj
            and _has_alert_or_vuln_noun
            and plan.primary_tool in _SEV_DOMAIN_TOOLS
        ):
            _primary_sev = (plan.primary_args.get("severity") or "").lower() or None
            if plan.primary_tool == "get_wazuh_critical_vulnerabilities":
                _primary_sev = "critical"
            _generic_tool = _SEV_TOOL_GENERIC.get(plan.primary_tool, plan.primary_tool)
            for _sev in _sev_in_query:
                if _sev == _primary_sev:
                    continue
                _already_sev = any(
                    _t in _SEV_DOMAIN_TOOLS and str(_a.get("severity", "")).lower() == _sev
                    for _t, _a in zip(plan.secondary_tools, plan.secondary_args)
                )
                if _already_sev:
                    continue
                _sec_args_sev = {_k: _v for _k, _v in plan.primary_args.items() if _k.lower() != "severity"}
                _sec_args_sev["severity"] = _sev.capitalize()
                plan.secondary_tools.append(_generic_tool)
                plan.secondary_args.append(_sec_args_sev)
            logger.warning(
                f"[CORRECTION] Multi-severity fan-out: severities={_sev_in_query} "
                f"primary_severity={_primary_sev} primary_tool={plan.primary_tool}"
            )
            plan.reasoning += f" [CORRECTED: multi-severity fan-out -> {_sev_in_query}]"

        # ----------------------------------------------------------------
        # Correction 2.7 (Phase 3.2): Negation fan-out (status only for now)
        # "agents not active" -> include disconnected + never_connected
        # "agents that are not active" -> same
        # ----------------------------------------------------------------
        _neg_m = re.search(
            r"\b(?:not|except|excluding|without|but\s+not|other\s+than)\s+(\w+(?:\s+\w+)?)\b",
            query_lower
        )
        _NEG_AGENT_TOOLS = ("get_wazuh_agents", "get_wazuh_running_agents")
        if _neg_m and ("agent" in query_lower or any(_w in query_lower for _w in ("host","endpoint","machine","server","node","device"))):
            _neg_phrase = _neg_m.group(1).strip()
            _neg_token = _neg_phrase.split()[-1]
            _neg_canon = _STATUS_SYNS.get(_neg_token) or _STATUS_SYNS.get(_neg_phrase)
            # Force primary to agents tool when negation+agent-noun present but
            # routing picked a non-agent tool (mis-routing recovery).
            if _neg_canon and plan.primary_tool not in _NEG_AGENT_TOOLS:
                logger.warning(
                    f"[CORRECTION] Negation+agent noun but primary={plan.primary_tool}; "
                    f"forcing primary_tool=get_wazuh_agents"
                )
                plan.primary_tool = "get_wazuh_agents"
                # Strip args that don't apply to get_wazuh_agents
                plan.primary_args = {_k: _v for _k, _v in plan.primary_args.items()
                                     if _k in ("limit", "agent_id", "status")}
                if "limit" not in plan.primary_args:
                    plan.primary_args["limit"] = 500
                # Drop secondaries that belong to non-agent tools
                _kept_t, _kept_a = [], []
                for _t, _a in zip(plan.secondary_tools, plan.secondary_args):
                    if _t in _NEG_AGENT_TOOLS:
                        _kept_t.append(_t); _kept_a.append(_a)
                plan.secondary_tools = _kept_t
                plan.secondary_args = _kept_a
            if _neg_canon and plan.primary_tool in _NEG_AGENT_TOOLS:
                _all_canon = ["active", "disconnected", "never_connected"]
                _wanted = [_c for _c in _all_canon if _c != _neg_canon]
                # Force primary to first wanted canonical
                plan.primary_args["status"] = _wanted[0]
                # Add secondaries for remaining wanted
                for _w in _wanted[1:]:
                    _already_neg = any(
                        _t in _NEG_AGENT_TOOLS and _a.get("status") == _w
                        for _t, _a in zip(plan.secondary_tools, plan.secondary_args)
                    )
                    if _already_neg:
                        continue
                    _sec_args_neg = {_k: _v for _k, _v in plan.primary_args.items() if _k != "status"}
                    _sec_args_neg["status"] = _w
                    plan.secondary_tools.append("get_wazuh_agents")
                    plan.secondary_args.append(_sec_args_neg)
                # Drop any pre-existing secondary that matches the NEGATED status
                _filtered_t, _filtered_a = [], []
                for _t, _a in zip(plan.secondary_tools, plan.secondary_args):
                    if _t in _NEG_AGENT_TOOLS and _a.get("status") == _neg_canon:
                        continue
                    _filtered_t.append(_t)
                    _filtered_a.append(_a)
                plan.secondary_tools = _filtered_t
                plan.secondary_args = _filtered_a
                logger.warning(
                    f"[CORRECTION] Status negation: excluded={_neg_canon} -> included={_wanted}"
                )
                plan.reasoning += f" [CORRECTED: negation '{_neg_token}' -> include {_wanted}]"

        # ----------------------------------------------------------------
        # Correction 2: Multi-intent detection (vulnerabilities AND alerts)
        # When user asks for BOTH in one query ("show vulns and also alerts"),
        # ensure secondary tools include the missing intent.
        # ----------------------------------------------------------------
        has_conjunction = bool(re.search(r'\b(and\s+also|and\s+share|and\s+show|also\s+share|also\s+show|plus\s+show|as\s+well\s+as|and\s+the|and\s+critical|and\s+also)\b', query_lower))
        if not has_conjunction:
            # Simpler conjunction: "X and Y" where X and Y are different data types
            has_conjunction = bool(re.search(r'\balerts?\b.*\band\b.*\b(vulnerabilit\w*|vuln|cve)\b', query_lower)) or \
                              bool(re.search(r'\b(vulnerabilit\w*|vuln|cve)\b.*\band\b.*\balerts?\b', query_lower))

        if has_conjunction:
            has_alert_intent = bool(re.search(r'\balerts?\b', query_lower))
            has_vuln_intent = bool(re.search(r'\b(vulnerabilit\w*|vuln|cve)\b', query_lower))

            if has_alert_intent and has_vuln_intent:
                # User asked for BOTH — make sure both tools are in the plan
                vuln_tools = {"get_wazuh_vulnerabilities", "get_wazuh_critical_vulnerabilities", "get_wazuh_vulnerability_summary"}
                alert_tools = {"get_wazuh_alerts", "get_wazuh_alert_summary"}
                all_plan_tools = {plan.primary_tool} | set(plan.secondary_tools)

                has_vuln_tool = bool(all_plan_tools & vuln_tools)
                has_alert_tool = bool(all_plan_tools & alert_tools)

                if has_vuln_tool and not has_alert_tool:
                    # Vuln tool selected but alerts missing → inject alerts as secondary
                    logger.info("[CORRECTION] Multi-intent: adding get_wazuh_alerts as secondary (user asked for vulns AND alerts)")
                    plan.secondary_tools.append("get_wazuh_alerts")
                    alert_args = self._build_arguments("get_wazuh_alerts", entities, user_query=query)
                    plan.secondary_args.append(alert_args)
                    plan.requires_correlation = True
                    plan.correlation_strategy = "alerts_with_vulnerabilities"
                    plan.reasoning += " [CORRECTED: multi-intent → added alerts]"

                elif has_alert_tool and not has_vuln_tool:
                    # Alert tool selected but vulns missing → inject vulns as secondary
                    logger.info("[CORRECTION] Multi-intent: adding get_wazuh_vulnerabilities as secondary (user asked for alerts AND vulns)")
                    vuln_args = self._build_arguments("get_wazuh_vulnerabilities", entities, user_query=query)
                    if "critical" in query_lower:
                        vuln_args["severity"] = "critical"
                    plan.secondary_tools.append("get_wazuh_vulnerabilities")
                    plan.secondary_args.append(vuln_args)
                    # Also add agents for enrichment if not already present
                    if "get_wazuh_agents" not in plan.secondary_tools:
                        plan.secondary_tools.append("get_wazuh_agents")
                        plan.secondary_args.append(self._build_arguments("get_wazuh_agents", entities, user_query=query))
                    plan.requires_correlation = True
                    plan.correlation_strategy = "alerts_with_vulnerabilities"
                    plan.reasoning += " [CORRECTED: multi-intent → added vulnerabilities]"

        # ----------------------------------------------------------------
        # Phase 5.4: General time_range auto-inject for ANY tool path
        # If entity extraction found a time_range and the primary tool accepts
        # it (heuristically: alerts/vulns/search tools), propagate it. Existing
        # search_security_events-specific blocks already inject; this is a
        # defensive net for other tools that the user explicitly time-bounded.
        # ----------------------------------------------------------------
        _ent_time = entities.get("time_range") if isinstance(entities, dict) else None
        _TIME_AWARE_TOOLS = (
            "get_wazuh_alerts", "get_wazuh_alert_summary",
            "get_wazuh_vulnerabilities", "get_wazuh_vulnerability_summary",
            "get_wazuh_critical_vulnerabilities",
            "get_suricata_alerts", "get_suricata_alert_summary",
            "analyze_alert_patterns",
        )
        if _ent_time and plan.primary_tool in _TIME_AWARE_TOOLS:
            if "time_range" not in plan.primary_args:
                plan.primary_args["time_range"] = _ent_time
                logger.info(
                    f"[CORRECTION] Phase 5.4: injected time_range={_ent_time!r} "
                    f"into primary {plan.primary_tool} from entities"
                )

        # E1.5b (2026-06-12): M3 source-pin + M6d dimension-swap moved
        # to orchestrator/planning/post_corrections.apply_followup_post_corrections.
        # Same narrow-marker gate, same family classification, same
        # log-line prefixes ([FOLLOWUP_DIMENSION_SWAP], [FOLLOWUP_PIN]).
        from .orchestrator.planning.post_corrections import apply_followup_post_corrections
        plan = apply_followup_post_corrections(plan, entities, query_lower, query, context)
        return plan

    def _build_fallback_guidance(self, query: str, entities: Optional[Dict[str, Any]] = None, error: str = "") -> str:
        """Build intelligent guidance + Phase 6.1 diagnostic prefix."""
        query_lower = query.lower()
        _diag_lines = []
        try:
            _diag_lines.append("## I couldn't confidently route this query")
            _diag_lines.append("")
            _diag_lines.append("**What I understood from your query:**")
            if entities:
                _any = False
                for _k in ("agent_id","agent_name","status","severity","time_range",
                           "cve_id","ip_address","framework","rule_id","mitre_id",
                           "mitre_tactic","os","negation"):
                    if _k in entities and entities[_k]:
                        _diag_lines.append(f"- **{_k}**: `{entities[_k]}`")
                        _any = True
                if not _any:
                    _diag_lines.append("- (no recognized filters)")
            else:
                _diag_lines.append("- (entity extraction not run for this code path)")
            _diag_lines.append("")
            if error:
                _diag_lines.append(f"**Why I fell back:** {error[:200]}")
                _diag_lines.append("")
        except Exception:
            _diag_lines = []

        # Detect what the user might be asking about
        topic_suggestions = {
            "agent": [
                "show active agents",
                "show disconnected agents",
                "check health of agent 019",
                "show processes for agent 019",
                "show open ports for agent 019",
            ],
            "alert": [
                "show recent alerts",
                "show critical alerts from last 24 hours",
                "show top alert rules",
                "search brute force attacks",
                "search T1110",
            ],
            "vuln": [
                "show critical vulnerabilities",
                "show vulnerabilities for agent 019",
                "which agents have the most CVEs?",
                "show vulnerability summary",
            ],
            "suricata": [
                "show suricata alerts",
                "show top attack signatures",
                "show HTTP traffic analysis",
                "show JA3 fingerprints",
            ],
            "search": [
                "search brute force attacks",
                "search rule 80255",
                "search credential access",
                "search T1110",
                "search level 10 alerts",
            ],
            "correlat": [
                "show active agents with critical vulnerabilities",
                "correlate SIEM alerts with NIDS detections",
                "full stack investigation for agent 019",
                "compare vulnerabilities active vs disconnected",
            ],
        }

        # Find relevant suggestions based on query content
        relevant_suggestions = []
        for keyword, suggestions in topic_suggestions.items():
            if keyword in query_lower:
                relevant_suggestions = suggestions
                break

        # Default suggestions if no topic detected
        if not relevant_suggestions:
            relevant_suggestions = [
                "show active agents",
                "show recent alerts",
                "show critical vulnerabilities",
                "show suricata alerts",
                "search brute force attacks",
                "show agent 019 health",
                "correlate alerts with vulnerabilities",
            ]

        suggestions_text = "\n".join([f"  - `{s}`" for s in relevant_suggestions])

        return ("\n".join(_diag_lines) + "\n\n---\n\n" if _diag_lines else "") + (
            f"## Unable to Process Query\n\n"
            f"I wasn't able to determine the right tool to answer your question:\n"
            f"> *\"{query}\"*\n\n"
            f"This could be because the query is too general, uses unfamiliar phrasing, "
            f"or asks for something outside the available data sources.\n\n"
            f"### Try Rephrasing Your Query\n\n"
            f"Here are some examples that work well:\n\n"
            f"{suggestions_text}\n\n"
            f"### Tips for Better Results\n\n"
            f"- **Be specific** — mention what you want: agents, alerts, vulnerabilities, or network data\n"
            f"- **Include identifiers** — agent IDs (e.g., agent 019), rule IDs (e.g., rule 80255), CVE IDs, or IP addresses\n"
            f"- **Specify time** — add \"last 24 hours\", \"last 7 days\" for time-bound queries\n"
            f"- **Use MITRE terms** — T-codes (T1110), tactics (credential access), techniques (brute force)\n"
            f"- **Ask for correlations** — \"agents with critical vulnerabilities\", \"alerts with agent context\"\n\n"
            f"### Available Capabilities\n\n"
            f"| Category | What You Can Ask |\n"
            f"|----------|------------------|\n"
            f"| Agents | Status, health, processes, ports, config |\n"
            f"| Alerts | Recent alerts, severity breakdown, top rules, patterns |\n"
            f"| Vulnerabilities | CVEs by severity, affected agents, risk scores |\n"
            f"| Search | MITRE tactics/techniques, rule IDs, security events |\n"
            f"| Suricata | Network alerts, HTTP traffic, JA3/JA4, signatures |\n"
            f"| Correlations | Cross-source: agents + vulns, alerts + agents |\n"
            f"| Decoders | Generate, test, and explain Wazuh decoders |\n"
            f"| IOC | IP reputation via AbuseIPDB |\n"
        )

    def _build_search_events_args(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build args for search_security_events with intelligent query translation.

        Strategy (priority order):
        1. Structured entity filters (rule_id, mitre_id, level, mitre_tactic) → exact param filters
        2. MITRE technique keyword mapping → mitre_technique param
        3. Security keyword extraction → free-text query with OR joins
        4. NL boilerplate stripping → fallback free-text
        """
        args = {"time_range": entities.get("time_range", "24h"), "limit": 100}

        # Forward agent_id when present (regardless of which structured filter wins)
        if entities.get("agent_id"):
            args["agent_id"] = entities["agent_id"]

        # PRIORITY 1: Structured entity filters — use exact params, skip free-text
        if entities.get("rule_id"):
            args["rule_id"] = entities["rule_id"]
            args["query"] = "*"
            logger.info(f"[SEARCH_ARGS] Structured filter: rule_id={entities['rule_id']}")
            return args
        if entities.get("mitre_id"):
            args["mitre_id"] = entities["mitre_id"]
            args["query"] = "*"
            logger.info(f"[SEARCH_ARGS] Structured filter: mitre_id={entities['mitre_id']}")
            return args
        if entities.get("mitre_technique"):
            # User typed an exact technique name (e.g., "PowerShell", "Modify Registry")
            args["mitre_technique"] = entities["mitre_technique"]
            args["query"] = "*"
            logger.info(f"[SEARCH_ARGS] Structured filter: mitre_technique={entities['mitre_technique']}")
            return args
        if entities.get("level"):
            args["level"] = int(entities["level"])
            args["query"] = "*"
            logger.info(f"[SEARCH_ARGS] Structured filter: level={entities['level']}")
            return args
        if entities.get("mitre_tactic"):
            args["mitre_tactic"] = entities["mitre_tactic"]
            args["query"] = "*"
            logger.info(f"[SEARCH_ARGS] Structured filter: mitre_tactic={entities['mitre_tactic']}")
            return args
        # severity → translate to a level range filter (critical=15, high=12+, medium=7+, low=4+)
        if entities.get("severity"):
            sev = str(entities["severity"]).lower()
            sev_level_floor = {"critical": 12, "high": 7, "medium": 4, "low": 1}.get(sev)
            if sev_level_floor is not None:
                args["min_level"] = sev_level_floor
                args["query"] = "*"
                logger.info(f"[SEARCH_ARGS] Severity filter: {sev} → min_level={sev_level_floor}")
                return args

        # PRIORITY 2: MITRE technique keyword mapping (covers partial / informal names)
        MITRE_TECHNIQUES = {
            "brute force": "Brute Force",
            "phishing": "Phishing",
            "spearphishing": "Spearphishing Attachment",
            "credential dumping": "OS Credential Dumping",
            "pass the hash": "Pass the Hash",
            "kerberoasting": "Kerberoasting",
            "golden ticket": "Steal or Forge Kerberos Tickets",
            "valid accounts": "Valid Accounts",
            "local accounts": "Local Accounts",                              # NEW
            "software deployment": "Software Deployment Tools",
            "email forwarding": "Email Forwarding Rule",
            "file deletion": "File Deletion",
            "ingress tool transfer": "Ingress Tool Transfer",
            "domain policy modification": "Domain Policy Modification",
            "stored data manipulation": "Stored Data Manipulation",
            "exploit public facing": "Exploit Public-Facing Application",
            "command injection": "Command and Scripting Interpreter",
            "sql injection": "Exploit Public-Facing Application",
            "web shell": "Server Software Component",
            "rootkit": "Rootkit",
            "keylogger": "Input Capture",
            "dns tunneling": "Protocol Tunneling",
            "port scanning": "Network Service Discovery",
            "account manipulation": "Account Manipulation",
            "data staging": "Data Staged",
            "scheduled task": "Scheduled Task/Job",
            # Top techniques on the user's dashboard
            "powershell": "PowerShell",                                       # NEW
            "modify registry": "Modify Registry",                             # NEW
            "application layer protocol": "Application Layer Protocol",       # NEW
            "masquerading": "Masquerading",                                   # NEW
            "process injection": "Process Injection",                         # NEW
            "service execution": "Service Execution",                         # NEW
            "obfuscated files": "Obfuscated Files or Information",            # NEW
            "system information discovery": "System Information Discovery",   # NEW
            "remote services": "Remote Services",                             # NEW
            "lateral tool transfer": "Lateral Tool Transfer",                 # NEW
            "data from local system": "Data from Local System",               # NEW
            "exfiltration over c2": "Exfiltration Over C2 Channel",           # NEW
            "impair defenses": "Impair Defenses",                             # NEW
            "indicator removal": "Indicator Removal",                         # NEW
        }
        query_lower = query.lower()
        for keyword, technique_name in MITRE_TECHNIQUES.items():
            if keyword in query_lower:
                args["mitre_technique"] = technique_name
                args["query"] = "*"
                logger.info(f"[SEARCH_ARGS] MITRE technique match: '{keyword}' → {technique_name}")
                return args

        # PRIORITY 3: Security keyword extraction (free-text with fields search)
        SECURITY_KEYWORDS = [
            r'virustotal', r'suricata', r'ossec', r'fim', r'syscheck', r'rootcheck',
            r'malware', r'authentication', r'login failure',
            r'ssh', r'rdp', r'powershell', r'mimikatz', r'ransomware', r'trojan',
            r'backdoor', r'exploit', r'lateral\s+movement', r'privilege\s+escalation',
            r'data\s+exfiltration', r'c2', r'command\s+and\s+control',
            r'aws', r'cloudtrail', r'iam', r's3\s+bucket', r'azure', r'azure\s+ad',
            r'o365', r'office\s*365', r'github', r'repository',
            r'mfa', r'multi[\s\-]factor', r'2fa', r'oauth', r'api\s+key', r'api\s+token',
            r'kerberos', r'kerberoast', r'ntlm',
            r'account\s+lockout', r'password\s+policy', r'password\s+reset',
            r'disabled\s+account', r'service\s+account',
            r'group\s+membership', r'admin\s+group',
            r'scheduled\s+task', r'schtasks', r'service\s+creation', r'service\s+install',
            r'encoded\s+command', r'base64', r'wmic', r'mshta', r'regsvr32', r'certutil',
            r'usb', r'removable\s+media', r'removable\s+device',
            r'file\s+access', r'object\s+access',
            r'remote\s+login', r'remote\s+desktop', r'logon\s+type',
        ]
        found_keywords = []
        for kw_pattern in SECURITY_KEYWORDS:
            m = re.search(kw_pattern, query, re.IGNORECASE)
            if m:
                found_keywords.append(m.group(0).lower())

        if found_keywords:
            args["query"] = " OR ".join(found_keywords)
            logger.info(f"[SEARCH_ARGS] Keyword extraction: {found_keywords}")
            return args

        # PRIORITY 4: Strip NL boilerplate and use what remains
        query_clean = query.lower().strip()
        query_clean = re.sub(r'\b(in\s+)?(the\s+)?(last|past)\s+\d+\s+(hour|day|week|month)s?\b', '', query_clean)
        query_clean = re.sub(r'\b\d+(h|d|w|m)\b', '', query_clean)

        boilerplate = [
            "please", "search", "find", "lookup", "look up", "show me", "show",
            "get", "list", "display", "fetch", "retrieve", "query",
            "security events", "security event", "security",
            "events", "event", "alerts", "alert",
            "in wazuh", "in the system", "from", "the", "for me",
            "in last", "in the last", "for the last",
            "hours", "hour", "days", "day", "last", "past", "in",
        ]
        for phrase in sorted(boilerplate, key=len, reverse=True):
            query_clean = query_clean.replace(phrase, " ")

        query_clean = re.sub(r'[^a-z0-9\s]', ' ', query_clean)
        search_query = " ".join(query_clean.split()).strip()

        if not search_query or len(search_query) < 2:
            search_query = "*"

        args["query"] = search_query
        logger.info(f"[SEARCH_ARGS] Fallback text: query='{search_query}'")
        return args

    def _build_indicator_args(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX R4: Build proper args for analyze_security_threat / check_ioc_reputation.
        Extracts the indicator (IP, hash, domain) from the user's query.
        """
        args = {}

        # Extract time range if present
        time_range = entities.get("time_range", "30d")

        # Try to extract IP address
        ip_match = re.search(
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', query
        )
        if ip_match:
            args["indicator"] = ip_match.group(1)
            args["indicator_type"] = "ip"
            logger.info(f"[INDICATOR_ARGS] Extracted IP: {args['indicator']}")
            return args

        # Try to extract hash (MD5, SHA1, SHA256)
        hash_match = re.search(r'\b([a-fA-F0-9]{32,64})\b', query)
        if hash_match:
            h = hash_match.group(1)
            args["indicator"] = h
            args["indicator_type"] = "hash"
            logger.info(f"[INDICATOR_ARGS] Extracted hash: {h[:16]}...")
            return args

        # Try to extract domain
        domain_match = re.search(
            r'\b([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
            r'(?:\.[a-zA-Z]{2,})+)\b', query
        )
        if domain_match:
            domain = domain_match.group(1)
            # Exclude common non-indicator words
            if domain not in ("wazuh.com", "example.com"):
                args["indicator"] = domain
                args["indicator_type"] = "domain"
                logger.info(f"[INDICATOR_ARGS] Extracted domain: {domain}")
                return args

        # Fallback: strip boilerplate and use remaining text as indicator
        query_clean = query.lower().strip()
        boilerplate = [
            "please", "check", "verify", "lookup", "look up", "analyze", "analyse",
            "ioc", "reputation", "threat", "indicator", "security",
            "for", "the", "of", "a", "an", "this", "that",
        ]
        for phrase in sorted(boilerplate, key=len, reverse=True):
            query_clean = query_clean.replace(phrase, " ")
        indicator = " ".join(query_clean.split()).strip()

        if indicator and len(indicator) >= 2:
            args["indicator"] = indicator
            args["indicator_type"] = "ip"  # Default
        else:
            args["indicator"] = "unknown"
            args["indicator_type"] = "ip"

        logger.info(f"[INDICATOR_ARGS] Fallback indicator: {args['indicator']}")
        return args

    def _build_plan_from_config(
        self,
        config: Dict[str, Any],
        entities: Dict[str, Any],
        selection_method: str,
        reasoning: str,
        user_query: str = ""
    ) -> ToolPlan:
        """Build ToolPlan from pattern configuration"""

        primary_tool = config["tool"]
        secondary_tools = config.get("secondary", [])
        base_args = config.get("args", {})

        # Build primary arguments
        primary_args = self._build_arguments(primary_tool, entities, base_args, user_query=user_query)

        # FIX R5.1: Inject agent_id for agent tools even if not in schema
        if "agent_id" in entities:
            if primary_tool == "get_wazuh_agents":
                primary_args["agent_id"] = entities["agent_id"]
            for tool in secondary_tools:
                pass  # secondary args built below will also get agent_id injection

        # Build secondary arguments
        secondary_args = [
            self._build_arguments(tool, entities, user_query=user_query)
            for tool in secondary_tools
        ]
        # FIX R5.1: Inject agent_id into secondary args for agent-specific tools
        if "agent_id" in entities:
            for i, tool in enumerate(secondary_tools):
                if tool in ("get_wazuh_vulnerabilities", "get_agent_ports", "get_agent_processes",
                            "check_agent_health", "get_agent_configuration"):
                    secondary_args[i]["agent_id"] = entities["agent_id"]

        # Determine correlation
        requires_correlation = len(secondary_tools) > 0 and config.get("correlation") is not None
        correlation_strategy = config.get("correlation")

        return ToolPlan(
            primary_tool=primary_tool,
            secondary_tools=secondary_tools,
            primary_args=primary_args,
            secondary_args=secondary_args,
            confidence=config.get("confidence", 0.8),
            selection_method=selection_method,
            reasoning=reasoning,
            requires_correlation=requires_correlation,
            correlation_strategy=correlation_strategy
        )

    # Query canonicalization: rewrite common question forms to statement forms.
    # Runs BEFORE synonym expansion so regex patterns matching "show X" catch
    # "how many X", "what are X", "which X" too. Ordered longest-first to avoid
    # partial replacement conflicts.
    CANONICALIZATION_RULES = [
        # Phase 4.1: time-phrase normalization to the canonical "last N <unit>" form
        # that the existing time_range entity regex already matches. Order matters:
        # specific phrases must be tried before generic stems.
        (r"\b(in\s+the\s+)?(last|past|over\s+the\s+(?:last|past))\s+(\d+)\s*(h|hr|hrs|hour|hours)\b", r"last \3 hours"),
        (r"\b(in\s+the\s+)?(last|past|over\s+the\s+(?:last|past))\s+(\d+)\s*(d|day|days)\b", r"last \3 days"),
        (r"\b(in\s+the\s+)?(last|past|over\s+the\s+(?:last|past))\s+(\d+)\s*(w|wk|wks|week|weeks)\b", r"last \3 weeks"),
        (r"\b(in\s+the\s+)?(last|past|over\s+the\s+(?:last|past))\s+(\d+)\s*(mo|mon|month|months)\b", r"last \3 months"),
        (r"\b(\d+)\s*(?:h|hr|hrs|hour|hours)\s+ago\b", r"last \1 hours"),
        (r"\b(\d+)\s*(?:d|day|days)\s+ago\b", r"last \1 days"),
        (r"\b(\d+)\s*(?:w|wk|wks|week|weeks)\s+ago\b", r"last \1 weeks"),
        (r"\b(today|in\s+the\s+last\s+24h|within\s+24\s+hours)\b", "last 24 hours"),
        (r"\b(yesterday|day\s+before|prior\s+day)\b", "last 24 hours"),
        (r"\b(this\s+week|current\s+week|past\s+week|past\s+7\s+days)\b", "last 7 days"),
        (r"\b(last\s+week)\b", "last 7 days"),
        (r"\b(this\s+month|current\s+month|past\s+month|past\s+30\s+days)\b", "last 30 days"),
        (r"\b(last\s+month)\b", "last 30 days"),
        (r"\b(now|currently|at\s+the\s+moment|right\s+now)\b", "last 1 hours"),
        # Phase 4.4: light stemming — normalize morphology so downstream patterns
        # see consistent canonical forms.
        (r"\bvulns?\b", "vulnerability"),
        (r"\bvulnerabilities\b", "vulnerability"),
        (r"\bvulnerable\b", "vulnerability"),
        (r"\balerting\b", "alert"),
        (r"\balerted\b", "alert"),
        (r"\bdetections\b", "detection"),
        (r"\bagents'\b", "agents"),
        (r"\balerts'\b", "alerts"),

        # Counting intent -> statement form
        (r"\bhow\s+many\s+", "count "),
        (r"\bwhat('?s|\s+is)\s+the\s+count\s+of\s+", "count "),
        # Which/what X -> show X
        (r"\bwhich\s+agents?\s+(are|is)\s+", "show agents "),
        (r"\bwhich\s+agents?\s+have\s+", "show agents with "),
        (r"\bwhat\s+agents?\s+(are|have)\s+", "show agents "),
        (r"\bwhich\s+(cves?|vulnerabilities?)\s+", "show vulnerabilities "),
        (r"\bwhat\s+(cves?|vulnerabilities?)\s+", "show vulnerabilities "),
        (r"\bwhich\s+(alerts?|incidents?)\s+", "show alerts "),
        (r"\bwhat\s+(alerts?|incidents?)\s+", "show alerts "),
        # Is/are X ... -> show X (generic status questions)
        (r"\bis\s+my\s+system\s+compromised\??", "show top threats"),
        (r"\banything\s+suspicious\s+(today|now|happening)?\??", "show suspicious activity"),
        (r"\banything\s+(bad|wrong|weird)\s+(today|now|happening)?\??", "show top threats"),
        (r"\bwhat'?s?\s+wrong\s+with\s+agent\s+(\d+)\b", r"check agent \1 health"),
        (r"\bwhat'?s?\s+the\s+status\s+of\s+agent\s+(\d+)\b", r"check agent \1 health"),
        (r"\bis\s+agent\s+(\d+)\s+(ok|healthy|up)\??", r"check agent \1 health"),
        # Generic question starters
        (r"^can\s+you\s+(show|list|display|give|share)\s+", r"\1 "),
        (r"^could\s+you\s+(show|list|display|give|share)\s+", r"\1 "),
        (r"^please\s+(show|list|display|give|share)\s+", r"\1 "),
        (r"^i\s+(want|need)\s+to\s+see\s+", "show "),
        (r"^give\s+me\s+", "show "),
        (r"^tell\s+me\s+about\s+", "show "),
        # User said "share/shared/sharing/bring/pull (up)" — collapse to "show"
        # so the regex layer's existing "show X" patterns catch the query.
        (r"^\s*shared?\s+", "show "),
        (r"^\s*sharing\s+", "show "),
        (r"^\s*bring\s+(?:me\s+)?", "show "),
        (r"^\s*pull\s+up\s+", "show "),
        (r"^\s*pull\s+", "show "),
        (r"^\s*fetch\s+(?:me\s+)?", "show "),
        (r"^\s*display\s+", "show "),
        # Trailing question marks / polite suffixes
        (r"\?+\s*$", ""),
        (r"\bplease\s*$", ""),
    ]

    def _canonicalize_query(self, query_lower: str) -> str:
        """
        Convert common question forms to statement forms.
        Lets the regex layer catch "how many critical alerts" the same way it catches "count critical alerts".
        Non-destructive: only triggers on recognized forms; leaves unknown queries unchanged.
        """
        canon = query_lower
        for pattern, replacement in self.CANONICALIZATION_RULES:
            canon = re.sub(pattern, replacement, canon, flags=re.IGNORECASE)
        canon = " ".join(canon.split())  # collapse whitespace
        return canon

    # SOC acronym / synonym expansion applied BEFORE regex matching.
    # Keeps original user phrase intact but adds canonical tokens so patterns fire.
    # Safe because it only appends synonyms; never removes user-typed words.
    SYNONYM_EXPANSIONS = [
        # Phase 4.2: spelled-out compound nouns -> include their acronym so
        # acronym-based routing patterns also catch the spelled-out forms.
        (r"\bendpoint\s+detection\s+(?:and\s+)?response\b", "edr endpoint detection response"),
        (r"\bmanaged\s+detection\s+(?:and\s+)?response\b", "mdr managed detection response"),
        (r"\bextended\s+detection\s+(?:and\s+)?response\b", "xdr extended detection response"),
        (r"\bsecurity\s+information\s+(?:and\s+)?event\s+management\b", "siem security information event management"),
        (r"\bsecurity\s+orchestration\s+(?:automation\s+(?:and\s+)?response|and\s+automated\s+response)\b", "soar security orchestration response"),
        (r"\bthreat\s+(?:and\s+)?vulnerability\s+management\b", "tvm vulnerability"),
        (r"\bhost\s+intrusion\s+detection(?:\s+system)?\b", "hids host intrusion detection"),
        (r"\bnetwork\s+intrusion\s+detection(?:\s+system)?\b", "nids network intrusion detection"),
        (r"\bnids\b", "nids network intrusion detection"),
        (r"\bhids\b", "hids host intrusion detection"),
        (r"\bnips\b", "nips network intrusion prevention"),
        (r"\bhips\b", "hips host intrusion prevention"),
        (r"\bnetflow\b", "netflow network flow"),
        (r"\bc2\b", "c2 command and control"),
        (r"\bc&c\b", "c2 command and control"),
        (r"\btlp\b", "tlp threat"),
        (r"\bioc\b", "ioc indicator"),
        (r"\btt[pP]s?\b", "ttp tactic technique"),
        (r"\bedr\b", "edr endpoint detection"),
        (r"\bxdr\b", "xdr extended detection"),
        (r"\bsiem\b", "siem wazuh"),
        # Defender — bare "defender" should be recognized as Microsoft Defender.
        # Guard: don't expand when already preceded by "microsoft|windows|ms".
        (r"(?<!microsoft\s)(?<!windows\s)(?<!ms\s)\bdefender\b", "microsoft defender"),
        (r"\bms\s+defender\b", "microsoft defender"),
        (r"\bwindows\s+defender\b", "microsoft defender"),
        (r"\bmsdefender\b", "microsoft defender"),
        (r"\bmsft\s+defender\b", "microsoft defender"),
        # CIS / CAS / CS / SCA compliance — all map to SCA (Security Configuration Assessment).
        (r"\bcis\s+compliance\b", "sca security configuration assessment cis"),
        (r"\bcas\s+compliance\b", "sca security configuration assessment cis"),
        (r"\bcs\s+compliance\b", "sca security configuration assessment cis"),
        (r"\bcis\s+benchmarks?\b", "sca security configuration assessment cis"),
        (r"\bsecurity\s+configuration\s+assessment\b", "sca security configuration assessment"),
        (r"\bscap\b", "sca scap security configuration assessment"),
    ]

    def _normalize_synonyms(self, query_lower: str) -> str:
        """
        Expand common SOC acronyms so regex/keyword patterns catch synonymous terms.
        Appends canonical forms rather than replacing, to preserve user phrasing.
        Example: "show nids alerts" -> "show nids network intrusion detection alerts"
        """
        expanded = query_lower
        for pattern, expansion in self.SYNONYM_EXPANSIONS:
            if re.search(pattern, expanded):
                # Only replace first match to avoid duplicate expansion on already-expanded text
                expanded = re.sub(pattern, expansion, expanded, count=1)
        return expanded

    def set_tool_executor(self, executor) -> None:
        """Allow QueryOrchestrator to inject the ToolExecutor (so the selector
        can resolve agent names to IDs by calling get_wazuh_agents)."""
        self._tool_executor = executor

    async def _resolve_agent_name_to_id(self, name: str) -> Optional[str]:
        """Resolve a captured agent name (e.g., 'test-sensor') to its numeric
        agent_id ('071') via a cached call to get_wazuh_agents. Returns None
        when not found or when the executor isn't wired up.

        Match is case-insensitive exact. Cache TTL is 5 minutes.
        """
        if self._tool_executor is None:
            return None
        name_lc = (name or "").strip().lower()
        if not name_lc:
            return None

        # Refresh cache if stale or empty
        if not self._agent_name_cache or (
            time.time() - self._agent_name_cache_ts > self._agent_name_cache_ttl
        ):
            try:
                resp = await self._tool_executor.execute(
                    "get_wazuh_agents", {"limit": 500}
                )
                items = []
                if hasattr(resp, "data"):
                    items = (resp.data or {}).get("data", {}).get("affected_items", [])
                elif isinstance(resp, dict):
                    items = resp.get("data", {}).get("affected_items", [])
                self._agent_name_cache = {
                    str(a.get("name", "")).strip().lower():
                        str(a.get("id", "")).strip().zfill(3)
                    for a in items
                    if a.get("name") and a.get("id") is not None
                }
                self._agent_name_cache_ts = time.time()
                logger.info(
                    f"[ENTITY] Refreshed agent name cache: "
                    f"{len(self._agent_name_cache)} agents"
                )
            except Exception as e:
                logger.warning(f"[ENTITY] Agent name cache refresh failed: {e}")
                return None

        # Exact match (existing behavior)
        rid = self._agent_name_cache.get(name_lc)
        if rid:
            return rid
        # Fuzzy fallback: prefix match, then substring. This lets the analyst
        # type "ushna" and resolve to "ushna-macos", "dev-z" to "dev-zeek", etc.
        cache_items = list(self._agent_name_cache.items())
        # 1) prefix: cached name starts with the query, or query starts with cached
        prefix_matches = [(n, i) for (n, i) in cache_items if n.startswith(name_lc) or name_lc.startswith(n)]
        # Tighten: if MULTIPLE prefix matches, prefer the one with shortest cached name
        if len(prefix_matches) == 1:
            n, rid = prefix_matches[0]
            logger.info(f"[ENTITY] fuzzy(prefix) resolved {name!r} -> {n!r} = {rid}")
            return rid
        if len(prefix_matches) > 1:
            n, rid = min(prefix_matches, key=lambda x: len(x[0]))
            logger.info(f"[ENTITY] fuzzy(prefix, ambiguous) {name!r} -> {n!r} = {rid} (of {len(prefix_matches)} candidates)")
            return rid
        # 2) substring: cached name contains query, or query contains cached name
        contains_matches = [(n, i) for (n, i) in cache_items if name_lc in n or n in name_lc]
        if len(contains_matches) == 1:
            n, rid = contains_matches[0]
            logger.info(f"[ENTITY] fuzzy(substring) resolved {name!r} -> {n!r} = {rid}")
            return rid
        if len(contains_matches) > 1:
            n, rid = min(contains_matches, key=lambda x: len(x[0]))
            logger.info(f"[ENTITY] fuzzy(substring, ambiguous) {name!r} -> {n!r} = {rid} (of {len(contains_matches)} candidates)")
            return rid
        # 3) spell-tolerance via difflib.get_close_matches (handles typos)
        try:
            import difflib as _df
            _cnames = [n for (n, _) in cache_items]
            _close = _df.get_close_matches(name_lc, _cnames, n=1, cutoff=0.72)
            if _close:
                rid = self._agent_name_cache.get(_close[0])
                logger.info(f"[ENTITY] fuzzy(spell) resolved {name!r} -> {_close[0]!r} = {rid}")
                return rid
            # 4) space-normalized retry: "dev zeak" -> "dev-zeak" / "devzeak"
            _alt = name_lc.replace(" ", "-")
            if _alt != name_lc:
                rid = self._agent_name_cache.get(_alt)
                if rid:
                    return rid
                _close = _df.get_close_matches(_alt, _cnames, n=1, cutoff=0.72)
                if _close:
                    rid = self._agent_name_cache.get(_close[0])
                    logger.info(f"[ENTITY] fuzzy(spell+space) resolved {name!r} -> {_close[0]!r} = {rid}")
                    return rid
        except Exception as _de:
            logger.debug(f"[ENTITY] difflib step failed: {_de}")
        logger.info(f"[ENTITY] could not resolve agent name {name!r} (cache has {len(cache_items)} agents)")
        return None

    async def _resolve_agent_names_to_ids(self, names):
        """Batch variant of _resolve_agent_name_to_id with fuzzy fallback per name.
        Returns a dict mapping each input name -> resolved id (or None)."""
        result = {}
        if not names:
            return result
        # Cache refresh via the first name (single resolver handles cache load)
        await self._resolve_agent_name_to_id(names[0])
        for n in names:
            if not (n or "").strip():
                result[n] = None
                continue
            # Call the single resolver (which now does exact + fuzzy) for each.
            result[n] = await self._resolve_agent_name_to_id(n)
        return result

    def _is_complex_hunt_query(self, query: str) -> bool:
        """Detect multi-indicator hunt queries that should go to the LLM router.

        Indicators:
          - The verb "correlate" combined with 2+ specific signals
          - "find systems with" / "hunt for" / "detect" + 2+ concept clauses
          - 3+ comma-separated items in the query
          - "and" appearing 2+ times in a hunt-like phrasing

        When True, callers should skip deterministic Strategy 0.4 (keyword
        overrides) and Strategy 0.45 (correlation preempt) and let Strategy
        0.5 (LLM router) handle the query end-to-end.
        """
        if not query or len(query) < 25:
            return False
        q = query.lower()
        # Hunt verbs
        hunt_verbs = ("correlate", "find systems with", "hunt for", "detect ", "look for", "investigate ")
        has_hunt_verb = any(v in q for v in hunt_verbs)
        # Comma count
        comma_items = q.count(",")
        # "and" count
        and_count = len(re.findall(r"\band\b", q))
        # Specific-signal vocabulary (security verbs/nouns that are common in
        # hunt queries and indicate the analyst wants multiple things checked).
        signals = (
            "login", "logon", "logins", "access", "dumping", "creation",
            "escalation", "exfiltration", "persistence", "movement",
            "execution", "credential", "rdp", "smb", "usb", "scheduled task",
            "configuration change", "registry", "process injection",
            "admin account", "service account", "lateral", "beacon", "c2",
        )
        signal_hits = sum(1 for s in signals if s in q)
        # Heuristics — must be deliberate, not trigger-happy
        # H1: explicit hunt verb + at least 2 signals
        if has_hunt_verb and signal_hits >= 2:
            return True
        # H2: 3+ comma-separated items AND 1+ signal AND 1+ "and"
        if comma_items >= 2 and signal_hits >= 1 and and_count >= 1:
            return True
        # H3: 2+ "and" + 3+ signals (no commas, but lots of conjunctions)
        if and_count >= 2 and signal_hits >= 3:
            return True
        return False

    def _canonicalize_os(self, raw: str) -> str:
        """Map captured OS terms to canonical platform values used by the indexer.

        Returns one of: 'linux', 'windows', 'darwin' — or the original value
        (lowercased) if it doesn't match a known family. The indexer's bool/should
        OS clause is tolerant; this canonicalization just gives predictable
        downstream behavior in the formatter / correlation layers.
        """
        s = (raw or "").lower().replace(" ", "").strip()
        if s in ("windows", "win", "winserver", "winserv", "win10", "win11"):
            return "windows"
        if s in (
            "linux", "ubuntu", "debian", "centos", "redhat", "rhel",
            "fedora", "alpine", "kali", "amazonlinux", "amzn", "suse",
        ):
            return "linux"
        if s in ("macos", "mac", "darwin", "osx"):
            return "darwin"
        return s

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from query"""

        entities = {}

        # Phase 4.1: canonicalize the query first so time phrases like "today",
        # "yesterday", "this week", "2 hours ago" become "last N <unit>" which
        # the ENTITY_PATTERNS time_range regex catches.
        try:
            _canon = self._canonicalize_query(query.lower())
            _entity_query = _canon
        except Exception:
            _entity_query = query

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, _entity_query, re.IGNORECASE)
            if matches:
                if entity_type == "time_range":
                    # Parse time range into standard format.
                    # Units mapped to OpenSearch date-math letters:
                    #   minute(s)/min(s) -> "m" (lowercase m = minutes in OS)
                    #   hour(s)/hr(s)    -> "h"
                    #   day(s)/d         -> "d"
                    #   week(s)/wk(s)    -> "w"
                    amount, unit = matches[0]
                    _u = (unit or "").lower().strip()
                    _u_letter = {
                        "minute": "m", "min": "m", "mins": "m",
                        "hour": "h", "hr": "h", "hrs": "h",
                        "day": "d", "d": "d",
                        "week": "w", "wk": "w", "wks": "w",
                    }.get(_u, _u[:1] if _u else "h")
                    entities[entity_type] = f"{amount}{_u_letter}"  # e.g., "30m", "24h", "7d"


                elif entity_type == "framework":
                    # Normalize framework name
                    framework = matches[0].upper()
                    if framework == "PCI":
                        framework = "PCI-DSS"
                    entities[entity_type] = framework

                elif entity_type == "cve_id":
                    # Phase 4.3: Rebuild canonical "CVE-YYYY-NNNN" form whether
                    # the user typed "CVE 2024 1234", "CVE-2024-1234", or "CVE2024-1234".
                    _m = matches[0]
                    if isinstance(_m, tuple) and len(_m) == 2:
                        entities[entity_type] = f"CVE-{_m[0]}-{_m[1]}"
                    else:
                        entities[entity_type] = str(_m)

                elif entity_type == "status":
                    # Normalize status to canonical Wazuh values
                    raw = matches[0].lower().replace("_", " ").strip()
                    STATUS_SYNONYM_MAP = {
                        "active": "active", "running": "active", "connected": "active",
                        "online": "active", "alive": "active", "healthy": "active",
                        "operational": "active", "responding": "active", "live": "active",
                        "up": "active",
                        "disconnected": "disconnected", "offline": "disconnected",
                        "inactive": "disconnected", "down": "disconnected",
                        "dead": "disconnected", "unreachable": "disconnected",
                        "lost": "disconnected", "stale": "disconnected",
                        "never connected": "never_connected", "never_connected": "never_connected",
                        "pending": "never_connected", "unenrolled": "never_connected",
                        "unregistered": "never_connected",
                    }
                    entities[entity_type] = STATUS_SYNONYM_MAP.get(raw, raw)

                elif entity_type == "mitre_tactic":
                    # Title-case for Elasticsearch matching (e.g., "credential access" → "Credential Access")
                    entities[entity_type] = matches[0].strip().title()

                elif entity_type == "mitre_technique":
                    # Title-case + normalize "PowerShell" specifically (wazuh stores
                    # it as "PowerShell" not "Powershell"). Other techniques use
                    # standard Title Case which matches Wazuh's rule.mitre.technique.
                    raw = str(matches[0]).strip()
                    titled = raw.title()
                    if titled.lower() == "powershell":
                        titled = "PowerShell"
                    entities[entity_type] = titled

                elif entity_type == "mitre_id":
                    # Uppercase T-code (e.g., "t1110" → "T1110")
                    entities[entity_type] = matches[0].strip().upper()

                elif entity_type == "os":
                    # Canonicalize OS to indexer values: linux / windows / darwin
                    entities[entity_type] = self._canonicalize_os(matches[0])

                elif entity_type == "agent_name":
                    # Skip — handled by the token-scan pass below (which correctly
                    # handles overlapping prefixes like "on agent X" where regex
                    # findall would only see "agent" as the first match).
                    pass

                else:
                    entities[entity_type] = matches[0]

        # Extra agent_id extraction (numeric IDs like 001) when query mentions 'agent'
        if "agent_id" not in entities:
            m_id = re.search(r"\bagent\s*(?:id)?\s*[:#\-]?\s*(\d{1,6})\b", query, re.IGNORECASE)
            if m_id:
                raw = m_id.group(1)
                entities["agent_id"] = raw.zfill(3) if len(raw) <= 3 else raw

        # Token-scan agent_name extraction. Regex findall with prefix-anchored
        # patterns can't catch "on agent NAME" because it greedily consumes
        # "on agent" first and captures only "agent" (a stop word). Iterating
        # tokens manually finds the real name regardless of prefix order.
        if "agent_name" not in entities:
            stop_words = {
                "agent", "agents", "host", "hosts", "all", "active",
                "disconnected", "windows", "linux", "macos", "darwin",
                "critical", "high", "medium", "low", "vulnerability",
                "vulnerabilities", "alerts", "events", "ports", "processes",
                "the", "this", "that", "these", "those", "any", "every",
                "system", "systems", "endpoint", "endpoints",
                "machine", "machines", "server", "servers",
            }
            prefix_words = {"agent", "host", "machine", "endpoint", "server", "for", "on"}
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-.]+", query)
            for i, tok in enumerate(tokens[:-1]):
                if tok.lower() in prefix_words:
                    cand = tokens[i + 1].rstrip(".,;:?!").strip()
                    if (
                        cand
                        and cand.lower() not in stop_words
                        and not cand.isdigit()
                        and len(cand) >= 3
                    ):
                        entities["agent_name"] = cand
                        break

        # ----------------------------------------------------------------
        # NLP Multi-entity: collect all matches (not just the first) for
        # agent_id, agent_name, cve_id, ip, so the orchestrator can fan
        # out per-entity tool calls when the analyst names multiple things.
        # The singular keys above stay populated for back-compat; list-plural
        # keys are added only when 2+ matches are found.
        # ----------------------------------------------------------------
        try:
            _multi_text = (_entity_query or query or "")
            # agent_id: all numeric matches after "agent" prefix
            _aid_re = re.compile(r"\bagent\s*(?:id)?\s*[:#\-]?\s*(\d{1,6})\b", re.IGNORECASE)
            _aids = []
            for m in _aid_re.finditer(_multi_text):
                raw = m.group(1)
                norm = raw.zfill(3) if len(raw) <= 3 else raw
                if norm not in _aids:
                    _aids.append(norm)
            if len(_aids) >= 2:
                entities["agent_ids"] = _aids
                logger.info(f"[MULTI-ENTITY] detected {len(_aids)} agent_ids: {_aids}")

            # agent_name: walk tokens and gather every name that follows a
            # known prefix-word (agent, host, machine, endpoint, server,
            # for, on). Deduplicate case-insensitively, preserve order.
            _stop_words = {
                "agent", "agents", "host", "hosts", "all", "active",
                "disconnected", "windows", "linux", "macos", "darwin",
                "critical", "high", "medium", "low", "vulnerability",
                "vulnerabilities", "alerts", "events", "ports", "processes",
                "the", "this", "that", "these", "those", "any", "every",
                "system", "systems", "endpoint", "endpoints",
                "machine", "machines", "server", "servers",
                "and", "or", "with", "vs", "versus", "between", "compare",
                "performance", "details", "health", "status",
            }
            _prefix_words = {"agent", "host", "machine", "endpoint", "server", "for", "on"}
            _connectives = {"and", "or", "vs", "versus", ",", "&", "plus"}
            _tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-.]+|,", query)
            _names = []
            _seen_lc = set()
            i = 0
            _prev_was_name = False
            while i < len(_tokens) - 1:
                tok = _tokens[i]
                tok_lc = tok.lower()
                # Capture name when preceded by a prefix word (existing behavior)
                # OR by a connective when the previous capture was a name. This
                # lets "compare X and Y" pull both X (after "compare/for") and
                # Y (after "and").
                trigger = (
                    tok_lc in _prefix_words
                    or (tok_lc in _connectives and _prev_was_name)
                )
                if trigger:
                    cand = _tokens[i + 1].rstrip(".,;:?!").strip()
                    cl = cand.lower()
                    if (
                        cand and cl not in _stop_words and cl not in _connectives
                        and not cand.isdigit() and len(cand) >= 3 and cl not in _seen_lc
                    ):
                        _names.append(cand)
                        _seen_lc.add(cl)
                        _prev_was_name = True
                        i += 2
                        continue
                _prev_was_name = False
                i += 1
            if len(_names) >= 2:
                entities["agent_names"] = _names
                logger.info(f"[MULTI-ENTITY] detected {len(_names)} agent_names: {_names}")

            # cve_id: all matches
            _cve_re = re.compile(r"\bCVE[\s\-]?(\d{4})[\s\-]?(\d{4,7})\b", re.IGNORECASE)
            _cves = []
            for m in _cve_re.finditer(_multi_text):
                canon = f"CVE-{m.group(1)}-{m.group(2)}"
                if canon not in _cves:
                    _cves.append(canon)
            if len(_cves) >= 2:
                entities["cve_ids"] = _cves
                logger.info(f"[MULTI-ENTITY] detected {len(_cves)} cve_ids: {_cves}")

            # ip: all matches
            _ip_re = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
            _ips = []
            for m in _ip_re.finditer(_multi_text):
                v = m.group(0)
                if v not in _ips:
                    _ips.append(v)
            if len(_ips) >= 2:
                entities["ips"] = _ips
                logger.info(f"[MULTI-ENTITY] detected {len(_ips)} ips: {_ips}")
        except Exception as _me_exc:
            logger.warning(f"[MULTI-ENTITY] extraction failed: {_me_exc}")

        # NLP absolute time-window parser. Catches phrasings like:
        #   "from 9 AM to 5 PM", "9am to 5pm", "between 14:00 and 18:00",
        #   "9 to 17 today", "9 AM yesterday to 5 PM today".
        # On success sets entities["time_range"] to an ISO range:
        #   "<from_iso>..<to_iso>"  (UTC, second-precision)
        # which downstream indexers translate into gte/lte clauses.
        try:
            import datetime as _dt
            _q = (query or "").lower()
            _now = _dt.datetime.utcnow().replace(microsecond=0)
            _today = _now.date()
            _yest = _today - _dt.timedelta(days=1)
            _tomo = _today + _dt.timedelta(days=1)

            def _to_24h(_h, _ampm):
                """Convert ('9', 'am'|'pm'|None) -> 0..23 int. Returns None on parse fail."""
                try:
                    _hi = int(_h)
                except Exception:
                    return None
                if _hi < 0 or _hi > 23:
                    return None
                if _ampm:
                    _ampm = _ampm.lower()
                    if _ampm.startswith("p") and _hi < 12:
                        _hi += 12
                    elif _ampm.startswith("a") and _hi == 12:
                        _hi = 0
                return _hi

            # Pattern: "from 9 am to 5 pm", "9am - 5pm", "between 9 am and 5 pm",
            # "9 am yesterday to 5 pm today" (per-side day markers).
            _abs_re = re.compile(
                r"(?:from\s+|between\s+)?"
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(yesterday|today|tomorrow)?"
                r"\s*(?:to|-|until|and|\u2013|\u2014)\s*"
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(yesterday|today|tomorrow)?",
                re.IGNORECASE,
            )
            _m_abs = _abs_re.search(_q)

            def _day_of(_label):
                if not _label:
                    return None
                _l = _label.lower()
                if _l == "yesterday": return _yest
                if _l == "today":     return _today
                if _l == "tomorrow":  return _tomo
                return None

            # Default: window is on today. If query has explicit per-side
            # day markers, use those. Else fall back to global "yesterday"
            # / "tomorrow" mentions for the from / to side respectively.
            _from_day = _today
            _to_day = _today
            if _m_abs:
                _from_explicit = _day_of(_m_abs.group(4))
                _to_explicit   = _day_of(_m_abs.group(8))
                if _from_explicit:
                    _from_day = _from_explicit
                if _to_explicit:
                    _to_day = _to_explicit
                if not _from_explicit and not _to_explicit:
                    if re.search(r"\byesterday\b", _q):
                        _from_day = _yest
                    if re.search(r"\btomorrow\b", _q):
                        _to_day = _tomo
                    if "yesterday" in _q and "today" in _q:
                        _from_day, _to_day = _yest, _today

            if _m_abs:
                _h1 = _to_24h(_m_abs.group(1), _m_abs.group(3))
                _m1 = int(_m_abs.group(2) or 0)
                _h2 = _to_24h(_m_abs.group(5), _m_abs.group(7))
                _m2 = int(_m_abs.group(6) or 0)
                if _h1 is not None and _h2 is not None:
                    _from_dt = _dt.datetime.combine(_from_day, _dt.time(_h1, _m1))
                    _to_dt   = _dt.datetime.combine(_to_day,   _dt.time(_h2, _m2))
                    # If end is earlier than start on the same day, assume end is the next day
                    if _to_dt <= _from_dt:
                        _to_dt += _dt.timedelta(days=1)
                    entities["time_range"] = f"{_from_dt.isoformat()}Z..{_to_dt.isoformat()}Z"
                    logger.info(f"[ENTITY] absolute time-window parsed: {entities['time_range']}")
        except Exception as _abs_e:
            logger.debug(f"[ENTITY] absolute time-window parser failed: {_abs_e}")

        return entities

    def _build_arguments(
        self,
        tool_name: str,
        entities: Dict[str, Any],
        base_args: Dict[str, Any] = None,
        user_query: str = ""
    ) -> Dict[str, Any]:
        """Build tool arguments from entities"""

        args = base_args.copy() if base_args else {}

        # Get tool metadata
        tool_meta = ToolRegistry.get_tool(tool_name)
        if not tool_meta:
            logger.warning(f"Tool not found in registry: {tool_name}")
            return args

        schema = tool_meta.input_schema

        # Map entities to arguments based on schema
        if "severity" in entities and "severity" in schema:
            args["severity"] = entities["severity"]

        # 2026-05-15: For Wazuh alert tools, translate severity entity into
        # min_level/max_level rule-level filters matching the Wazuh dashboard:
        #   Critical: rule.level >= 15
        #   High:     rule.level 12-14
        #   Medium:   rule.level 7-11
        #   Low:      rule.level 0-6
        # Without this, "share endpoint critical alerts" would fetch ALL alerts
        # and the user has to read the breakdown to find the critical count.
        _ALERT_TOOLS_SEV = ("get_wazuh_alerts", "get_wazuh_alert_summary", "analyze_alert_patterns")
        if tool_name in _ALERT_TOOLS_SEV and entities.get("severity"):
            _sev = str(entities["severity"]).lower()
            _sev_to_level = {
                "critical": {"min_level": 15},
                "high":     {"min_level": 12, "max_level": 14},
                "medium":   {"min_level": 7,  "max_level": 11},
                "low":      {"min_level": 0,  "max_level": 6},
            }
            _lvl_args = _sev_to_level.get(_sev)
            if _lvl_args:
                # Don't clobber explicit user-provided level args.
                for _k, _v in _lvl_args.items():
                    args.setdefault(_k, _v)
                logger.info(
                    f"[ARGS] Translated severity={_sev!r} -> {_lvl_args} for {tool_name}"
                )

        if "agent_id" in entities and "agent_id" in schema:
            args["agent_id"] = entities["agent_id"]

        if "status" in entities and "status" in schema:
            args["status"] = entities["status"]

        if "time_range" in entities and "time_range" in schema:
            args["time_range"] = entities["time_range"]

        if "framework" in entities and "framework" in schema:
            args["framework"] = entities["framework"]

        if "level" in entities and "level" in schema:
            args["level"] = entities["level"]

        if "cve_id" in entities and "cve_id" in schema:
            args["cve_id"] = entities["cve_id"]

        # OS filter — forwards entities["os"] (linux/windows/darwin) to any tool
        # whose schema declares an `os` parameter. Without this mapping, queries
        # like "show windows vulnerabilities" never apply the OS filter and
        # return all-platform results.
        if "os" in entities and "os" in schema:
            args["os"] = entities["os"]

        # MITRE filters (technique name + tactic) — same flow: only forward when
        # the target tool actually accepts them in its schema.
        if "mitre_technique" in entities and "mitre_technique" in schema:
            args["mitre_technique"] = entities["mitre_technique"]
        if "mitre_tactic" in entities and "mitre_tactic" in schema:
            args["mitre_tactic"] = entities["mitre_tactic"]
        if "mitre_id" in entities and "mitre_id" in schema:
            args["mitre_id"] = entities["mitre_id"]
        if "rule_id" in entities and "rule_id" in schema:
            args["rule_id"] = entities["rule_id"]

        # ============================================================
        # M6a (2026-06-10) PALLAS_P4_SURICATA_SEARCH_EXTRACT
        # search_suricata_alerts uses an OpenSearch multi_match phrase_prefix
        # on alert.signature / alert.category / src_ip / dest_ip. If the
        # router passes the WHOLE user query as the search term, the prefix
        # never matches anything ("search NIDS alerts for X" is not a
        # signature prefix). Strip common natural-language prefixes here
        # and keep just the search term.
        # ============================================================
        if (
            tool_name == "search_suricata_alerts"
            and "query" in schema
            and user_query
        ):
            _qtext = user_query.strip().strip("'\"")
            _strip_patterns = [
                # "search/find/show/filter NIDS/suricata alerts for|with|matching|by|where|in [signature|category|sig|cat] X"
                r"^(?:search|find|show|filter)\s+(?:nids|suricata)\s+alerts?\s+(?:for|with|matching|by|where|in|on)\s+(?:signature[s]?\s+|category|categor(?:y|ies)\s+|sig\s+|cat\s+|name\s+)?",
                # "search/find/show/filter NIDS/suricata by|for|in signature|category X"
                r"^(?:search|find|show|filter)\s+(?:nids|suricata)\s+(?:by|for|in|on)\s+(?:signature[s]?|categor(?:y|ies)|sig|cat|name)\s+",
                # "show alerts in category X" (no NIDS prefix)
                r"^(?:show|find|filter)\s+alerts?\s+(?:in|by|with|for)\s+(?:signature[s]?|categor(?:y|ies))\s+",
                # "search/find for NIDS/suricata X" (generic fallback)
                r"^(?:search|find)\s+(?:for\s+)?(?:nids|suricata)\s+(?:alerts?\s+)?",
            ]
            for _pat in _strip_patterns:
                _m = re.match(_pat, _qtext, re.IGNORECASE)
                if _m:
                    _qtext = _qtext[_m.end():].strip().strip("'\"")
                    break
            # Strip trailing time-range phrases the extractor may have left in
            _qtext = re.sub(
                r"\s+(?:in\s+the\s+)?(?:last|past)\s+\d+\s*(?:h|hr|hrs|hour|hours|d|day|days|w|week|weeks)\s*$",
                "",
                _qtext,
                flags=re.IGNORECASE,
            ).strip()
            if _qtext and _qtext.lower() not in (user_query.lower().strip(), "nids", "suricata", "alerts"):
                args["query"] = _qtext
                logger.info(
                    f"[ARGS] M6a search_suricata_alerts: extracted query={_qtext!r} "
                    f"(from user_query={user_query[:80]!r})"
                )

        # Special: extract raw_log for decoder generator
        if tool_name == "generate_wazuh_decoder" and "raw_log" in schema and "raw_log" not in args:
            # Strategy 1: Everything after common separator keywords
            for pat in [
                r'(?:for|log|from|decode|analyze)\s*:\s*(.+)',
                r':\s*(.{20,})',
            ]:
                m = re.search(pat, user_query, re.IGNORECASE | re.DOTALL)
                if m:
                    args["raw_log"] = m.group(1).strip()
                    break

            # Strategy 2: Everything after "decoder/decode" keyword phrase
            if "raw_log" not in args:
                m = re.search(
                    r'\b(?:decoder|decode)\b.*?(?:for|from|:)\s*(.{20,})',
                    user_query, re.IGNORECASE | re.DOTALL
                )
                if m:
                    args["raw_log"] = m.group(1).strip()

            # Strategy 3: Extract everything after a recognizable syslog timestamp
            if "raw_log" not in args and len(user_query) > 40:
                m = re.search(
                    r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+.+)',
                    user_query, re.IGNORECASE | re.DOTALL
                )
                if m:
                    args["raw_log"] = m.group(1).strip()

        # Add defaults from schema
        for field, spec in schema.items():
            if field not in args and "default" in spec:
                args[field] = spec["default"]

        # FIX: Cap limits to Wazuh API hard maximums to prevent 400 errors.
        # The ToolRegistry schema defaults may be set high (e.g. 10000) but
        # the Wazuh REST API enforces strict maximums per endpoint.
        API_LIMIT_CAPS = {
            "get_wazuh_manager_error_logs": 500,
            "search_wazuh_manager_logs":    500,
            # 2026-05-15 audit: get_wazuh_alerts is also indexer-backed
            # (wazuh_client.get_alerts delegates to indexer_client.search_alerts).
            # Same 1000->10000 cap fix as the vuln tools below.
            "get_wazuh_alerts":             10000,
            # Phase 6.5: indexer-backed vuln tools were under-reporting because
            # the cap truncated to 1000 even though OpenSearch max_result_window
            # defaults to 10000. Raising to 10000 covers most environments
            # (current test box has ~7,900 vulns total). Long-term fix is real
            # pagination in api/wazuh_indexer.py.
            "get_wazuh_vulnerabilities":    10000,
            "get_wazuh_critical_vulnerabilities": 10000,
            "get_wazuh_agents":             500,
            "get_wazuh_running_agents":     500,
        }
        if tool_name in API_LIMIT_CAPS and "limit" in args:
            cap = API_LIMIT_CAPS[tool_name]
            if args["limit"] > cap:
                logger.debug(f"[LIMIT_CAP] {tool_name}: capping limit {args['limit']} → {cap}")
                args["limit"] = cap

        # Pallas 3.2: IP entity injection for IP investigation pivot
        if "ip_address" in entities:
            ip = entities["ip_address"]
            if tool_name == "search_security_events" and "query" not in args:
                args["query"] = ip
            elif tool_name == "get_suricata_alerts" and "src_ip" not in args:
                args["src_ip"] = ip

        return args

    async def _llm_tool_selection(self, query: str, entities: Dict[str, Any]) -> ToolPlan:
        """
        Use LLM for tool selection (complex queries only).
        """

        # Build tool list for LLM (limit to prevent token overflow)
        available_tools = ToolRegistry.get_all_tools()

        # Prioritize tools by category for better context
        tool_descriptions = []

        # Add vulnerability tools first (most common)
        vuln_tools = [t for t in available_tools if t.category == ToolCategory.VULNERABILITY_MANAGEMENT]
        for tool in vuln_tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        # Add agent tools
        agent_tools = [t for t in available_tools if t.category == ToolCategory.AGENT_MANAGEMENT]
        for tool in agent_tools[:3]:  # Limit to top 3
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        # Add alert tools
        alert_tools = [t for t in available_tools if t.category == ToolCategory.ALERT_MANAGEMENT]
        for tool in alert_tools[:2]:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        # Add analysis tools
        analysis_tools = [t for t in available_tools if t.category == ToolCategory.SECURITY_ANALYSIS]
        for tool in analysis_tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        tools_text = "\n".join(tool_descriptions[:20])  # Max 20 tools to prevent overflow

        # Structured prompt for tool selection
        system_prompt = """You are a security operations expert. Select 1-3 tools to answer the query.

RULES:
1. Primary tool provides the main data. Secondary tools enrich it.
2. For cross-domain questions (e.g., "agents with vulnerabilities"), select BOTH tools.
3. For deep dives (e.g., "full posture of agent"), select agent + vuln + ports + processes.
4. For correlation queries (e.g., "alerts AND vulnerabilities"), select all relevant tools.
5. ALWAYS set correlation_strategy when selecting 2+ tools.
6. ONLY select tools from the available list.
7. For "search virustotal" or "search [keyword] event" queries --> use search_security_events

CORRELATION STRATEGIES (use exact names):
- vulnerability_with_agents: vulns enriched with agent details
- agents_with_vulnerabilities: agents ranked by vuln count
- alerts_with_agents: alerts enriched with agent context
- alerts_with_vulnerabilities: agents appearing in BOTH alerts + vulns
- agent_posture_deep_dive: full agent view with vulns + ports + processes
- fim_with_agent_posture: FIM events with agent + process context
- disconnected_agents_with_critical_vulns: disconnected agents with critical CVEs
- active_agents_with_high_vulns: active agents with high/critical vulns
- compare_vulns_active_vs_disconnected: vuln distribution by agent status

If no strategy fits, set correlation_strategy to null."""

        prompt = f"""FILTER VOCABULARY (use canonical values for filter args):
- status: "active" | "disconnected" | "never_connected"
  Map synonyms: inactive/offline/down/dead/unreachable -> disconnected;
  online/connected/up/alive/healthy/running -> active;
  pending/unenrolled/never connected -> never_connected.
- severity: "critical" | "high" | "medium" | "low".
- time_range format: "{{N}}{{h|d|w|m}}" (e.g., "24h", "7d", "2w").

If the user asks for BOTH X AND Y (e.g., "active AND inactive agents",
"alerts AND vulnerabilities"), include BOTH as primary + secondary tools
so the response covers both buckets.

AVAILABLE TOOLS:
{tools_text}

EXAMPLES (follow this exact JSON format):

Query: "show critical vulnerabilities for agent 019"
JSON: {{"primary_tool": "get_wazuh_vulnerabilities", "secondary_tools": ["get_wazuh_agents"], "correlation_strategy": "vulnerability_with_agents", "reasoning": "agent-specific critical vuln filter", "confidence": 0.9}}

Query: "any network attacks in the last hour"
JSON: {{"primary_tool": "get_suricata_alerts", "secondary_tools": [], "correlation_strategy": null, "reasoning": "recent IDS alerts", "confidence": 0.85}}

Query: "show me alerts and vulnerabilities for agent 011"
JSON: {{"primary_tool": "get_wazuh_alerts", "secondary_tools": ["get_wazuh_vulnerabilities", "get_wazuh_agents"], "correlation_strategy": "alerts_with_vulnerabilities", "reasoning": "multi-intent: alerts and vulns for same agent", "confidence": 0.88}}

Query: "which agent has the most critical CVEs"
JSON: {{"primary_tool": "get_wazuh_agents", "secondary_tools": ["get_wazuh_vulnerabilities"], "correlation_strategy": "top_agents_by_vuln_count", "reasoning": "agent ranking by vuln count", "confidence": 0.88}}

Query: "show HTTP traffic analysis"
JSON: {{"primary_tool": "get_suricata_http_analysis", "secondary_tools": [], "correlation_strategy": null, "reasoning": "Suricata HTTP analysis", "confidence": 0.9}}

Query: "disconnected agents with critical vulns"
JSON: {{"primary_tool": "get_wazuh_agents", "secondary_tools": ["get_wazuh_vulnerabilities"], "correlation_strategy": "disconnected_agents_with_critical_vulns", "reasoning": "disconnected agents + critical CVE correlation", "confidence": 0.92}}

Query: "check health of agent 067"
JSON: {{"primary_tool": "check_agent_health", "secondary_tools": [], "correlation_strategy": null, "reasoning": "agent health check", "confidence": 0.95}}

EXTRACTED ENTITIES: {json.dumps(entities)}

USER QUERY: "{query}"

Select tools. Follow the JSON format above exactly. Use EXACT tool names from AVAILABLE TOOLS. If no correlation strategy fits, set it to null.

JSON:"""

        try:
            # LOW sensitivity — tool selection only sees query text + tool names, no client data
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.1,  # Very low temp for consistency
                call_type="tool_selection",  # Routes to CLOUD model
            )

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))

                llm_correlation = plan_data.get("correlation_strategy")
                secondary = plan_data.get("secondary_tools", [])
                return ToolPlan(
                    primary_tool=plan_data.get("primary_tool", "get_wazuh_agents"),
                    secondary_tools=secondary,
                    primary_args={},  # Will be filled by _build_arguments
                    secondary_args=[],
                    confidence=float(plan_data.get("confidence", 0.6)),
                    selection_method="llm",
                    reasoning=plan_data.get("reasoning", "LLM selection"),
                    requires_correlation=bool(secondary and llm_correlation),
                    correlation_strategy=llm_correlation if llm_correlation and llm_correlation != "null" else None
                )
            else:
                raise ValueError("No JSON found in LLM response")

        except Exception as e:
            logger.error(f"LLM tool selection parsing failed: {e}")
            logger.error(f"LLM response was: {response if 'response' in locals() else 'N/A'}")
            raise

    def _validate_and_enrich_plan(
        self,
        plan: ToolPlan,
        query: str,
        entities: Dict[str, Any]
    ) -> ToolPlan:
        """
        Validate LLM-generated plan and enrich with arguments.
        """

        # Check if primary tool exists
        primary_tool_meta = ToolRegistry.get_tool(plan.primary_tool)
        if not primary_tool_meta:
            logger.warning(f"LLM selected invalid primary tool: {plan.primary_tool}")
            # Override with safe default
            plan.primary_tool = "get_wazuh_agents"
            plan.confidence *= 0.5
            plan.reasoning += " [CORRECTED: invalid primary tool]"

        # Check secondary tools
        valid_secondary = []
        for tool in plan.secondary_tools:
            if ToolRegistry.get_tool(tool):
                valid_secondary.append(tool)
            else:
                logger.warning(f"LLM selected invalid secondary tool: {tool}, removing")

        plan.secondary_tools = valid_secondary

        # Build arguments from entities
        plan.primary_args = self._build_arguments(plan.primary_tool, entities, user_query=query)
        plan.secondary_args = [
            self._build_arguments(tool, entities, user_query=query)
            for tool in plan.secondary_tools
        ]

        # Determine correlation strategy — respect LLM's choice if valid
        if plan.secondary_tools:
            valid_strategies = set(CorrelationEngine.STRATEGIES.keys()) if hasattr(CorrelationEngine, 'STRATEGIES') else set()
            llm_strategy_valid = (
                plan.correlation_strategy is not None
                and plan.correlation_strategy in valid_strategies
            )

            if llm_strategy_valid:
                # LLM provided a known strategy — keep it
                plan.requires_correlation = True
                logger.info(f"[VALIDATE] Keeping LLM correlation_strategy: {plan.correlation_strategy}")
            else:
                # LLM didn't provide one or provided invalid — try inference
                correlation = self._infer_correlation_strategy(
                    plan.primary_tool,
                    plan.secondary_tools
                )
                if correlation:
                    plan.requires_correlation = True
                    plan.correlation_strategy = correlation
                    logger.info(f"[VALIDATE] Inferred correlation_strategy: {correlation}")
                elif plan.correlation_strategy:
                    logger.warning(f"[VALIDATE] Unknown strategy '{plan.correlation_strategy}', clearing")
                    plan.correlation_strategy = None
                    plan.requires_correlation = False

        # Boost confidence if LLM matches known patterns
        if self._llm_matches_known_pattern(query, plan.primary_tool):
            plan.confidence = min(0.9, plan.confidence + 0.15)
            plan.reasoning += " [VALIDATED: matches known pattern]"

        return plan

    def _infer_correlation_strategy(
        self,
        primary_tool: str,
        secondary_tools: List[str]
    ) -> Optional[str]:
        """
        Infer appropriate correlation strategy from tool combination.
        Rules are ordered most-specific-first to prevent generic rules from swallowing
        specific combinations (e.g., agent deep-dive vs simple agent+vuln).
        """
        p = primary_tool.lower()
        s_lower = [t.lower() for t in secondary_tools]

        # Helper: match root "vulnerabilit" to cover both "vulnerability" and "vulnerabilities"
        def has_vuln(tool_name: str) -> bool:
            return "vulnerabilit" in tool_name

        # 24. agents primary + vulns + alerts + ports = dynamic risk scoring (before #1: more specific)
        if "agent" in p and any(has_vuln(t) for t in s_lower) and any("alert" in t and "wazuh" in t for t in s_lower):
            if any("port" in t for t in s_lower):
                return "dynamic_risk_scoring"

        # 1. Agent deep-dive: agent + 2+ tools including vuln + port/process
        if "agent" in p and len(secondary_tools) >= 2:
            s_has_vuln = any(has_vuln(t) for t in s_lower)
            s_has_port_or_proc = any("port" in t or "process" in t for t in s_lower)
            if s_has_vuln and s_has_port_or_proc:
                return "agent_posture_deep_dive"

        # 2. Alert + Vulnerability cross-correlation (agents with BOTH alerts AND vulns)
        if "alert" in p and any(has_vuln(t) for t in s_lower):
            return "alerts_with_vulnerabilities"

        # 3. FIM + Agent correlation
        if "fim" in p and any("agent" in t for t in s_lower):
            return "fim_with_agent_posture"

        # 4. Rule/MITRE + Agent correlation
        if ("rule" in p or "mitre" in p) and any("agent" in t for t in s_lower):
            return "rule_mitre_with_agents"

        # 5. Vulnerability + Agent correlation
        if has_vuln(p) and any("agent" in t for t in s_lower):
            return "vulnerability_with_agents"

        # 6. Agent + Vulnerability correlation
        if "agent" in p and any(has_vuln(t) for t in s_lower):
            return "agents_with_vulnerabilities"

        # 25. suricata primary + wazuh alerts + agents = cross-platform threat (before #7: more specific)
        if "suricata" in p and "alert" in p and any("wazuh" in t and "alert" in t for t in s_lower) and any("agent" in t for t in s_lower):
            return "cross_platform_threat_correlation"

        # 29. wazuh alerts primary + suricata + agents = campaign detection (before #7: more specific)
        if "wazuh" in p and "alert" in p and any("suricata" in t for t in s_lower) and any("agent" in t for t in s_lower):
            return "multi_asset_campaign_detection"

        # 7. Alert + Agent correlation
        if "alert" in p and any("agent" in t for t in s_lower):
            return "alerts_with_agents"

        # --- Pallas 3.2: Extended inference rules ---

        # 8. Comprehensive security posture: agent + 3+ tools including BOTH vuln+alert+suricata
        #    (MUST be checked BEFORE rule 9 because both match agent + 3+ tools)
        if "agent" in p and len(secondary_tools) >= 3:
            s_has_vuln = any(has_vuln(t) for t in s_lower)
            s_has_alert = any("alert" in t and "suricata" not in t for t in s_lower)
            s_has_suricata = any("suricata" in t for t in s_lower)
            if s_has_vuln and s_has_alert and s_has_suricata:
                return "comprehensive_security_posture"

        # 9. Agent + 3+ tools = full investigation (generic multi-tool agent investigation)
        if "agent" in p and len(secondary_tools) >= 3:
            return "full_agent_investigation"

        # 10. Suricata alerts + HTTP analysis
        if "suricata" in p and "alert" in p and any("http" in t for t in s_lower):
            return "suricata_http_alert_context"

        # 11. Suricata TLS + alerts
        if "tls" in p and any("alert" in t for t in s_lower):
            return "suricata_tls_threat_detection"

        # 12. Suricata MITRE + Wazuh alerts
        if "mitre" in p and any("wazuh" in t or ("alert" in t and "suricata" not in t) for t in s_lower):
            return "wazuh_suricata_mitre_unified"

        # 13. Suricata suspicious + attackers
        if "suspicious" in p and any("attacker" in t for t in s_lower):
            return "unified_scanning_detection"

        # 14. FIM + alerts (without agent context already handled by rule 3)
        if "fim" in p and any("alert" in t for t in s_lower):
            return "fim_alert_correlation"

        # 15. Ports + Suricata alerts
        if "port" in p and any("suricata" in t for t in s_lower):
            return "port_exposure_risk"

        # --- Pallas 3.3: Additional inference rules ---

        # 16. Combined alert view: Suricata alerts + Wazuh alerts (either direction)
        if "suricata" in p and "alert" in p and any("wazuh" in t and "alert" in t for t in s_lower):
            return "combined_alert_view"
        if "wazuh" in p and "alert" in p and any("suricata" in t and "alert" in t for t in s_lower):
            return "combined_alert_view"

        # 17. Wazuh agents + Suricata detections (without vulnerabilities)
        if "agent" in p and any("suricata" in t for t in s_lower) and not any(has_vuln(t) for t in s_lower):
            return "wazuh_agents_suricata_detections"

        # 18. Suricata alerts + network analysis = attacker-target map
        if "suricata" in p and "alert" in p and any("network" in t for t in s_lower):
            return "suricata_attacker_target_map"

        # 19. Suricata summary + critical alerts = signature severity analysis
        if "suricata" in p and "summary" in p and any("critical" in t for t in s_lower):
            return "suricata_signature_severity_analysis"

        # 20. Suricata network + summary = network threat profile
        if "network" in p and "suricata" in p and any("summary" in t for t in s_lower):
            return "suricata_network_threat_profile"

        # 21. IoC + Suricata context
        if "ioc" in p and any("suricata" in t for t in s_lower):
            return "ioc_with_suricata_context"

        # 22. Alert + alert (cross-platform) when both contain alerts
        if "alert" in p and any("alert" in t for t in s_lower) and p != s_lower[0] if s_lower else False:
            return "combined_alert_view"

        # --- Pallas 3.5: Advanced SOC inference rules ---

        # 23. search_security_events + agents + suricata = temporal attack sequence
        if "search" in p and "event" in p and any("agent" in t for t in s_lower) and any("suricata" in t for t in s_lower):
            return "temporal_attack_sequence"

        # 24. (moved above rule #1 — more specific: requires vuln + alert + port)

        # 25. (moved above rule #7 — more specific: requires suricata+wazuh+agent)

        # 26. FIM primary + processes + suricata = data exfiltration
        if "fim" in p and any("process" in t for t in s_lower):
            return "data_exfiltration_detection"

        # 27. search events + agents (no suricata) + off-hours context = off hours
        if "search" in p and "event" in p and any("agent" in t for t in s_lower) and not any("suricata" in t for t in s_lower):
            # This is a weaker rule - only matches if explicitly routed here by pattern
            pass

        # 28. rule_trigger primary + timeline + agents = alert noise
        if "rule" in p and "trigger" in p and any("timeline" in t for t in s_lower):
            return "alert_noise_analysis"

        # 29. (moved above rule #7 — more specific: requires wazuh+suricata+agent)

        return None

    def _llm_matches_known_pattern(self, query: str, selected_tool: str) -> bool:
        """Check if LLM selection matches what rules would have chosen"""

        query_lower = query.lower()

        # Check keyword patterns
        for config in self.KEYWORD_PATTERNS.values():
            if config["tool"] == selected_tool:
                if any(kw in query_lower for kw in config.get("keywords", [])):
                    return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get tool selection statistics"""

        total = self.stats["total"]
        if total == 0:
            return {
                **self.stats,
                "recent_llm_queries": list(self.recent_llm_queries),
                "recent_fallback_queries": list(self.recent_fallback_queries),
            }

        return {
            **self.stats,
            "percentages": {
                "rule_regex": (self.stats["rule_regex"] / total) * 100,
                "rule_keyword": (self.stats["rule_keyword"] / total) * 100,
                "llm": (self.stats["llm"] / total) * 100,
                "fallback": (self.stats["fallback"] / total) * 100
            },
            "recent_llm_queries": list(self.recent_llm_queries),
            "recent_fallback_queries": list(self.recent_fallback_queries),
        }


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class ToolExecutor:
    """Executes MCP tools with validation and error handling"""

    def __init__(self, mcp_tool_executor):
        """
        Args:
            mcp_tool_executor: Async function that executes MCP tools
                               Signature: async def(tool_name: str, args: dict) -> dict
        """
        self.mcp_tool_executor = mcp_tool_executor

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute a single tool with validation"""

        # Validate tool exists
        tool_meta = ToolRegistry.get_tool(tool_name)
        if not tool_meta:
            return ExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
                execution_time=0.0
            )

        # E1.2 (2026-06-11): the ~190-line backfill + coercion + Title-Case
        # block was lifted into orchestrator/execution/arg_guards.py.
        # Same env flags (PALLAS_P3_SEARCH_QUERY_BACKFILL,
        # PALLAS_P4_CORR_ARG_COERCE, PALLAS_P4_SCHEMA_COERCE), same logic,
        # same warn-log format. Mutates arguments in place.
        from .orchestrator.execution.arg_guards import coerce_tool_args
        coerce_tool_args(tool_name, arguments)

        # Validate arguments
        valid, error_msg = ToolRegistry.validate_tool_arguments(tool_name, arguments)
        if not valid:
            # E1.2 (2026-06-11): PALLAS_P4_VALIDATION_RETRY moved to
            # orchestrator/execution/arg_guards.retry_coerce_from_error.
            # Same single-retry cap, same env flag, same warn-log.
            from .orchestrator.execution.arg_guards import retry_coerce_from_error
            _changed, _ = retry_coerce_from_error(tool_name, arguments, error_msg)
            if _changed:
                valid, error_msg = ToolRegistry.validate_tool_arguments(tool_name, arguments)
            if not valid:
                return ExecutionResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    success=False,
                    data=None,
                    error=f"Invalid arguments: {error_msg}",
                    execution_time=0.0
                )

        # Execute
        start_time = time.time()

        try:
            logger.info(f"[EXEC] Tool: {tool_name} | Args: {json.dumps(arguments, default=str)}")
            result = await self.mcp_tool_executor(tool_name, arguments)
            execution_time = time.time() - start_time

            logger.info(f"[EXEC] ✅ Tool {tool_name} completed in {execution_time:.2f}s")

            return ExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                data=result,
                error=None,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[EXEC] ❌ Tool {tool_name} failed after {execution_time:.2f}s: {e}")

            return ExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )

    async def execute_batch(
        self,
        tool_specs: List[Tuple[str, Dict[str, Any]]]
    ) -> List[ExecutionResult]:
        """Execute multiple tools in parallel"""

        tasks = [self.execute(tool_name, args) for tool_name, args in tool_specs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results


# ============================================================================
# QUERY ORCHESTRATOR - Main Entry Point
# ============================================================================

class QueryOrchestrator:
    """
    Main orchestrator coordinating:
    - Hybrid tool selection (rules + LLM)
    - Multi-tool execution
    - Data correlation
    - LLM analysis
    - Response formatting

    This REPLACES chat_handler.py
    """

    def __init__(self, mcp_tool_executor, ollama_client: LLMClient, suricata_enabled: bool = False):
        """
        Args:
            mcp_tool_executor: Async function to execute MCP tools
            ollama_client: LLMClient instance for analysis (NOT tool selection)
            suricata_enabled: Whether Suricata IDS integration is active
        """
        self.suricata_enabled = suricata_enabled
        self.selector = HybridToolSelector(ollama_client, suricata_enabled=suricata_enabled)
        self.executor = ToolExecutor(mcp_tool_executor)
        self.correlator = CorrelationEngine()
        self.formatter = ResponseFormatter()
        self.llm = ollama_client

        # Wire executor into the selector so it can resolve agent name → ID
        self.selector.set_tool_executor(self.executor)

        # Pallas 4.1: Response cache for expensive multi-tool queries
        # Key = normalized query string, Value = {"response": str, "turn_data": dict, "timestamp": float}
        #
        # 2026-05-22: gated behind ENABLE_RESPONSE_CACHE env var (default OFF)
        # because identical queries kept returning identical errors during QA's
        # backend regression — analysts thought Pallas "wasn't trying" when it
        # was just replaying cached failures. Set ENABLE_RESPONSE_CACHE=true to
        # re-enable once the data-layer regressions are fixed.
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "180"))
        self._cache_max_size = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "30"))
        self._cache_enabled = os.getenv("ENABLE_RESPONSE_CACHE", "false").lower() in ("1", "true", "yes", "on")
        if not self._cache_enabled:
            logger.info("Response cache DISABLED (set ENABLE_RESPONSE_CACHE=true to enable)")

        logger.info(f"QueryOrchestrator initialized (suricata={'ON' if suricata_enabled else 'OFF'})")


    def _detect_query_intent(self, tool_name: str, user_query: str = "") -> str:
        """Detect query intent from tool name and user query"""
        query_lower = user_query.lower()

        if 'running_agents' in tool_name or 'active' in query_lower:
            return 'active_agents'
        elif 'agent' in tool_name and 'status' in query_lower:
            return 'agent_status'
        elif 'critical_vulnerabilities' in tool_name:
            return 'critical_vulnerabilities'
        elif 'vulnerability' in tool_name:
            return 'vulnerabilities'
        elif 'alert' in tool_name:
            return 'recent_alerts'
        elif 'agent' in tool_name:
            return 'agents'
        else:
            return 'generic'

    def _normalize_tool_payload(self, payload: Any) -> Dict[str, Any]:
        """
        Normalize raw MCP tool responses to a consistent structure.
        Handles:
        - {"content":[{"text":"{json}"}]}  -- MCP wrapper format
        - {"data": {...}}                  -- Wazuh API direct format
        - {"affected_items":[...]}         -- Data-like already unwrapped
        Returns: {"affected_items": [...], "_raw": original_payload}
        """
        if payload is None:
            return {"affected_items": [], "_source": "none"}

        try:
            if isinstance(payload, dict):
                # Check for raw data alongside MCP content (agent tools include both)
                if "data" in payload and isinstance(payload["data"], dict) and "content" in payload:
                    raw_data = payload["data"]
                    if "affected_items" in raw_data and isinstance(raw_data["affected_items"], list):
                        return {"affected_items": raw_data["affected_items"], "_source": "raw_data_with_content"}

                # MCP wrapper: {"content":[{"text":"...json..."}]}
                if "content" in payload and isinstance(payload.get("content"), list) and payload["content"]:
                    text = payload["content"][0].get("text", "")
                    if isinstance(text, str) and text.strip():
                        # Strip common label prefixes before JSON parsing
                        # (server.py handlers prepend "Alert Summary:\n", "Security Events:\n", etc.)
                        stripped = text.strip()
                        for prefix in ("Alert Summary:", "Alert Patterns:", "Security Events:",
                                        "Threat Analysis:", "IoC Reputation:", "Rules Summary:",
                                        "Compliance Check:", "Vulnerability Summary:"):
                            if stripped.startswith(prefix):
                                stripped = stripped[len(prefix):].lstrip()
                                break
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                if "data" in parsed and isinstance(parsed["data"], dict):
                                    items = parsed["data"].get("affected_items", [])
                                    return {"affected_items": items if isinstance(items, list) else [], "_source": "content_data"}
                                if "affected_items" in parsed:
                                    items = parsed["affected_items"]
                                    return {"affected_items": items if isinstance(items, list) else [], "_source": "content_items"}
                                # Treat entire parsed as data
                                return {"affected_items": [], "_parsed": parsed, "_source": "content_parsed"}
                        except json.JSONDecodeError:
                            return {"affected_items": [], "_raw_text": text[:200], "_source": "content_parse_error"}

                # Wazuh-style: {"data": {"affected_items": [...]}}
                if "data" in payload and isinstance(payload["data"], dict):
                    items = payload["data"].get("affected_items", [])
                    return {"affected_items": items if isinstance(items, list) else [], "_source": "data_dict"}

                # Already data-like
                if "affected_items" in payload:
                    items = payload["affected_items"]
                    return {"affected_items": items if isinstance(items, list) else [], "_source": "direct_items"}

            return {"affected_items": [], "_source": "unknown", "_type": str(type(payload))}

        except Exception as e:
            logger.error(f"[NORMALIZE] Payload normalization error: {e}")
            return {"affected_items": [], "_source": "error", "_error": str(e)}

    def _extract_count(self, result):
        """Best-effort count of rows/events a tool result represents.
        Prefers the true indexer total (total_affected_items / hits.total.value),
        falls back to len(affected_items). Used for live progress telemetry."""
        payload = getattr(result, "data", result)

        def _peek_total(p):
            try:
                if isinstance(p, dict):
                    for keys in (("data", "total_affected_items"),
                                 ("total_affected_items",),
                                 ("hits", "total", "value")):
                        cur, ok = p, True
                        for k in keys:
                            if isinstance(cur, dict) and k in cur:
                                cur = cur[k]
                            else:
                                ok = False
                                break
                        if ok and isinstance(cur, int) and cur >= 0:
                            return cur
                    if isinstance(p.get("content"), list) and p["content"]:
                        txt = p["content"][0].get("text", "")
                        if isinstance(txt, str) and txt.strip().startswith("{"):
                            return _peek_total(json.loads(txt))
            except Exception:
                return None
            return None

        total = _peek_total(payload)
        if isinstance(total, int):
            return total
        try:
            return len(self._normalize_tool_payload(payload).get("affected_items", []))
        except Exception:
            return 0

    def _noun_for_tool(self, tool_name):
        """The unit a query's result represents, for honest live telemetry."""
        t = (tool_name or "").lower()
        if "agent" in t:
            return "agents"
        if "vulnerab" in t or "cve" in t:
            return "CVEs"
        if "suricata" in t:
            return "events"
        if "compliance" in t or "sca" in t:
            return "checks"
        if "fim" in t:
            return "FIM events"
        if "alert" in t or "search_security" in t or "event" in t:
            return "alerts"
        return "records"

    def _result_breakdown(self, tool_name, result):
        """Small, meaningful per-domain breakdown for live progress (best-effort).
        Returns e.g. [{'label':'active','value':7,'tone':'ok'}, ...] or []."""
        t = (tool_name or "").lower()
        try:
            items = self._normalize_tool_payload(getattr(result, "data", result)).get("affected_items", [])
        except Exception:
            return []
        if not isinstance(items, list) or not items:
            return []
        out = []
        try:
            if "agent" in t:
                active = sum(1 for a in items if isinstance(a, dict) and str(a.get("status", "")).lower() == "active")
                offline = len(items) - active
                if active:
                    out.append({"label": "active", "value": active, "tone": "ok"})
                if offline:
                    out.append({"label": "offline", "value": offline, "tone": "warn"})
            elif "vulnerab" in t or "cve" in t:
                def _sev(v):
                    return str((v or {}).get("severity", "")).lower()
                crit = sum(1 for v in items if isinstance(v, dict) and _sev(v) == "critical")
                high = sum(1 for v in items if isinstance(v, dict) and _sev(v) == "high")
                if crit:
                    out.append({"label": "critical", "value": crit, "tone": "crit"})
                if high:
                    out.append({"label": "high", "value": high, "tone": "warn"})
            elif "alert" in t or "search_security" in t:
                def _lvl(a):
                    if not isinstance(a, dict):
                        return 0
                    r = a.get("rule") or {}
                    try:
                        return int(r.get("level") or a.get("level") or 0)
                    except Exception:
                        return 0
                crit = sum(1 for a in items if _lvl(a) >= 12)
                high = sum(1 for a in items if 8 <= _lvl(a) < 12)
                if crit:
                    out.append({"label": "critical", "value": crit, "tone": "crit"})
                if high:
                    out.append({"label": "high", "value": high, "tone": "warn"})
        except Exception:
            return []
        return out[:3]

    async def _emit_progress(self, cb, **payload):
        """Send one progress frame via the optional async callback. Best-effort:
        a callback error never interrupts query processing."""
        if cb is None:
            return
        try:
            await cb(payload)
        except Exception:
            pass

    async def process_query(self, user_query: str, context: Optional[QueryContext] = None, progress_cb=None) -> str:
        """
        Process natural language query end-to-end.

        Flow:
        1. Hybrid tool selection (rules + LLM fallback)
        2. Execute primary tool
        3. Execute secondary tools (parallel)
        4. Correlate results
        5. Generate LLM analysis
        6. Format response
        Args:
            user_query: Natural language query
            context: Optional query context

        Returns:
            Formatted markdown response
        """
        total_start = time.time()

        logger.info(f"Processing query: {user_query[:100]}...")

        # Pallas 4.1: Check response cache for identical recent queries.
        # Gated by ENABLE_RESPONSE_CACHE env var — off by default since stale
        # / error replies stuck for 3 min mask real backend issues during QA.
        cache_key = user_query.strip().lower()
        if self._cache_enabled:
            cached = self._response_cache.get(cache_key)
            if cached and (time.time() - cached["timestamp"]) < self._cache_ttl:
                logger.info(f"[CACHE HIT] Returning cached response for: {user_query[:60]}...")
                self._last_turn_data = cached.get("turn_data", {})
                return cached["response"]

        # Log follow-up detection
        if context and context.is_follow_up:
            logger.info(f"[CONTEXT] Follow-up query detected. Previous tool: {context.previous_tool}")
            logger.info(f"[CONTEXT] Carried entities: {context.carried_entities}")

        try:
            # Step 1: Hybrid tool selection (with carried entities from conversation)
            plan = await self.selector.select_tools(user_query, context)
            entities = getattr(self.selector, '_last_entities', {})

            # NLP Multi-entity fan-out: when 2+ agent_ids / agent_names / cve_ids
            # are present in the query AND the chosen tool is per-entity, expand
            # the plan so each entity gets its own tool call (parallel via the
            # existing execute_batch path). Cap N at MAX_FANOUT to bound cost.
            try:
                plan.fanout_entities = None  # default
                # Fleet-name scan with spell-tolerance: catches names typed with
                # spaces ("dev zeek" -> "dev-zeek"), typos ("dev zeak"), token
                # substrings ("zeek" -> "dev-zeek"), and host/agent/endpoint
                # synonyms (handled by the prefix words below).
                _existing_names = entities.get("agent_names") or []
                if not _existing_names or len(_existing_names) < 2:
                    try:
                        import difflib as _df
                        await self.selector._resolve_agent_name_to_id("__warmcache__")
                        _cache = self.selector._agent_name_cache or {}
                        if _cache:
                            _qlow = (user_query or "").lower()
                            _matched = []
                            _stop = {
                                "and", "or", "for", "with", "vs", "versus", "the",
                                "show", "get", "list", "compare", "find", "give",
                                "agent", "agents", "host", "hosts", "machine",
                                "machines", "endpoint", "endpoints", "server",
                                "servers", "system", "systems", "asset", "assets",
                                "vulnerabilities", "vulnerability", "ports",
                                "processes", "health", "details", "detail",
                                "alerts", "events", "all", "active", "inactive",
                                "disconnected", "windows", "linux", "macos",
                                "critical", "high", "medium", "low", "this",
                                "that", "these", "those",
                            }

                            def _add(cn):
                                if cn and cn not in _matched:
                                    _matched.append(cn)

                            _cache_names = list(_cache.keys())

                            # (1) Bare-substring scan across variants: handles
                            # "dev zeek" -> "dev-zeek" by stripping/swapping
                            # separators in the query.
                            _variants = {
                                _qlow,
                                _qlow.replace(" ", "-"),
                                _qlow.replace(" ", ""),
                                _qlow.replace("-", " "),
                                _qlow.replace("-", ""),
                            }
                            for _cn in _cache_names:
                                if len(_cn) < 4:
                                    continue
                                for _v in _variants:
                                    if _cn in _v:
                                        _add(_cn)
                                        break

                            # (2) Token scan + substring fallback for short
                            # distinctive tokens (e.g. "zeek" inside "dev-zeek").
                            _tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-.]+", _qlow)
                            _content_tokens = [t for t in _tokens if t not in _stop and len(t) >= 3]
                            for _tok in _content_tokens:
                                if _tok in _cache:
                                    _add(_tok)
                                    continue
                                # Substring within any fleet name
                                if len(_tok) >= 4:
                                    _hit = next((cn for cn in _cache_names if _tok in cn), None)
                                    if _hit:
                                        _add(_hit)
                                        continue
                                # Spell-tolerance: closest fleet name by
                                # difflib ratio (handles typos like "zeak"
                                # for "zeek", "uhsna" for "ushna-macos").
                                if len(_tok) >= 4:
                                    _close = _df.get_close_matches(_tok, _cache_names, n=1, cutoff=0.72)
                                    if _close:
                                        _add(_close[0])

                            # (3) Adjacent-token joins ("dev" + "zeek" ->
                            # "dev-zeek" / "devzeek" / "dev_zeek") matched
                            # against the cache, plus a spell-tolerant fall-back.
                            for _i in range(len(_content_tokens) - 1):
                                _a, _b = _content_tokens[_i], _content_tokens[_i + 1]
                                for _joined in (f"{_a}-{_b}", f"{_a}{_b}", f"{_a}_{_b}"):
                                    if _joined in _cache:
                                        _add(_joined)
                                        break
                                else:
                                    _candidates = (f"{_a}-{_b}", f"{_a}{_b}")
                                    for _cand in _candidates:
                                        if len(_cand) >= 5:
                                            _close = _df.get_close_matches(_cand, _cache_names, n=1, cutoff=0.78)
                                            if _close:
                                                _add(_close[0])
                                                break

                            if len(_matched) >= 2:
                                entities["agent_names"] = _matched
                                logger.info(f"[MULTI-ENTITY] fleet-scan(spell) found {len(_matched)} agent_names: {_matched}")
                    except Exception as _fs_exc:
                        logger.debug(f"[MULTI-ENTITY] fleet-scan(spell) failed: {_fs_exc}")
                PER_AGENT_TOOLS = {
                    "check_agent_health", "get_agent_processes", "get_agent_ports",
                    "get_agent_inventory", "get_agent_configuration",
                    "get_sca_results", "get_rootcheck_results", "get_agent_packages",
                    "get_fim_events",
                    # Vulnerability tools also accept agent_id as a filter, so
                    # "compare vulnerabilities for A and B" can fan out per-agent.
                    "get_wazuh_vulnerabilities", "get_wazuh_critical_vulnerabilities",
                    "get_defender_vulnerabilities",
                }
                PER_CVE_TOOLS = {"get_wazuh_vulnerabilities", "get_defender_vulnerabilities"}
                MAX_FANOUT = 5

                # Build the agent-id fan-out list: prefer agent_ids; otherwise
                # resolve agent_names to ids.
                fanout_ids = []
                fanout_labels = {}
                aids = entities.get("agent_ids") or []
                if isinstance(aids, list) and len(aids) >= 2:
                    fanout_ids = list(dict.fromkeys(aids))
                names = entities.get("agent_names") or []
                if isinstance(names, list) and len(names) >= 2 and not fanout_ids:
                    resolved = await self.selector._resolve_agent_names_to_ids(names)
                    for nm in names:
                        rid = resolved.get(nm)
                        if rid and rid not in fanout_ids:
                            fanout_ids.append(rid)
                            fanout_labels[rid] = nm
                if names and len(fanout_ids) >= 2 and not fanout_labels:
                    # If we got ids from `agent_ids`, also try to look up names
                    try:
                        # Best-effort: trigger a cache refresh, then look up names by id
                        await self.selector._resolve_agent_name_to_id(names[0] if names else "")
                        rev = {v: k for k, v in (self.selector._agent_name_cache or {}).items()}
                        for rid in fanout_ids:
                            if rid in rev:
                                fanout_labels[rid] = rev[rid]
                    except Exception:
                        pass

                # Detect implicit fan-out: Qwen already produced a same-tool plan
                # (e.g. primary=get_wazuh_vulnerabilities + secondary=[get_wazuh_vulnerabilities])
                # with DIFFERENT per-entity args. If so, surface it for the renderer
                # so each agent gets its own section.
                _qwen_dup_count = sum(1 for t in (plan.secondary_tools or []) if t == plan.primary_tool)
                _qwen_implicit_fanout = False
                if _qwen_dup_count >= 1 and plan.primary_tool in PER_AGENT_TOOLS:
                    # Collect per-entity args across primary + matching secondaries
                    _calls = [(plan.primary_args or {})]
                    for _t, _a in zip(plan.secondary_tools or [], plan.secondary_args or []):
                        if _t == plan.primary_tool:
                            _calls.append(_a or {})
                    # Check whether agent_id differs across calls
                    _aids_seen = [str(c.get("agent_id")) for c in _calls if c.get("agent_id")]
                    if len(set(_aids_seen)) >= 2:
                        _qwen_implicit_fanout = True
                        logger.info(f"[MULTI-ENTITY] Qwen implicit fan-out detected: agent_ids={_aids_seen}")
                        # Build fanout_entities from Qwen's plan directly
                        # Try to enrich labels with agent names from cache
                        try:
                            await self.selector._resolve_agent_name_to_id("__warmcache__")
                        except Exception:
                            pass
                        _cache = (self.selector._agent_name_cache or {})
                        _name_by_id = {v: k for k, v in _cache.items()}
                        plan.fanout_entities = []
                        for c in _calls:
                            _aid = str(c.get("agent_id", ""))
                            _nm = _name_by_id.get(_aid, "")
                            plan.fanout_entities.append({
                                "label": f"Agent {_aid}" + (f" — {_nm}" if _nm else ""),
                                "agent_id": _aid,
                                "args": dict(c),
                            })
                        plan.requires_correlation = False
                        plan.correlation_strategy = None
                        plan.selection_method = f"{plan.selection_method}+multi_entity_fanout"

                if (
                    plan.primary_tool in PER_AGENT_TOOLS
                    and len(fanout_ids) >= 2
                    and not _qwen_implicit_fanout
                    and not (plan.secondary_tools and any(t == plan.primary_tool for t in plan.secondary_tools))
                ):
                    fanout_ids = fanout_ids[:MAX_FANOUT]
                    plan.primary_args = dict(plan.primary_args or {})
                    plan.primary_args["agent_id"] = fanout_ids[0]
                    plan.secondary_tools = [plan.primary_tool] * (len(fanout_ids) - 1)
                    plan.secondary_args = [{"agent_id": aid} for aid in fanout_ids[1:]]
                    plan.requires_correlation = False
                    plan.correlation_strategy = None
                    plan.selection_method = f"{plan.selection_method}+multi_entity_fanout"
                    plan.fanout_entities = [
                        {
                            "label": f"Agent {aid}" + (f" — {fanout_labels[aid]}" if aid in fanout_labels else ""),
                            "agent_id": aid,
                            "args": ({"agent_id": fanout_ids[0]} if i == 0 else {"agent_id": aid}),
                        }
                        for i, aid in enumerate(fanout_ids)
                    ]
                    logger.info(f"[MULTI-ENTITY] fan-out: {plan.primary_tool} x{len(fanout_ids)} -> {fanout_ids}")

                # CVE fan-out (for vulnerability tools)
                cves = entities.get("cve_ids") or []
                if (
                    plan.primary_tool in PER_CVE_TOOLS
                    and isinstance(cves, list) and len(cves) >= 2
                    and not plan.fanout_entities
                ):
                    cves = cves[:MAX_FANOUT]
                    plan.primary_args = dict(plan.primary_args or {})
                    plan.primary_args["cve_id"] = cves[0]
                    plan.secondary_tools = [plan.primary_tool] * (len(cves) - 1)
                    plan.secondary_args = [{"cve_id": c} for c in cves[1:]]
                    plan.requires_correlation = False
                    plan.correlation_strategy = None
                    plan.selection_method = f"{plan.selection_method}+multi_entity_fanout"
                    plan.fanout_entities = [
                        {"label": cves[i], "cve_id": cves[i], "args": ({"cve_id": cves[0]} if i == 0 else {"cve_id": cves[i]})}
                        for i in range(len(cves))
                    ]
                    logger.info(f"[MULTI-ENTITY] CVE fan-out: {plan.primary_tool} x{len(cves)} -> {cves}")
            except Exception as _fo_exc:
                logger.warning(f"[MULTI-ENTITY] fan-out planner failed: {_fo_exc}")

            # NLP time_range authoritative-merge: our entity parser is more
            # reliable than the LLM for absolute windows like "9 AM to 5 PM"
            # (Qwen sometimes echoes the placeholder "<today>" literally) and
            # for minutes ("30m"). When the parser produced a concrete value
            # we override whatever the LLM put in primary_args.secondary_args.
            try:
                _ent_tr = (entities or {}).get("time_range")
                if _ent_tr:
                    _existing = (plan.primary_args or {}).get("time_range") or ""
                    _existing_str = str(_existing)
                    # Replace if the LLM produced a placeholder, an empty value,
                    # or any value different from our parser's authoritative one.
                    _placeholder = ("<today>" in _existing_str or "<yesterday>" in _existing_str
                                    or "<tomorrow>" in _existing_str)
                    if _placeholder or not _existing_str or _existing_str != _ent_tr:
                        plan.primary_args = dict(plan.primary_args or {})
                        plan.primary_args["time_range"] = _ent_tr
                        # Propagate into any secondary_args that already mention time_range
                        _sec = list(plan.secondary_args or [])
                        for i, _a in enumerate(_sec):
                            if isinstance(_a, dict) and "time_range" in _a:
                                _a = dict(_a)
                                _a["time_range"] = _ent_tr
                                _sec[i] = _a
                        plan.secondary_args = _sec
                        logger.info(f"[TIME_RANGE] override: parser={_ent_tr!r} overrode LLM value {_existing_str!r}")
            except Exception as _tr_e:
                logger.warning(f"[TIME_RANGE] authoritative-merge failed: {_tr_e}")

            logger.info(f"[DEBUG] Selected primary_tool={plan.primary_tool}, args={plan.primary_args}")
            logger.info(f"[DEBUG] Secondary tools={plan.secondary_tools}, secondary_args={plan.secondary_args}")
            logger.info(f"[DEBUG] Correlation strategy={plan.correlation_strategy}, requires_correlation={plan.requires_correlation}")
            logger.info(f"   Confidence: {plan.confidence:.2f}, Method: {plan.selection_method}")

            # Guardrail: Very low confidence — return intelligent guidance
            if plan.confidence < 0.3 and plan.selection_method in ("llm", "fallback"):
                if plan.fallback_guidance:
                    return plan.fallback_guidance
                return self.selector._build_fallback_guidance(user_query)


            # === PALLAS_NLTEST_IOC_FIX === DO NOT REMOVE
            # Added 20260512T122707Z: when check_ioc_reputation is routed but the indicator arg
            # is missing, scan the raw query for an IPv4 / domain / hash and inject.
            if plan.primary_tool == "check_ioc_reputation" and not plan.primary_args.get("indicator"):
                _ioc_re_v4 = re.search(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", user_query)
                _ioc_re_hash = re.search(r"\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b", user_query)
                _ioc_re_domain = re.search(r"\b([a-zA-Z0-9-]+\.(?:com|net|org|io|co|me|app|dev|info|biz|gov|edu)(?:\.[a-z]{2,3})?)\b", user_query)
                if _ioc_re_v4:
                    plan.primary_args["indicator"] = _ioc_re_v4.group(0)
                    plan.primary_args.setdefault("indicator_type", "ip")
                elif _ioc_re_hash:
                    h = _ioc_re_hash.group(0)
                    plan.primary_args["indicator"] = h
                    plan.primary_args.setdefault("indicator_type", "md5" if len(h)==32 else ("sha1" if len(h)==40 else "sha256"))
                elif _ioc_re_domain:
                    plan.primary_args["indicator"] = _ioc_re_domain.group(0)
                    plan.primary_args.setdefault("indicator_type", "domain")
            # === END PALLAS_NLTEST_IOC_FIX ===

            # Guardrail: Agent deep-dive tools require a specific agent_id
            # Extended 2026-05-12 (NL-test): added sca/fim/rootcheck/inventory/packages
            _AGENT_SCOPED = (
                "get_agent_ports", "get_agent_processes",
                "get_agent_configuration", "check_agent_health",
                "get_sca_results", "get_fim_events", "get_rootcheck_results",
                "get_agent_inventory", "get_agent_packages",
            )
            if plan.primary_tool in _AGENT_SCOPED:
                if not plan.primary_args.get("agent_id"):
                    return (
                        f"To run **{plan.primary_tool}** I need the **agent ID** (or exact agent name).\n\n"
                        "Examples:\n"
                        "- `Show open ports for agent 001`\n"
                        "- `List running processes for agent 015`\n"
                        "- `Get configuration for agent 001`\n"
                        "- `Check health for agent 001`\n"
                        "- `Show sca results for agent 011`\n"
                        "- `Show fim events for agent 019`\n\n"
                        "Tip: try `show active agents` first to see which agents are available."
                    )


            # Live progress: strategy chosen
            await self._emit_progress(
                progress_cb, stage="strategy",
                tool=plan.primary_tool,
                sources=1 + len(plan.secondary_tools or []),
                correlation=plan.correlation_strategy,
                confidence=plan.confidence,
                selection_method=plan.selection_method,
            )

            # Step 2: Execute primary tool
            primary_result = await self.executor.execute(
                plan.primary_tool,
                plan.primary_args
            )

            if not primary_result.success:
                return self._format_error_response(primary_result, plan)

            # Live progress: primary tool queried (real count + domain breakdown)
            await self._emit_progress(
                progress_cb, stage="querying",
                tool=plan.primary_tool,
                source=plan.primary_tool,
                events=self._extract_count(primary_result),
                sources=1,
                noun=self._noun_for_tool(plan.primary_tool),
                breakdown=self._result_breakdown(plan.primary_tool, primary_result),
            )

            # ================================================================
            # M5 (2026-06-10) PALLAS_P4_AUTO_AGGREGATE - kick off the matching
            # aggregation tool in parallel so the response can include a
            # full-population analysis section above the sample table. Result
            # awaited later before formatting. Any failure is silently logged
            # and the response is identical to pre-M5.
            # ================================================================
            m5_task = None
            m5_aug_tool = None
            m5_primary_total = None
            try:
                _m5_on = os.getenv("PALLAS_P4_AUTO_AGGREGATE", "true").lower() != "false"
                if _m5_on and plan.primary_tool in _M5_AGG_AUGMENT_MAP:
                    _agg_tool = _M5_AGG_AUGMENT_MAP[plan.primary_tool]
                    if _agg_tool not in (plan.secondary_tools or []):
                        # Peek the true total + sample size from the primary
                        # payload. Handles BOTH plain dict shapes AND the MCP
                        # envelope {"content":[{"type":"text","text":"<json>"}]}
                        # that Suricata + Wazuh tools use.
                        def _m5_unwrap_mcp(p):
                            """If p is an MCP envelope, return the parsed inner
                            JSON. Otherwise return p unchanged."""
                            try:
                                if (
                                    isinstance(p, dict)
                                    and isinstance(p.get("content"), list)
                                    and p["content"]
                                ):
                                    first = p["content"][0]
                                    if isinstance(first, dict) and first.get("type") == "text":
                                        txt = first.get("text", "")
                                        if isinstance(txt, str) and txt.strip().startswith("{"):
                                            return json.loads(txt)
                            except Exception:
                                pass
                            return p
                        def _m5_peek(p):
                            try:
                                p = _m5_unwrap_mcp(p)
                                if isinstance(p, dict):
                                    for keys in (
                                        ("data", "total_affected_items"),
                                        ("total_affected_items",),
                                        ("total_alerts",),
                                        ("data", "total_alerts"),
                                        ("total",),
                                        ("data", "total"),
                                        ("hits", "total", "value"),
                                    ):
                                        cur = p; ok = True
                                        for k in keys:
                                            if isinstance(cur, dict) and k in cur:
                                                cur = cur[k]
                                            else:
                                                ok = False; break
                                        if ok and isinstance(cur, int) and cur >= 0:
                                            return cur
                            except Exception:
                                pass
                            return None
                        _peeked_total = _m5_peek(primary_result.data)
                        _peeked_sample = None
                        try:
                            _p = _m5_unwrap_mcp(primary_result.data)
                            if isinstance(_p, dict):
                                _d = (
                                    _p.get("data")
                                    if isinstance(_p.get("data"), dict)
                                    else _p
                                )
                                _items = (
                                    _d.get("affected_items")
                                    or _d.get("alerts")
                                    or _d.get("items")
                                    or []
                                )
                                if isinstance(_items, list):
                                    _peeked_sample = len(_items)
                        except Exception:
                            _peeked_sample = None
                        logger.info(
                            f"[M5_AUGMENT] peek result: primary={plan.primary_tool} "
                            f"total={_peeked_total} sample={_peeked_sample}"
                        )
                        if (
                            isinstance(_peeked_total, int)
                            and isinstance(_peeked_sample, int)
                            and _peeked_sample > 0
                            and _peeked_total > _peeked_sample * 2
                        ):
                            _agg_args = {
                                k: v for k, v in (plan.primary_args or {}).items()
                                if k in ("time_range", "severity", "min_severity", "agent_id")
                            }
                            m5_aug_tool = _agg_tool
                            m5_primary_total = _peeked_total
                            m5_task = asyncio.create_task(
                                asyncio.wait_for(
                                    self.executor.execute(_agg_tool, _agg_args),
                                    timeout=10.0,
                                )
                            )
                            logger.info(
                                f"[M5_AUGMENT] dispatched {plan.primary_tool} "
                                f"-> {_agg_tool}: total={_peeked_total}, sample={_peeked_sample}"
                            )
            except Exception as _m5_dispatch_exc:
                logger.warning(f"[M5_AUGMENT] dispatch failed: {_m5_dispatch_exc}", exc_info=True)
                m5_task = None
            # Reset prior-turn M5 state so a fresh query doesn't inherit it
            self._m5_aug_used = False
            self._m5_aug_tool = None
            self._m5_primary_total = None

            # Step 3: Execute secondary tools (parallel)
            secondary_results = []
            if plan.secondary_tools:
                # Backfill missing required args from primary_args + entities so
                # secondary tools in agent-scoped investigations inherit the
                # agent_id, time_range, etc. that the user supplied. Without this,
                # tools like get_rootcheck_results die with
                # "Required field 'agent_id' missing".
                _primary_args = plan.primary_args or {}
                _backfilled_args: List[Dict[str, Any]] = []
                for _sec_tool, _sec_args in zip(plan.secondary_tools, plan.secondary_args):
                    _args = dict(_sec_args or {})
                    try:
                        _meta = ToolRegistry.get_tool(_sec_tool)
                        _schema = getattr(_meta, "input_schema", {}) if _meta else {}
                    except Exception:
                        _schema = {}
                    for _field, _spec in _schema.items():
                        if not isinstance(_spec, dict):
                            continue
                        if not _spec.get("required"):
                            continue
                        if _args.get(_field) not in (None, ""):
                            continue
                        # Try primary_args first, then entities (agent_id only).
                        _val = _primary_args.get(_field)
                        if _val in (None, "") and _field == "agent_id":
                            _val = (entities or {}).get("agent_id")
                        if _val not in (None, ""):
                            _args[_field] = _val
                            logger.info(
                                f"[SEC_BACKFILL] {_sec_tool}: injected {_field}={_val!r} "
                                f"from primary"
                            )
                    _backfilled_args.append(_args)
                tool_specs = list(zip(plan.secondary_tools, _backfilled_args))
                secondary_results = await self.executor.execute_batch(tool_specs)

                # Log failures
                failed = [r for r in secondary_results if not r.success]
                if failed:
                    logger.warning(f"⚠️ {len(failed)} secondary tool(s) failed")

            # Live progress: secondary tools queried (cumulative events)
            _ok_secondary = [r for r in secondary_results if getattr(r, "success", False)]
            await self._emit_progress(
                progress_cb, stage="querying",
                events=self._extract_count(primary_result) + sum(self._extract_count(r) for r in _ok_secondary),
                sources=1 + len(_ok_secondary),
                tools=1 + len(secondary_results),
            )

            # Step 4: Correlate results
            correlated_data = None
            if plan.requires_correlation and secondary_results:
                successful_secondary = [r for r in secondary_results if r.success]
                if successful_secondary:
                    try:
                        logger.info(f"[CORRELATION] Strategy: {plan.correlation_strategy}")
                        correlated_data = self.correlator.correlate(
                            strategy=plan.correlation_strategy,
                            primary_result=primary_result.data,
                            secondary_results=[r.data for r in successful_secondary],
                            context={**entities, "query": user_query}
                        )
                        logger.info(f"[CORRELATION] Applied: {correlated_data.correlation_type}, items={len(correlated_data.correlated_data)}")
                    except Exception as e:
                        logger.error(f"Correlation failed: {e}")

            # Live progress: correlation done (signals = correlated rows)
            await self._emit_progress(
                progress_cb, stage="correlating",
                signals=(len(correlated_data.correlated_data) if correlated_data else 0),
                correlation=plan.correlation_strategy,
            )

            # Step 5: Format response FIRST (before LLM — so LLM gets clean, filtered data)
            if getattr(plan, "fanout_entities", None):
                # Multi-entity fan-out: render each per-entity result separately,
                # then stitch with section headers. Each call shares plan.primary_tool
                # so per-tool formatters apply uniformly.
                _per_results = [primary_result] + list(secondary_results)
                _labels = [e.get("label") for e in plan.fanout_entities]
                _parts = []
                for _label, _r in zip(_labels, _per_results):
                    if not _r.success:
                        _parts.append(f"## {_label}\n\n_⚠ Tool call failed: {_r.error or 'unknown error'}_\n")
                        continue
                    _section = self._format_response(
                        plan=plan,
                        primary_result=_r,
                        secondary_results=[],
                        correlated_data=None,
                        llm_analysis=None,
                        user_query=user_query,
                    )
                    _parts.append(f"## {_label}\n\n{_section}")
                response = "\n\n---\n\n".join(_parts)
                logger.info(f"[MULTI-ENTITY] rendered {len(_per_results)} per-entity sections")
            else:
                response = self._format_response(
                    plan=plan,
                    primary_result=primary_result,
                    secondary_results=secondary_results,
                    correlated_data=correlated_data,
                    llm_analysis=None,
                    user_query=user_query
                )

            # ================================================================
            # M5 (2026-06-10) PALLAS_P4_AUTO_AGGREGATE - await the aggregation
            # task kicked off after primary execute and prepend its formatted
            # output below the first ## heading so the response leads with
            # population-level analysis above the sample table.
            # ================================================================
            try:
                if m5_task is not None:
                    _agg_result = None
                    try:
                        _agg_result = await m5_task
                    except asyncio.TimeoutError:
                        logger.warning(f"[M5_AUGMENT] {m5_aug_tool} timed out at 10s")
                    except Exception as _m5_await_exc:
                        logger.warning(f"[M5_AUGMENT] {m5_aug_tool} await failed: {_m5_await_exc}")
                    if _agg_result is not None and getattr(_agg_result, "success", False):
                        _agg_section = self._format_m5_section(
                            _agg_result.data, m5_aug_tool, m5_primary_total
                        )
                        if _agg_section:
                            _lines = response.split("\n")
                            _insert_idx = None
                            for _i, _ln in enumerate(_lines):
                                if _ln.startswith("## "):
                                    _insert_idx = _i + 1
                                    break
                            if _insert_idx is None:
                                response = _agg_section + "\n\n" + response
                            else:
                                while _insert_idx < len(_lines) and not _lines[_insert_idx].strip():
                                    _insert_idx += 1
                                _lines[_insert_idx:_insert_idx] = ["", _agg_section, ""]
                                response = "\n".join(_lines)
                            self._m5_aug_used = True
                            self._m5_aug_tool = m5_aug_tool
                            self._m5_primary_total = m5_primary_total
                            logger.info(
                                f"[M5_AUGMENT] prepended aggregation section "
                                f"(tool={m5_aug_tool}, total={m5_primary_total})"
                            )
            except Exception as _m5_outer_exc:
                logger.warning(f"[M5_AUGMENT] prepend block failed: {_m5_outer_exc}", exc_info=True)

            # NLP Hunt auto-expand: when an LLM-routed hunt query targets the
            # alerts indexer and the result was truncated (10k of N where N>>10k),
            # automatically run get_wazuh_alert_summary on the full corpus so the
            # analyst sees the indexer aggregations alongside the sample analysis.
            try:
                _sm = (plan.selection_method or "")
                _is_hunt = ("llm_hunt" in _sm) or ("hunt_fallback" in _sm) or ("llm_router_hunt" in _sm)
                _alerts_family = {"get_wazuh_alerts", "search_security_events", "get_suricata_alerts"}
                if _is_hunt and plan.primary_tool in _alerts_family:
                    try:
                        _norm = self._normalize_tool_payload(primary_result.data)
                    except Exception:
                        _norm = {}
                    _items = _norm.get("affected_items", []) if isinstance(_norm, dict) else []
                    _total = _norm.get("total_affected_items", len(_items)) if isinstance(_norm, dict) else len(_items)
                    logger.info(f"[HUNT-EXPAND] sample check: items={len(_items)} total={_total}")
                    if isinstance(_total, int) and isinstance(_items, list) and _total > len(_items):
                        _time_range = (plan.primary_args or {}).get("time_range") or "24h"
                        # Pick the right summary tool for the indexer in play
                        _summary_tool = {
                            "get_wazuh_alerts": "get_wazuh_alert_summary",
                            "search_security_events": "get_wazuh_alert_summary",
                            "get_suricata_alerts": "get_suricata_alert_summary",
                        }.get(plan.primary_tool, "get_wazuh_alert_summary")
                        logger.info(f"[HUNT-EXPAND] sample truncated ({len(_items)} of {_total}); appending {_summary_tool} on time_range={_time_range}")
                        _summary_res = await self.executor.execute(_summary_tool, {"time_range": _time_range})
                        if _summary_res.success and _summary_res.data:
                            try:
                                _summary_md = self._extract_raw_text(_summary_res.data, "")
                            except Exception:
                                _summary_md = ""
                            if _summary_md and _summary_md.strip():
                                response += (
                                    "\n\n---\n\n"
                                    f"## Full-Corpus Summary (no sampling — all {_total:,} alerts aggregated)\n\n"
                                    + _summary_md.strip()
                                )
                                logger.info(f"[HUNT-EXPAND] appended {len(_summary_md)} chars of full-corpus summary")
            except Exception as _xe_exc:
                logger.warning(f"[HUNT-EXPAND] failed: {_xe_exc}")

            # Live progress: formatting done
            await self._emit_progress(progress_cb, stage="formatting")

            # Step 6: Generate SOC insight (query intro + analysis) using formatted response
            await self._emit_progress(progress_cb, stage="soc")
            soc_insight = await self._generate_soc_insight(plan, user_query, response, context)

            # Step 7: Assemble final output — header + intro + data + SOC analysis + suggestions + metadata
            final = ""

            # Deterministic query understanding header
            query_header = self._build_query_header(plan, user_query, primary_result, correlated_data)
            final += query_header

            # Query interpretation intro (blockquote style)
            if soc_insight.get("intro"):
                final += f"> {soc_insight['intro']}\n\n---\n\n"

            # Formatted data response
            final += response

            # SOC Analysis section
            soc_analysis = soc_insight.get("analysis") or self._static_soc_fallback(plan.primary_tool)
            final += f"\n\n---\n\n## SOC ANALYSIS\n\n{soc_analysis}\n"

            total_time = time.time() - total_start
            logger.info(f"Query processed in {total_time:.2f}s")

            # Suggested follow-up queries
            suggested = self._build_suggested_queries(plan, correlated_data)
            if suggested:
                final += suggested

            # Metadata footer
            final += "\n---\n"
            final += f"*Selection: {plan.selection_method} ({plan.confidence:.0%} confidence)*\n"
            final += f"*Primary tool: {primary_result.tool_name} ({primary_result.execution_time:.2f}s)*\n"

            if plan.secondary_tools:
                successful_secondary = [r for r in secondary_results if r.success]
                # PALLAS_P4_PARTIAL_SUCCESS (Track B6, fixes audit F7) - when any
                # secondary failed, surface it as a styled callout above the
                # metadata footer so analysts notice. Italic footer kept for
                # back-compat with anyone parsing the existing format.
                _failed_secondary = [r for r in secondary_results if not r.success]
                if _failed_secondary and os.getenv(
                    "PALLAS_P4_PARTIAL_SUCCESS", "true"
                ).lower() != "false":
                    _failed_names = ", ".join(
                        f"`{getattr(r, 'tool_name', '?')}`" for r in _failed_secondary[:5]
                    )
                    _ok = len(successful_secondary)
                    _total = len(plan.secondary_tools)
                    final = (
                        f"\n> **Partial result** -- {_ok} of {_total} secondary tools "
                        f"succeeded. Failed: {_failed_names}. Open Inspect for details.\n\n"
                        + final
                    )
                final += f"*Secondary tools: {len(successful_secondary)}/{len(plan.secondary_tools)} successful*\n"

            if correlated_data:
                final += f"*Correlation: {correlated_data.correlation_type}*\n"

            final += f"\n*Total execution time: {total_time:.2f}s*"

            # Store turn data for conversation memory + v2 metadata (server.py picks this up)
            try:
                normalized = self._normalize_tool_payload(primary_result.data)
                result_data = normalized.get("affected_items", [])
            except Exception:
                result_data = []

            # Build suggestion list (without markdown formatting) for v2 response
            v2_suggestions = self._get_suggestion_list(plan, correlated_data)

            # NLP Multi-entity graceful fallback: if the analyst named 2+ entities
            # but fan-out wasn't applied (e.g., correlation-style query), prepend
            # clickable suggestions so they can pull the missing pieces in one click.
            try:
                if not getattr(plan, "fanout_entities", None):
                    _extras = []
                    _aids = entities.get("agent_ids") or []
                    _names = entities.get("agent_names") or []
                    _cves = entities.get("cve_ids") or []
                    if isinstance(_aids, list) and len(_aids) >= 2:
                        # Skip the first one (presumably already covered)
                        for _aid in _aids[1:4]:
                            _extras.append(f"show health for agent {_aid}")
                    elif isinstance(_names, list) and len(_names) >= 2:
                        # When agent names are involved, surface the canonical
                        # resolved name (or flag that one didn't resolve).
                        try:
                            _resolved_map = await self.selector._resolve_agent_names_to_ids(_names)
                            for _nm in _names[1:4]:
                                rid = _resolved_map.get(_nm)
                                if rid:
                                    _extras.append(f"show health for agent {rid}")
                                else:
                                    _extras.append(f"agent named {_nm!r} not found in fleet")
                        except Exception:
                            for _nm in _names[1:4]:
                                _extras.append(f"show health for agent {_nm}")
                    if isinstance(_cves, list) and len(_cves) >= 2:
                        for _c in _cves[1:4]:
                            _extras.append(f"show details for {_c}")
                    if _extras:
                        # Insert at the FRONT so they appear first in the chip row
                        v2_suggestions = _extras + (v2_suggestions or [])
                        logger.info(f"[MULTI-ENTITY] added {len(_extras)} fallback suggestions")
            except Exception as _gs_exc:
                logger.warning(f"[MULTI-ENTITY] graceful fallback suggestion failed: {_gs_exc}")

            # Build the strategy display string with filter context appended.
            # This makes the response panel header read "Critical Vulnerabilities — Windows"
            # instead of just "Critical Vulnerabilities" when the user filtered by OS.
            # Without this, the header looks generic and the headline KPI can be misread
            # as the answer to an unfiltered question.
            strategy_base = plan.correlation_strategy or plan.primary_tool
            _filter_parts = []
            if plan.primary_args:
                _OS_DISPLAY = {"windows": "Windows", "linux": "Linux", "darwin": "macOS"}
                _os = plan.primary_args.get("os")
                if _os:
                    _filter_parts.append(_OS_DISPLAY.get(str(_os).lower(), str(_os).title()))
                _agent = plan.primary_args.get("agent_id")
                if _agent:
                    _filter_parts.append(f"Agent {_agent}")
                _sev = plan.primary_args.get("severity")
                if _sev:
                    _filter_parts.append(str(_sev).title())
                _cve = plan.primary_args.get("cve_id")
                if _cve:
                    _filter_parts.append(str(_cve))
                _ip = plan.primary_args.get("ip")
                if _ip:
                    _filter_parts.append(f"IP {_ip}")
                _fw = plan.primary_args.get("framework")
                if _fw:
                    _filter_parts.append(str(_fw).upper())
            strategy_display = (
                f"{strategy_base} — {' · '.join(_filter_parts)}"
                if _filter_parts else strategy_base
            )

            self._last_turn_data = {
                "query": user_query,
                "tool_used": plan.primary_tool,
                "entities": entities,
                "result_type": self._infer_result_type(plan.primary_tool),
                "key_ids": self._extract_key_ids(result_data, plan.primary_tool),
                "record_count": len(result_data),
                "result_summary": final[:200],
                # v2 metadata fields for structured frontend response
                "metadata": {
                    "strategy": strategy_display,
                    "primary_tool": plan.primary_tool,
                    "secondary_tools": list(plan.secondary_tools) if plan.secondary_tools else [],
                    "confidence": plan.confidence,
                    "execution_time_ms": int(total_time * 1000),
                    "correlation_type": correlated_data.correlation_type if correlated_data else None,
                    "selection_method": plan.selection_method,
                    # Structured filters (separate from strategy string) so the SPA can
                    # also render them as chips, badges, or breadcrumbs if it wants to
                    # without parsing the strategy string.
                    "filters": {
                        k: v for k, v in (plan.primary_args or {}).items()
                        if k in ("os", "agent_id", "severity", "cve_id", "ip", "framework", "time_range")
                        and v not in (None, "", [])
                    },
                    # ================================================================
                    # PALLAS_P4_OBSERVABILITY (Track C-backend, v2.4)
                    # Adds fields the v2.4 DiagnosticsPanel will read. Safe to
                    # ignore on the v2.3 frontend - additive only.
                    # ================================================================
                    "reasoning": (plan.reasoning or "") if hasattr(plan, "reasoning") else "",
                    "escalated_to_tier2": bool(
                        getattr(plan, "escalated_to_tier2", False)
                        or (plan.selection_method or "").endswith("_complex")
                    ),
                    # Per-tool execution timing for the Timeline tab.
                    "tool_timings": [
                        {
                            "tool": primary_result.tool_name,
                            "role": "primary",
                            "duration_ms": int((primary_result.execution_time or 0) * 1000),
                            "success": bool(getattr(primary_result, "success", False)),
                        }
                    ] + [
                        {
                            "tool": getattr(_sr, "tool_name", "?"),
                            "role": "secondary",
                            "duration_ms": int((getattr(_sr, "execution_time", 0) or 0) * 1000),
                            "success": bool(getattr(_sr, "success", False)),
                            "error": getattr(_sr, "error", None),
                        }
                        for _sr in (secondary_results or [])
                    ],
                    # Structured payload for v2.4 follow-up queries (A3). Top 20
                    # normalized affected_items so the next turn can filter/sort
                    # without re-running the tool.
                    "result_items": (result_data or [])[:20],
                    # Status of degradation guards (B2/B3/B6/B7) - lets the
                    # DiagnosticsPanel show "what was skipped or trimmed".
                    "degradation": {
                        "b2_correlation_skipped": getattr(self, "_b2_correlation_skipped", None),
                        "b3_promoted_secondary": bool(getattr(self, "_b3_footer", "")),
                        "b6_failed_secondary_count": sum(
                            1 for _sr in (secondary_results or [])
                            if not getattr(_sr, "success", False)
                        ),
                        # M5 (2026-06-10) - auto-aggregation marker so the
                        # Diagnostics Routing tab can show "Auto-aggregated from
                        # X, population: Y" when the bonus section was prepended.
                        "m5_aggregation_used": bool(getattr(self, "_m5_aug_used", False)),
                        "m5_aggregation_tool": getattr(self, "_m5_aug_tool", None),
                        "m5_primary_total": getattr(self, "_m5_primary_total", None),
                    },
                    # M2a (2026-06-10): surface the real follow-up + memory
                    # injection state so the Diagnostics Memory tab stops
                    # showing "(no follow-up)" for every response.
                    "memory": {
                        "injected": bool(context and context.is_follow_up),
                        "reason": (
                            f"Follow-up pattern matched: `{getattr(context, 'follow_up_pattern', None)}`"
                            if context and getattr(context, "follow_up_pattern", None)
                            else (
                                "Follow-up detected"
                                if (context and context.is_follow_up)
                                else "(no follow-up)"
                            )
                        ),
                        "content": (
                            (context.conversation_summary or "")[:1000]
                            if context and context.conversation_summary
                            else None
                        ),
                        "carried_entities": (
                            dict(context.carried_entities)
                            if context and context.carried_entities
                            else {}
                        ),
                        "previous_tool": (context.previous_tool if context else None),
                        # M14c (2026-06-11): entities the carry-merge dropped
                        # because the new query did not reference them (topic
                        # pivot). Surfaced so analysts can see WHY a follow-up
                        # routed fleet-wide instead of inheriting agent scope.
                        "dropped_carries": list(getattr(self.selector, "_m14_dropped_carries", []) or []),
                    },
                },
                "suggestions": v2_suggestions,
            }

            # NLP Sprint 1 #6/#7: enrich metadata with intent + executive summary
            # + (when JSON-mode SOC is enabled) the structured SOC payload.
            try:
                from .intent_map import resolve_intent as _resolve_intent
                from .executive_summary import build_executive_summary as _build_es
                _ct = correlated_data.correlation_type if correlated_data else None
                _intent = _resolve_intent(plan.primary_tool, _ct)
                self._last_turn_data["metadata"]["intent"] = _intent

                _structured = soc_insight.get("structured") if isinstance(soc_insight, dict) else None
                if _structured and isinstance(_structured, dict):
                    self._last_turn_data["metadata"]["soc_structured"] = _structured
                    self._last_turn_data["metadata"]["executive_summary"] = {
                        "headline":     _structured.get("headline", "") or "",
                        "context":      _structured.get("context", "") or "",
                        "significance": _structured.get("significance", "") or "",
                        "priority":     _structured.get("priority"),
                    }
                else:
                    _intro = (soc_insight.get("intro") if isinstance(soc_insight, dict) else "") or ""
                    _analy = (soc_insight.get("analysis") if isinstance(soc_insight, dict) else "") or ""
                    _es = _build_es(summary_text=_intro, soc_analysis=_analy, fallback_headline="")
                    if _es and (_es.get("headline") or _es.get("priority")):
                        self._last_turn_data["metadata"]["executive_summary"] = _es
            except Exception as _meta_exc:
                logger.warning(f"[NLP-S1] metadata enrichment failed: {_meta_exc}")

            # Pallas 4.1: Cache the response for identical follow-up queries.
            # Skip caching when (a) feature is disabled, (b) confidence is low,
            # or (c) the response looks like a tool error — replaying error
            # strings for 3 min hides the underlying problem from analysts.
            looks_like_error = isinstance(final, str) and (
                final.lstrip().startswith("Error:")
                or "'list' object has no attribute" in final
                or "Tool execution error" in final
            )
            if self._cache_enabled and plan.confidence >= 0.5 and not looks_like_error:
                # Evict oldest if at capacity
                if len(self._response_cache) >= self._cache_max_size:
                    oldest_key = min(self._response_cache, key=lambda k: self._response_cache[k]["timestamp"])
                    del self._response_cache[oldest_key]
                self._response_cache[cache_key] = {
                    "response": final,
                    "turn_data": self._last_turn_data,
                    "timestamp": time.time(),
                }

            # Live progress: done (final frame before the response is sent)
            await self._emit_progress(progress_cb, stage="done")

            return final

        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}", exc_info=True)
            return self._format_exception_response(e)

    async def _generate_analysis(
        self,
        plan: ToolPlan,
        correlated_data: CorrelationResult
    ) -> Optional[str]:
        """Generate LLM-powered analysis from correlated data"""

        # Prepare summary for LLM
        summary_lines = [
            f"Query confidence: {plan.confidence:.0%}",
            f"Selection method: {plan.selection_method}",
            f"Total items: {len(correlated_data.correlated_data)}",
            ""
        ]

        # Add correlation summary
        for key, value in correlated_data.summary.items():
            summary_lines.append(f"{key}: {value}")

        # Add sample data (top 5)
        summary_lines.append("\nSample data:")
        for i, item in enumerate(correlated_data.correlated_data[:5], 1):
            if "cve_id" in item:
                summary_lines.append(
                    f"{i}. {item.get('cve_id')} - Severity: {item.get('severity')} - "
                    f"Agent: {item.get('agent', {}).get('name', 'N/A')}"
                )
            elif "rule_description" in item:
                summary_lines.append(
                    f"{i}. Alert Level {item.get('level')} - {item.get('rule_description')[:50]}"
                )
            elif "name" in item:
                summary_lines.append(
                    f"{i}. Agent: {item.get('name')} - Status: {item.get('status')} - "
                    f"Risk: {item.get('risk_score', 'N/A')}"
                )

        data_summary = "\n".join(summary_lines)

        tool = plan.primary_tool.lower()

        if ("vulnerability" in tool) or ("vulnerabilities" in tool) or ("critical_vulnerabilities" in tool) or ("vulnerability_summary" in tool):
            analysis_type = "vulnerability"
        elif "alert" in plan.primary_tool.lower():
            analysis_type = "alert"
        elif "agent" in plan.primary_tool.lower():
            analysis_type = "agent"
        else:
            analysis_type = "general"

        try:
            analysis = await self.llm.analyze_security_data(
                data_summary=data_summary,
                analysis_type=analysis_type
            )
            return analysis
        except Exception as e:
            logger.warning(f"LLM analysis generation failed: {e}")
            return None

    async def _generate_primary_analysis(
        self,
        plan: ToolPlan,
        primary_result: ExecutionResult
    ) -> Optional[str]:
        """Generate SOC analysis from a single-tool result (no correlation needed).
        This ensures EVERY response gets a SOC analysis section."""

        # Extract text content from the primary result
        raw_text = self._extract_raw_text(primary_result.data, "")
        if not raw_text:
            return None

        # Truncate to first 800 chars for LLM input efficiency
        data_summary = raw_text[:800]
        if len(raw_text) > 800:
            data_summary += "\n... (truncated)"

        # Add context
        data_summary = (
            f"Tool: {plan.primary_tool}\n"
            f"Query confidence: {plan.confidence:.0%}\n"
            f"Selection method: {plan.selection_method}\n\n"
            f"Data:\n{data_summary}"
        )

        analysis_type = self._get_analysis_type(plan.primary_tool)

        try:
            analysis = await self.llm.analyze_security_data(
                data_summary=data_summary,
                analysis_type=analysis_type
            )
            return analysis
        except Exception as e:
            logger.warning(f"Primary LLM analysis failed: {e}")
            return None

    def _get_analysis_type(self, tool_name: str) -> str:
        """Map tool name to SOC analysis type for LLM prompting."""
        t = tool_name.lower()
        if "vulnerability" in t or "cve" in t:
            return "vulnerability"
        if "alert" in t or "event" in t:
            return "alert"
        if "agent" in t or "health" in t or "running" in t:
            return "agent"
        if "port" in t:
            return "port"
        if "process" in t:
            return "process"
        if "statistic" in t or "stats" in t or "remoted" in t:
            return "statistics"
        return "general"

    def _extract_key_ids(self, data: list, tool_name: str) -> List[str]:
        """Extract key identifiers from result data for conversation memory."""
        ids = []
        if not isinstance(data, list):
            return ids
        for item in data[:20]:
            if isinstance(item, dict):
                for id_field in ['id', 'agent_id', 'cve_id', 'rule_id', 'name']:
                    val = item.get(id_field)
                    if val and str(val) != '000':
                        ids.append(str(val))
                        break
        return ids

    def _infer_result_type(self, tool_name: str) -> str:
        """Map tool name to result type for conversation memory."""
        t = tool_name.lower()
        if "agent" in t or "health" in t or "running" in t:
            return "agents"
        if "vulnerability" in t or "cve" in t:
            return "vulnerabilities"
        if "alert" in t or "event" in t:
            return "alerts"
        if "port" in t:
            return "ports"
        if "process" in t:
            return "processes"
        if "fim" in t or "syscheck" in t:
            return "fim_events"
        return "results"

    async def _generate_soc_insight(self, plan, user_query: str, formatted_response: str, context: Optional[QueryContext] = None) -> Dict[str, Any]:
        """Generate query intro + SOC analysis from formatted response text."""
        # Use formatted response (clean, filtered) as data summary.
        # Scale limit with response complexity:
        #   - Multi-entity fan-out: each entity contributes ~1.2-1.5k of markdown;
        #     allow ~3k per entity so DeepSeek sees every section in full.
        #   - Correlation strategies: 2500 (multiple tools, denser sections).
        #   - Single tool: 1500 (default).
        # No truncation: DeepSeek's context window is large; cost is not a concern.
        # Cap at 200k chars purely as a runaway safeguard.
        _fanout = getattr(plan, "fanout_entities", None) or []
        summary_limit = 200_000
        data_summary = formatted_response[:summary_limit]
        if len(formatted_response) > summary_limit:
            # ================================================================
            # PALLAS_P4_TRUNCATION_SAFE_STOP (Track B7, fixes audit F8)
            # Plain truncation at a fixed char limit can leave a Markdown
            # table half-open ("| 2026-06-... | ..." with no closing pipe or
            # newline). BlockParser then bails on the broken row and falls
            # back to plain prose for the whole section. Walk back to the
            # last complete line, then close any open table by emitting an
            # empty row and a visible "truncated" note. Behaviour-flag for
            # rollback: PALLAS_P4_TRUNCATION_SAFE_STOP=false.
            # ================================================================
            if os.getenv("PALLAS_P4_TRUNCATION_SAFE_STOP", "true").lower() != "false":
                # Walk back to the most recent newline so we never cut mid-row.
                _last_nl = data_summary.rfind("\n")
                if _last_nl > 0:
                    data_summary = data_summary[:_last_nl]
                # Detect we ended inside a Markdown table by looking at the
                # last non-empty line.
                _tail_lines = data_summary.rstrip("\n").split("\n")
                _last_nonempty = next(
                    (ln for ln in reversed(_tail_lines) if ln.strip()), ""
                )
                _in_table = _last_nonempty.lstrip().startswith("|")
                _rows_kept = sum(
                    1 for ln in _tail_lines if ln.lstrip().startswith("|")
                )
                if _in_table:
                    # Emit a clean closing row so the table is well-formed.
                    data_summary += (
                        f"\n| ... | ... | ... | ... | ... |\n"
                    )
                data_summary += (
                    f"\n\n> **Note:** response truncated at ~{_rows_kept} rows due to "
                    f"the {summary_limit // 1000}k char token cap.\n"
                )
            else:
                data_summary += "\n... (truncated at 200k chars - extremely long response)"

        # NLP fan-out anti-hallucination: prepend a deterministic per-entity
        # TL;DR so DeepSeek cannot miss or invent counts even if the markdown
        # body is dense. Extracted from the rendered section headers.
        if _fanout:
            import re as _re
            _per = []
            for _entity in _fanout:
                _label = _entity.get("label") or ""
                # Search for the section under this label and grab its leading
                # "X findings (Y unique CVEs)" / "X items" line.
                _esc = _re.escape(_label)
                _m = _re.search(
                    rf"## {_esc}.*?\n## [^\n]*?(\d[\d,]*)[^\n]*?(?:findings|items|ports|processes|events|alerts)",
                    formatted_response, _re.DOTALL,
                )
                _count = _m.group(1) if _m else "(see section)"
                _per.append(f"- {_label}: {_count}")
            if _per:
                _prefix = (
                    "PER-ENTITY BREAKDOWN (authoritative — use these exact counts):\n"
                    + "\n".join(_per)
                    + "\n\n---\n\n"
                )
                data_summary = _prefix + data_summary

        # Determine analysis type — use strategy-specific type for multi-tool results
        if plan.requires_correlation and plan.correlation_strategy:
            analysis_type = f"correlation_{plan.correlation_strategy}"
        else:
            analysis_type = self._get_analysis_type(plan.primary_tool)

        conversation_context = ""
        if context and context.conversation_summary:
            conversation_context = context.conversation_summary

        try:
            return await self.llm.generate_soc_insight(
                user_query=user_query,
                data_summary=data_summary,
                analysis_type=analysis_type,
                conversation_context=conversation_context
            )
        except Exception as e:
            logger.warning(f"SOC insight generation failed: {e}")
            return {"intro": None, "analysis": self._static_soc_fallback(plan.primary_tool)}

    def _build_query_header(self, plan, user_query, primary_result, correlated_data):
        """Build deterministic header showing what was asked and what was found."""
        header = f"**Query:** {user_query}\n\n"

        # What tool(s) were used
        tools_used = [plan.primary_tool]
        if plan.secondary_tools:
            tools_used.extend(plan.secondary_tools)
        tool_names = ", ".join(f"`{t}`" for t in tools_used)
        header += f"**Tools Used:** {tool_names}\n"

        # ------------------------------------------------------------------
        # Inline helper: pull TRUE TOTAL (hits.total.value) out of a raw tool
        # payload. The normalizer drops total_affected_items; this helper
        # walks common shapes (Wazuh, Suricata, MCP-wrapped) to recover it.
        # ------------------------------------------------------------------
        def _peek_total(p):
            try:
                if isinstance(p, dict):
                    for keys in (
                        ("data", "total_affected_items"),
                        ("total_affected_items",),
                        ("hits", "total", "value"),
                    ):
                        cur = p; ok = True
                        for k in keys:
                            if isinstance(cur, dict) and k in cur:
                                cur = cur[k]
                            else:
                                ok = False; break
                        if ok and isinstance(cur, int) and cur >= 0:
                            return cur
                    if "content" in p and isinstance(p.get("content"), list) and p["content"]:
                        import json as _j
                        txt = p["content"][0].get("text", "")
                        if isinstance(txt, str) and txt.strip().startswith("{"):
                            try:
                                return _peek_total(_j.loads(txt))
                            except Exception:
                                return None
            except Exception:
                return None
            return None

        # What was found (count)
        if correlated_data and hasattr(correlated_data, 'correlated_data') and correlated_data.correlated_data:
            count = len(correlated_data.correlated_data)
            # Surface PRIMARY tool's true total alongside correlated record count
            # so "10,000 correlated records" no longer reads as "we only have 10K alerts".
            primary_true_total = _peek_total(primary_result.data) if primary_result else None
            if isinstance(primary_true_total, int) and primary_true_total > count:
                header += f"**Results:** {count:,} correlated records (from {primary_true_total:,} total {plan.primary_tool.replace('get_','').replace('_',' ')})"
            else:
                header += f"**Results:** {count:,} correlated records"
            if hasattr(correlated_data, 'correlation_type') and correlated_data.correlation_type:
                header += f" (Strategy: {correlated_data.correlation_type})"
            header += "\n"
        else:
            try:
                normalized = self._normalize_tool_payload(primary_result.data)
                items = normalized.get("affected_items", [])
                analyzed = len(items)

                # Pull true total from RAW payload (the normalizer drops it).
                # Indexer sets data.total_affected_items from hits.total.value.
                def _peek_total(p):
                    try:
                        if isinstance(p, dict):
                            for keys in (
                                ("data", "total_affected_items"),
                                ("total_affected_items",),
                                ("hits", "total", "value"),
                            ):
                                cur = p; ok = True
                                for k in keys:
                                    if isinstance(cur, dict) and k in cur:
                                        cur = cur[k]
                                    else:
                                        ok = False; break
                                if ok and isinstance(cur, int) and cur >= 0:
                                    return cur
                            # MCP wrapper: parse content JSON if present
                            if "content" in p and isinstance(p["content"], list) and p["content"]:
                                import json as _j
                                txt = p["content"][0].get("text", "")
                                if isinstance(txt, str) and txt.strip().startswith("{"):
                                    try:
                                        inner = _j.loads(txt)
                                        return _peek_total(inner)
                                    except Exception:
                                        return None
                    except Exception:
                        return None
                    return None

                true_total = _peek_total(primary_result.data)
                if true_total is None:
                    true_total = analyzed
                if analyzed > 0 or true_total > 0:
                    if true_total > analyzed:
                        header += f"**Results:** {true_total:,} total records (analyzing top {analyzed:,})\n"
                    else:
                        header += f"**Results:** {true_total:,} records found\n"
            except Exception:
                pass

        header += "\n---\n\n"
        return header

    def _build_suggested_queries(self, plan, correlated_data=None):
        """Suggest follow-up queries based on what was just shown. Only tested, working queries."""
        suggestions = []
        tool = plan.primary_tool
        strategy = plan.correlation_strategy

        # Pallas 3.2: Correlation-strategy-based suggestions (highest priority)
        STRATEGY_SUGGESTIONS = {
            # --- Pallas 3.2 originals ---
            "ip_investigation_pivot": ["show attack chain", "detect suspicious activity", "show active agents"],
            "full_agent_investigation": ["most at risk agents", "show critical vulnerabilities", "show attack chain"],
            "cross_platform_ip_correlation": ["detect suspicious activity", "suricata top attackers", "show active agents"],
            "unified_scanning_detection": ["suricata http analysis", "show attack chain", "suricata top attackers"],
            "attack_chain_analysis": ["detection coverage gaps", "suricata alert summary", "most at risk agents"],
            "unified_threat_summary": ["show attack chain", "most at risk agents", "show critical vulnerabilities"],
            "vulnerability_exploit_correlation": ["most at risk agents", "suricata alert summary", "show critical vulnerabilities"],
            "detection_coverage_gap": ["show attack chain", "suricata mitre mapping", "show mitre coverage"],
            "fim_alert_correlation": ["show alerts for last 24 hours", "show active agents", "show critical vulnerabilities"],
            "port_exposure_risk": ["show critical vulnerabilities", "suricata alert summary", "detect suspicious activity"],
            "top_risk_agents_composite": ["show critical vulnerabilities", "show attack chain", "detect suspicious activity"],
            "alert_context_enrichment": ["most at risk agents", "show critical vulnerabilities", "detect suspicious activity"],
            # --- Pallas 3.3: Wazuh correlations ---
            "vulnerability_with_agents": ["most at risk agents", "show critical vulnerabilities", "show disconnected agents"],
            "agents_with_vulnerabilities": ["show critical vulnerabilities", "vulnerability trend by severity", "check agent health"],
            "disconnected_agents_with_critical_vulns": ["show all disconnected agents", "show critical vulnerabilities", "show agent health"],
            "top_agents_by_vuln_count": ["show critical vulnerabilities", "most at risk agents", "show active agents"],
            "active_agents_vulns_over_threshold": ["vulnerability trend by severity", "most at risk agents", "show critical vulnerabilities"],
            "active_agents_with_high_vulns": ["show critical vulnerabilities", "show alerts", "most at risk agents"],
            "compare_vulns_active_vs_disconnected": ["show disconnected agents with critical vulns", "show active agents", "vulnerability summary"],
            "alerts_with_agents": ["show alert patterns", "show critical vulnerabilities", "most at risk agents"],
            "alerts_with_vulnerabilities": ["most at risk agents", "show attack chain", "show critical vulnerabilities"],
            "rule_mitre_with_agents": ["show attack chain", "detection coverage gaps", "show mitre coverage"],
            "fim_with_agent_posture": ["show alerts for last 24 hours", "show active agents", "show critical vulnerabilities"],
            "vulnerability_trend_by_severity": ["show critical vulnerabilities", "most at risk agents", "vulnerability summary"],
            # --- Pallas 3.3: Suricata correlations ---
            "suricata_attacker_target_map": ["suricata top signatures", "show flow analysis", "detect suspicious activity"],
            "suricata_signature_severity_analysis": ["suricata critical alerts", "suricata category breakdown", "show attack chain"],
            "suricata_network_threat_profile": ["suricata top attackers", "show flow analysis", "detect suspicious activity"],
            "suricata_http_alert_context": ["suricata http analysis", "detect suspicious activity", "show attack chain"],
            "suricata_tls_threat_detection": ["show JA4 analysis", "show JA3 fingerprints", "suricata critical alerts"],
            "suricata_recon_detection": ["suricata top attackers", "suricata http analysis", "show attack chain"],
            # --- Pallas 3.3: Cross-platform correlations ---
            "combined_alert_view": ["show attack chain", "most at risk agents", "detect suspicious activity"],
            "wazuh_agents_suricata_detections": ["suricata top attackers", "most at risk agents", "show active agents"],
            "ioc_with_suricata_context": ["suricata top signatures", "show alerts", "detect suspicious activity"],
            "comprehensive_security_posture": ["most at risk agents", "show critical vulnerabilities", "show attack chain"],
            "wazuh_suricata_mitre_unified": ["detection coverage gaps", "show attack chain", "show critical vulnerabilities"],
            # --- Pallas 3.4: Agent health suggestions ---
            "agent_health_overview": ["show active agents", "show disconnected agents", "show log source health"],
            "agent_event_volume": ["show agent health overview", "check agent health", "show log source health"],
            # --- Pallas 3.5: Advanced SOC suggestions ---
            "temporal_attack_sequence": ["calculate risk score", "show MITRE coverage", "show recent alerts"],
            "dynamic_risk_scoring": ["show attack sequence", "show critical vulnerabilities", "show agent health overview"],
            "cross_platform_threat_correlation": ["show attack sequence", "calculate risk score", "show suricata alerts"],
            "data_exfiltration_detection": ["show FIM events", "show attack sequence", "show processes"],
            "off_hours_anomaly_detection": ["show attack sequence", "detect data exfiltration", "show recent alerts"],
            "alert_noise_analysis": ["show rule trigger analysis", "show alert timeline", "show recent alerts"],
            "multi_asset_campaign_detection": ["show attack sequence", "calculate risk score", "show MITRE coverage"],
        }
        if strategy in STRATEGY_SUGGESTIONS:
            suggestions = STRATEGY_SUGGESTIONS[strategy][:]
            # Short-circuit: strategy-based suggestions are definitive
            seen = set()
            unique = []
            for s in suggestions:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            suggestions = unique[:3]
            lines = ["\n---\n\n**Suggested Follow-up Queries:**"]
            for s in suggestions:
                lines.append(f"- `{s}`")
            return "\n".join(lines) + "\n"

        # Suricata tool suggestions
        if "suricata" in tool:
            if tool == "get_suricata_alerts":
                suggestions.append("suricata alert summary")
                suggestions.append("suricata network analysis")
                suggestions.append("top suricata signatures")
            elif tool == "get_suricata_alert_summary":
                suggestions.append("show critical suricata alerts")
                suggestions.append("suricata network analysis")
                suggestions.append("show alerts for last 24 hours")
            elif tool == "get_suricata_network_analysis":
                suggestions.append("top suricata signatures")
                suggestions.append("suricata alert summary")
                suggestions.append("show active agents")
            elif tool == "get_suricata_top_signatures":
                suggestions.append("suricata alert summary")
                suggestions.append("suricata network analysis")
                suggestions.append("show critical suricata alerts")
            elif tool == "get_suricata_top_attackers":
                suggestions.append("suricata network analysis")
                suggestions.append("suricata alert summary")
                suggestions.append("show active agents")
            elif tool == "get_suricata_health":
                suggestions.append("show suricata alerts")
                suggestions.append("suricata alert summary")
                suggestions.append("show active agents")
            elif tool == "get_suricata_http_analysis":
                suggestions.append("search http for suspicious URLs")
                suggestions.append("suricata tls analysis")
                suggestions.append("detect suspicious activity")
            elif tool == "search_suricata_http":
                suggestions.append("suricata http analysis")
                suggestions.append("suricata alert summary")
                suggestions.append("detect suspicious activity")
            elif tool == "get_suricata_tls_analysis":
                suggestions.append("ja3 fingerprint analysis")
                suggestions.append("suricata http analysis")
                suggestions.append("detect suspicious activity")
            elif tool == "get_suricata_mitre_mapping":
                suggestions.append("show critical suricata alerts")
                suggestions.append("suricata alert summary")
                suggestions.append("suricata http analysis")
            elif tool == "get_suricata_ja3_fingerprints":
                suggestions.append("suricata tls analysis")
                suggestions.append("detect suspicious activity")
                suggestions.append("suricata network analysis")
            elif tool == "get_suricata_suspicious_activity":
                suggestions.append("suricata http analysis")
                suggestions.append("suricata tls analysis")
                suggestions.append("top suricata signatures")
            elif tool == "get_suricata_traffic_overview":
                suggestions.append("suricata alert summary")
                suggestions.append("suricata http analysis")
                suggestions.append("suricata network analysis")
            elif tool == "get_suricata_ja4_analysis":
                suggestions.append("suricata tls analysis")
                suggestions.append("show JA3 fingerprints")
                suggestions.append("detect suspicious activity")
            elif tool == "get_suricata_flow_analysis":
                suggestions.append("suricata top attackers")
                suggestions.append("suricata network analysis")
                suggestions.append("show attacker target map")
            else:
                suggestions.append("suricata alert summary")
                suggestions.append("show suricata alerts")
                suggestions.append("show active agents")

        # Agent-related suggestions
        elif "agent" in tool:
            if plan.primary_args.get("agent_id"):
                aid = plan.primary_args["agent_id"]
                suggestions.append(f"show processes for agent {aid}")
                suggestions.append(f"show ports for agent {aid}")
                suggestions.append(f"show vulnerabilities for agent {aid}")
            elif plan.primary_args.get("status") == "active":
                suggestions.append("show disconnected agents")
                suggestions.append("show critical vulnerabilities")
                if self.suricata_enabled:
                    suggestions.append("show suricata alerts")
                else:
                    suggestions.append("show alerts for last 24 hours")
            elif plan.primary_args.get("status") == "disconnected":
                suggestions.append("show active agents")
                suggestions.append("show critical vulnerabilities")
                suggestions.append("generate security report")
            else:
                suggestions.append("show active agents")
                suggestions.append("show disconnected agents")
                suggestions.append("show critical vulnerabilities")

        # Vulnerability suggestions
        elif "vulnerabilit" in tool:
            suggestions.append("show active agents")
            suggestions.append("show alerts for last 24 hours")
            if self.suricata_enabled:
                suggestions.append("suricata alert summary")
            else:
                suggestions.append("generate security report")

        # Alert suggestions (Wazuh)
        elif "alert" in tool:
            suggestions.append("show critical vulnerabilities")
            suggestions.append("show active agents")
            if self.suricata_enabled:
                suggestions.append("show suricata alerts")
            else:
                suggestions.append("generate security report")

        # IOC/threat suggestions
        elif tool in ("check_ioc_reputation", "analyze_security_threat", "get_top_security_threats"):
            suggestions.append("show alerts for last 24 hours")
            suggestions.append("show active agents")
            if self.suricata_enabled:
                suggestions.append("show suricata alerts")
            else:
                suggestions.append("generate security report")

        # Port/process follow-ups
        elif "port" in tool:
            if plan.primary_args.get("agent_id"):
                aid = plan.primary_args["agent_id"]
                suggestions.append(f"show processes for agent {aid}")
                suggestions.append(f"full security posture for agent {aid}")
            suggestions.append("show active agents")

        elif "process" in tool:
            if plan.primary_args.get("agent_id"):
                aid = plan.primary_args["agent_id"]
                suggestions.append(f"show ports for agent {aid}")
                suggestions.append(f"full security posture for agent {aid}")
            suggestions.append("show active agents")

        # Stats/health/report follow-ups
        elif "stats" in tool or "statistic" in tool or "health" in tool:
            suggestions.append("show active agents")
            suggestions.append("show alerts for last 24 hours")
            suggestions.append("show critical vulnerabilities")

        # Decoder follow-ups
        elif "decoder" in tool:
            suggestions.append("show alerts for last 24 hours")
            suggestions.append("show active agents")

        # Default fallback — always show something
        if not suggestions:
            suggestions = [
                "show active agents",
                "show alerts for last 24 hours",
                "show critical vulnerabilities",
            ]

        # Limit to 3 unique suggestions
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        suggestions = unique[:3]

        lines = ["\n---\n\n**Suggested Follow-up Queries:**"]
        for s in suggestions:
            lines.append(f"- `{s}`")
        return "\n".join(lines) + "\n"

    def _get_suggestion_list(self, plan, correlated_data=None):
        """Return suggestion list as plain strings (for v2 JSON response, no markdown)."""
        strategy = plan.correlation_strategy
        tool = plan.primary_tool
        STRATEGY_SUGGESTIONS = {
            "ip_investigation_pivot": ["show attack chain", "detect suspicious activity", "show active agents"],
            "full_agent_investigation": ["most at risk agents", "show critical vulnerabilities", "show attack chain"],
            "temporal_attack_sequence": ["calculate risk score", "show MITRE coverage", "show recent alerts"],
            "dynamic_risk_scoring": ["show attack sequence", "show critical vulnerabilities", "show agent health overview"],
            "cross_platform_threat_correlation": ["show attack sequence", "calculate risk score", "show suricata alerts"],
            "data_exfiltration_detection": ["show FIM events", "show attack sequence", "show processes"],
            "off_hours_anomaly_detection": ["show attack sequence", "detect data exfiltration", "show recent alerts"],
            "alert_noise_analysis": ["show rule trigger analysis", "show alert timeline", "show recent alerts"],
            "multi_asset_campaign_detection": ["show attack sequence", "calculate risk score", "show MITRE coverage"],
            "comprehensive_security_posture": ["most at risk agents", "show critical vulnerabilities", "show attack chain"],
            "combined_alert_view": ["show attack chain", "most at risk agents", "detect suspicious activity"],
        }
        if strategy and strategy in STRATEGY_SUGGESTIONS:
            return STRATEGY_SUGGESTIONS[strategy][:4]
        # Fallback based on tool
        if "agent" in tool:
            return ["show active agents", "show critical vulnerabilities", "show alerts for last 24 hours"]
        if "alert" in tool:
            return ["show critical vulnerabilities", "show active agents", "show attack chain"]
        if "vulnerabilit" in tool:
            return ["show active agents", "show alerts for last 24 hours", "most at risk agents"]
        if "suricata" in tool:
            return ["suricata alert summary", "suricata network analysis", "show attack chain"]
        return ["show active agents", "show alerts for last 24 hours", "show critical vulnerabilities"]

    def _static_soc_fallback(self, tool_name: str) -> str:
        """Static SOC analysis stub when LLM is completely unavailable."""
        return (
            "**Risk Summary:** Automated analysis temporarily unavailable. "
            "Manual review of the data above is required.\n\n"
            "**Observations:**\n"
            "- LLM analysis service is unreachable\n"
            "- Data has been formatted but not interpreted\n"
            "- Manual SOC review is required\n\n"
            "**Potential Threats:**\n"
            "- Cannot be automatically assessed at this time\n"
            "- Review data above using standard SOC procedures\n\n"
            "**Recommended Actions:**\n"
            f"1. Review the formatted data above for {tool_name}\n"
            "2. Verify LLM API key is configured: check LLM_API_KEY env var\n"
            "3. Check LLM connectivity: verify LLM_BASE_URL is reachable\n"
            "4. Escalate if analysis remains unavailable\n\n"
            "**Priority Level:** MEDIUM (analysis gap)"
        )

    # ================================================================
    # PALLAS_P4_EMPTY_PRIMARY_FALLBACK helper (Track B3)
    # ================================================================
    def _pick_secondary_promotion(
        self, secondary_results: List["ExecutionResult"]
    ) -> Optional[Tuple[str, List[Dict[str, Any]], Any]]:
        """If any secondary returned real data, return the most-populated one.

        Returns (tool_name, normalized_items, raw_response) or None.
        """
        if not secondary_results:
            return None
        best = None
        best_count = 0
        for _sr in secondary_results:
            if not getattr(_sr, "success", False):
                continue
            try:
                norm = self._normalize_tool_payload(_sr.data) or {}
                items = norm.get("affected_items") or []
                if not isinstance(items, list):
                    continue
                if len(items) > best_count:
                    best_count = len(items)
                    best = (_sr.tool_name, items, _sr.data)
            except Exception as e:
                logger.debug(
                    f"[B3] Could not normalize secondary {getattr(_sr, 'tool_name', '?')}: {e}"
                )
        return best

    def _format_response(
        self,
        plan: ToolPlan,
        primary_result: ExecutionResult,
        secondary_results: List[ExecutionResult],
        correlated_data: Optional[CorrelationResult],
        llm_analysis: Optional[str],
        user_query: str = ""
    ) -> str:
        """
        Format final response - FIXED with:
        - Exact tool name matching (no substring routing mistakes)
        - Safe fallback mode if formatter fails
        - Debug logs at each routing decision
        - Correlation empty fallback to primary data
        - PALLAS_P4_EMPTY_PRIMARY_FALLBACK (B3): promote secondary when primary empty
        """
        # B3 footer for the promoted-secondary case; cleared at the start of every call.
        self._b3_footer = ""

        tool_response = primary_result.data
        tool_name = plan.primary_tool

        logger.info(f"[FORMATTER] Routing for tool: {tool_name}")

        # ================================================================
        # FIX: Normalize payload using centralized normalizer
        # ================================================================
        try:
            normalized = self._normalize_tool_payload(tool_response)
            data = normalized.get("affected_items", [])
            logger.info(f"[FORMATTER] Normalized data: {len(data)} items from {tool_name} (source: {normalized.get('_source', 'unknown')})")
        except Exception as e:
            logger.error(f"[FORMATTER] Normalization failed: {e}")
            data = []

        summary = {"total_items": len(data), "tool": primary_result.tool_name}

        # Surface true total from raw payload (track_total_hits via total_affected_items)
        # so formatters can show "Total: X (analyzing top Y)" instead of len(items).
        try:
            _raw = tool_response
            _tt = None
            if isinstance(_raw, dict):
                _d = _raw.get("data") if isinstance(_raw.get("data"), dict) else None
                if _d and isinstance(_d.get("total_affected_items"), int):
                    _tt = _d["total_affected_items"]
                elif isinstance(_raw.get("total_affected_items"), int):
                    _tt = _raw["total_affected_items"]
                elif "content" in _raw and isinstance(_raw.get("content"), list) and _raw["content"]:
                    try:
                        _txt = _raw["content"][0].get("text", "")
                        if isinstance(_txt, str) and _txt.strip().startswith("{"):
                            _inner = json.loads(_txt)
                            if isinstance(_inner.get("data"), dict):
                                _tt = _inner["data"].get("total_affected_items")
                            elif isinstance(_inner.get("total_affected_items"), int):
                                _tt = _inner["total_affected_items"]
                    except Exception:
                        pass
            if isinstance(_tt, int) and _tt > 0:
                summary["total_affected_items"] = _tt
                logger.info(f"[FORMATTER] true total surfaced into summary: {_tt:,} (sample={len(data)})")
        except Exception as _e:
            logger.debug(f"[FORMATTER] could not surface true total: {_e}")

        # ================================================================
        # E1.4 (2026-06-11): Phase 2 / 2c / 2d / 2e moved verbatim to
        # orchestrator/correlation/data_union.py. Same env flags
        # (PALLAS_P3_MULTISOURCE_ALERTS, PALLAS_P4_MULTISOURCE_EXPANDED,
        # PALLAS_P4_CROSS_DOMAIN_UNION), same log line prefixes, same
        # Markdown shape. Helpers take normalize_payload as a callable so
        # the new module stays leaf-level for tests / quick import.
        # ================================================================
        from .orchestrator.correlation.data_union import (
            apply_same_domain_union, render_multisource_alerts,
            render_multisource_expanded, render_cross_domain_union,
        )
        apply_same_domain_union(plan, data, summary, secondary_results, self._normalize_tool_payload)
        _md = render_multisource_alerts(plan, data, secondary_results, self._normalize_tool_payload)
        if _md is not None:
            return _md
        _md = render_multisource_expanded(plan, data, secondary_results, self._normalize_tool_payload)
        if _md is not None:
            return _md
        _md = render_cross_domain_union(plan, data, secondary_results, self._normalize_tool_payload)
        if _md is not None:
            return _md


        # Override with correlation ONLY if it has data (NEVER return empty from correlation)
        # PALLAS_P4_GRACEFUL_CORR_FAIL (Track B2, fixes audit F3) - when the
        # correlation strategy returns empty, clear both the strategy AND
        # remember the fact so a footer note can be appended downstream. This
        # keeps the user informed instead of silently showing primary data
        # under a misleading correlation header.
        self._b2_correlation_skipped: Optional[str] = None
        if correlated_data and hasattr(correlated_data, "correlated_data"):
            if correlated_data.correlated_data and len(correlated_data.correlated_data) > 0:
                logger.info(f"[FORMATTER] Using correlation: {len(correlated_data.correlated_data)} items")
                data = correlated_data.correlated_data
                summary = correlated_data.summary
            else:
                _skipped_strategy = plan.correlation_strategy
                logger.info(f"[FORMATTER] Correlation empty -> fallback to primary data: {len(data)} items")
                # Clear correlation strategy so _route_formatter uses tool-based routing
                # instead of sending raw primary data to a correlation formatter
                plan.correlation_strategy = None
                if _skipped_strategy and os.getenv("PALLAS_P4_GRACEFUL_CORR_FAIL", "true").lower() != "false":
                    self._b2_correlation_skipped = _skipped_strategy

        logger.info(f"[FORMATTER] Final data count before formatting: {len(data)} items")

        # Graceful empty data handling
        if not data or len(data) == 0:
            # Some tools return text content instead of affected_items (stats, logs, decoder)
            # Check if there's raw text content before declaring empty
            raw_text = self._extract_raw_text(tool_response, "")
            if not raw_text or len(raw_text.strip()) < 10:
                # ================================================================
                # PALLAS_P4_EMPTY_PRIMARY_FALLBACK (Track B3, fixes audit F2)
                # Before declaring "No Results Found", check if any secondary
                # tool returned real data. If so, promote the most-populated
                # secondary's output for formatting and append a footer note.
                # Eliminates the worst silent-data-loss bug.
                # ================================================================
                if os.getenv("PALLAS_P4_EMPTY_PRIMARY_FALLBACK", "true").lower() != "false":
                    _promotion = self._pick_secondary_promotion(secondary_results)
                    if _promotion is not None:
                        _promoted_tool, _promoted_items, _promoted_response = _promotion
                        logger.warning(
                            f"[FORMATTER] PALLAS_P4_EMPTY_PRIMARY_FALLBACK: primary={tool_name} "
                            f"returned empty; promoting secondary={_promoted_tool} with "
                            f"{len(_promoted_items)} items"
                        )
                        # Rebind formatting inputs to the promoted secondary; the
                        # rest of the formatter path will format it normally.
                        tool_name = _promoted_tool
                        tool_response = _promoted_response
                        data = _promoted_items
                        # Attach a footer note that will be appended after formatting.
                        self._b3_footer = (
                            f"\n\n> **Note:** primary tool `{plan.primary_tool}` returned no "
                            f"results, so this view shows data from secondary tool "
                            f"`{_promoted_tool}` instead.\n"
                        )
                    else:
                        # Build context-aware suggestions based on what tool was used
                        tool_suggestions = {
                            "get_wazuh_agents": ["show all agents", "show active agents", "show disconnected agents"],
                            "get_wazuh_alerts": ["show recent alerts", "show alerts from last 7 days", "show critical alerts"],
                            "search_security_events": ["search brute force", "search T1110", "search rule 80255", "search credential access"],
                            "get_vulnerability_summary": ["show all vulnerabilities", "show critical vulnerabilities", "show vulnerability summary"],
                            "get_suricata_alerts": ["show suricata alerts", "show suricata alerts severity 1", "show top attack signatures"],
                        }
                        suggestions = tool_suggestions.get(tool_name, ["show active agents", "show recent alerts", "show vulnerabilities"])
                        suggestions_text = "\n".join([f"  - `{s}`" for s in suggestions])

                        return (
                            f"## No Results Found\n\n"
                            f"Your query was understood and the right tool was selected, but no matching data was returned.\n\n"
                            f"**Tool used:** `{tool_name}`\n"
                            f"**What was searched:** {plan.reasoning}\n\n"
                            f"### Why This Might Happen\n\n"
                            f"- The specified time range may not contain any matching events\n"
                            f"- The filters (severity, agent ID, rule ID) may be too restrictive\n"
                            f"- The data source may not have ingested events for this category yet\n\n"
                            f"### Try These Instead\n\n"
                            f"{suggestions_text}\n\n"
                            f"### Tips\n\n"
                            f"- **Broaden the time range** — try \"last 7 days\" instead of \"last 24 hours\"\n"
                            f"- **Remove filters** — try without specifying severity or agent ID\n"
                            f"- **Check connectivity** — run `show active agents` to verify agents are reporting\n"
                        )

        # Build correlation header ONLY when there's no dedicated formatter
        # Dedicated correlation formatters already include their own ## header
        correlation_header = ""
        if correlated_data and hasattr(correlated_data, "correlation_type"):
            if not plan.correlation_strategy:
                correlation_header = self._build_correlation_header(
                    correlated_data.correlation_type, summary
                )

        try:
            response = self._route_formatter(
                tool_name=tool_name,
                tool_response=tool_response,
                data=data,
                summary=summary,
                plan=plan,
                normalized=normalized
            )
        except Exception as fmt_err:
            logger.error(f"[FORMATTER] ❌ Formatter failed for {tool_name}: {fmt_err}", exc_info=True)
            # SAFE FALLBACK MODE: return raw CLI-style output
            response = self._safe_fallback_format(tool_name, tool_response, data, fmt_err)

        # Prepend correlation header if this was a correlated multi-tool response
        if correlation_header:
            response = correlation_header + "\n" + response

        # ================================================================
        # PALLAS_P4 footers (Tracks B2, B3) - append diagnostic context
        # without altering already-rendered data above.
        # ================================================================
        try:
            _b3 = getattr(self, "_b3_footer", "") or ""
            if _b3:
                response = response + _b3
            _b2_skipped = getattr(self, "_b2_correlation_skipped", None)
            if _b2_skipped:
                response = response + (
                    f"\n\n> **Note:** correlation step `{_b2_skipped}` was skipped "
                    f"(returned no joined results); showing primary tool data instead.\n"
                )
        except Exception:
            pass  # never let footer logic break the response

        # LLM analysis and metadata footer are now assembled in process_query()
        return response

    def _extract_raw_text(self, tool_response: Dict[str, Any], fallback: str = "⚠️ No data") -> str:
        """
        Universal helper: extract the inner text from an MCP content-wrapped response.
        Handles:
          {"content": [{"type": "text", "text": "..."}]}  → returns the text
          {"data": {...}}                                  → returns json.dumps
          anything else                                    → returns fallback
        """
        if isinstance(tool_response, dict):
            if "content" in tool_response and tool_response["content"]:
                content = tool_response["content"]
                if isinstance(content, list) and content:
                    text = content[0].get("text", "")
                    if text:
                        return text
            if "data" in tool_response:
                try:
                    return json.dumps(tool_response["data"], indent=2, default=str)
                except Exception:
                    pass
        return fallback

    def _route_formatter(
        self,
        tool_name: str,
        tool_response: Dict[str, Any],
        data: List[Dict[str, Any]],
        summary: Dict[str, Any],
        plan: ToolPlan,
        normalized: Dict[str, Any]
    ) -> str:
        """
        Route to the correct formatter based on EXACT tool name match.
        ALL 29 tools are explicitly handled — no tool falls into the generic else.
        Correlation-specific routing takes priority over tool_name routing.
        """

        # ----------------------------------------------------------------
        # PRIORITY: Correlation-specific routing (before tool_name routing)
        # ----------------------------------------------------------------
        if plan.correlation_strategy == "agent_posture_deep_dive" and data:
            logger.info("[FORMATTER] -> AGENT POSTURE DEEP DIVE formatter")
            return self._format_agent_posture_deep_dive(data, summary)

        # E1.3b (2026-06-11): the 36-row correlation_strategy -> formatter
        # method map and its dispatch helper live in
        # orchestrator/formatting/dispatcher.py now. Same lookup, same
        # log line format, same call signature into the formatter.
        from .orchestrator.formatting.dispatcher import try_correlation_formatter
        _tr_corr = (plan.primary_args or {}).get("time_range", "24h")
        _corr_md = try_correlation_formatter(
            plan.correlation_strategy, self.formatter, data, summary, _tr_corr,
        )
        if _corr_md is not None:
            return _corr_md

        # ----------------------------------------------------------------
        # GROUP A: Tools whose server.py already formats a nice text output.
        # Use _extract_raw_text() — NEVER json.dumps the MCP wrapper.
        # ----------------------------------------------------------------

        # Vulnerability summary — use dedicated formatter
        if tool_name == "get_wazuh_vulnerability_summary":
            logger.info(f"[FORMATTER] → VULNERABILITY SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No vulnerability summary data")
            return self.formatter.format_vulnerability_summary_response(raw_text)

        # Cluster tools
        elif tool_name == "get_wazuh_cluster_health":
            logger.info(f"[FORMATTER] → CLUSTER HEALTH - formatting")
            return self._format_cluster_health(tool_response)

        elif tool_name == "get_wazuh_cluster_nodes":
            logger.info(f"[FORMATTER] → CLUSTER NODES raw")
            return self._extract_raw_text(tool_response, "⚠️ No cluster node data")

        # Statistics / metrics tools — format nicely, not raw JSON dump
        elif tool_name == "get_wazuh_remoted_stats":
            logger.info(f"[FORMATTER] → REMOTED STATS - formatting")
            return self._format_remoted_stats(tool_response)

        elif tool_name == "get_wazuh_log_collector_stats":
            logger.info(f"[FORMATTER] → LOG COLLECTOR STATS - formatting")
            return self._format_generic_stats(tool_response, "📋 Log Collector Statistics")

        elif tool_name == "get_wazuh_weekly_stats":
            logger.info(f"[FORMATTER] → WEEKLY STATS - formatting")
            return self._format_generic_stats(tool_response, "📊 Weekly Statistics")

        elif tool_name == "get_wazuh_statistics":
            logger.info(f"[FORMATTER] → WAZUH STATISTICS - formatting")
            return self._format_generic_stats(tool_response, "📊 Wazuh System Statistics")

        elif tool_name == "get_wazuh_rules_summary":
            logger.info(f"[FORMATTER] → RULES SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No rules summary data")
            return self.formatter.format_rules_summary_response(raw_text)

        # Manager log tools
        elif tool_name == "get_wazuh_manager_error_logs":
            logger.info(f"[FORMATTER] → MANAGER ERROR LOGS raw")
            return self._extract_raw_text(tool_response, "⚠️ No error log data")

        elif tool_name == "search_wazuh_manager_logs":
            logger.info(f"[FORMATTER] → MANAGER LOGS SEARCH raw")
            return self._extract_raw_text(tool_response, "⚠️ No log data")

        # Connectivity / validation tool
        elif tool_name == "validate_wazuh_connection":
            logger.info(f"[FORMATTER] → VALIDATE CONNECTION raw")
            return self._extract_raw_text(tool_response, "⚠️ No connection status data")

        # Compliance check — use dedicated formatter
        elif tool_name == "run_compliance_check":
            logger.info(f"[FORMATTER] → COMPLIANCE CHECK formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No compliance data")
            return self.formatter.format_compliance_response(raw_text)

        # Threat analysis — dedicated formatter
        elif tool_name == "analyze_security_threat":
            logger.info(f"[FORMATTER] → THREAT ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No threat data")
            return self.formatter.format_threat_analysis_response(raw_text)

        # IoC reputation — dedicated formatter
        elif tool_name == "check_ioc_reputation":
            logger.info(f"[FORMATTER] → IOC REPUTATION formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No IoC data")
            return self.formatter.format_ioc_reputation_response(raw_text)

        # Security report — dedicated formatter
        elif tool_name == "generate_security_report":
            logger.info(f"[FORMATTER] → SECURITY REPORT formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No report data")
            return self.formatter.format_security_report_response(raw_text)

        # Remaining security analysis tools
        elif tool_name in ["get_top_security_threats", "perform_risk_assessment"]:
            logger.info(f"[FORMATTER] → SECURITY TOOL raw: {tool_name}")
            return self._extract_raw_text(tool_response, "⚠️ No data")

        # ----------------------------------------------------------------
        # GROUP B: Tools with structured formatters in ResponseFormatter
        # ----------------------------------------------------------------

        # Detailed vulnerability tools
        elif tool_name in ["get_wazuh_vulnerabilities", "get_wazuh_critical_vulnerabilities"]:
            logger.info(f"[FORMATTER] → VULNERABILITY formatter")
            # Pass user intent (severity, agent_id, os) so the formatter renders
            # the right view AND defensively filters CVEs that don't match.
            severity_filter = plan.primary_args.get("severity") if plan and plan.primary_args else None
            agent_id_filter = plan.primary_args.get("agent_id") if plan and plan.primary_args else None
            os_filter = plan.primary_args.get("os") if plan and plan.primary_args else None

            # ----------------------------------------------------------------
            # Phase 3.5: Multi-severity render
            # When plan has secondary vuln calls with different severities,
            # render one section per severity (the formatter accepts a
            # `severity` filter so each call returns just that bucket).
            # ----------------------------------------------------------------
            _VULN_TOOLS_RENDER = ("get_wazuh_vulnerabilities", "get_wazuh_critical_vulnerabilities")
            _vsevs = []
            _seen_v = set()
            def _add_sev(_s):
                if not _s: return
                _norm = str(_s).strip().lower()
                if _norm and _norm not in _seen_v:
                    _seen_v.add(_norm)
                    _vsevs.append(str(_s).strip())
            _add_sev(severity_filter)
            if tool_name == "get_wazuh_critical_vulnerabilities" and not _vsevs:
                _add_sev("Critical")
            for _sec_t, _sec_a in zip(plan.secondary_tools, plan.secondary_args):
                if _sec_t not in _VULN_TOOLS_RENDER:
                    continue
                _add_sev((_sec_a or {}).get("severity"))
            if len(_vsevs) > 1:
                # The vuln formatter only has 2 render modes: critical-only or
                # full breakdown. For multi-severity asks, force the full
                # breakdown by clearing severity AND overriding tool name (so
                # the critical-specific tool doesn't pin to critical-only view).
                logger.warning(
                    f"[FORMATTER] Multi-severity vuln: forcing full breakdown for {_vsevs}"
                )
                _ms_summary = dict(summary) if isinstance(summary, dict) else {}
                _ms_summary["tool"] = "get_wazuh_vulnerabilities"
                return self.formatter.format_vulnerability_response(
                    correlated_data=data,
                    summary=_ms_summary,
                    include_remediation=True,
                    severity=None,
                    agent_id=agent_id_filter,
                    os=os_filter,
                )
            return self.formatter.format_vulnerability_response(
                correlated_data=data,
                summary=summary,
                include_remediation=True,
                severity=severity_filter,
                agent_id=agent_id_filter,
                os=os_filter,
                tool=tool_name,
            )

        # Agent tools
        elif tool_name in ["get_wazuh_agents", "get_wazuh_running_agents"]:
            logger.warning(f"[FORMATTER] -> AGENT formatter (entered branch)")
            status_filter = plan.primary_args.get("status")
            # get_wazuh_running_agents only returns active agents — force the filter
            if not status_filter and tool_name == "get_wazuh_running_agents":
                status_filter = "active"

            # Phase 2 multi-status fan-out: if the plan has secondary
            # get_wazuh_agents calls with different statuses, render each
            # bucket and concatenate so the response covers all of them.
            _AGENT_TOOLS = ("get_wazuh_agents", "get_wazuh_running_agents")
            _statuses = []
            if status_filter:
                _statuses.append(status_filter)
            elif tool_name == "get_wazuh_running_agents":
                _statuses.append("active")
            for _sec_tool, _sec_args in zip(plan.secondary_tools, plan.secondary_args):
                if _sec_tool not in _AGENT_TOOLS:
                    continue
                _s = _sec_args.get("status")
                if not _s and _sec_tool == "get_wazuh_running_agents":
                    _s = "active"
                if _s and _s not in _statuses:
                    _statuses.append(_s)
            # === TEMP DIAG (remove me) ===
            try:
                _all_st = set()
                if isinstance(data, list):
                    for _it in data:
                        if isinstance(_it, dict):
                            _v = _it.get("status")
                            if _v: _all_st.add(str(_v).lower())
                logger.warning(
                    f"[DIAG] primary_args={plan.primary_args} secondary_args={plan.secondary_args} "
                    f"data_len={len(data) if isinstance(data,list) else type(data).__name__} "
                    f"data_statuses={sorted(_all_st)} computed_statuses={_statuses}"
                )
            except Exception as _dbg:
                logger.info(f"[DIAG] exception: {_dbg}")
            # === END TEMP DIAG ===
            if len(_statuses) > 1:
                logger.warning(f"[FORMATTER] Multi-status render: {_statuses} (data items={len(data) if isinstance(data, list) else 0})")
                _sections = []
                for _st in _statuses:
                    _sec = self.formatter.format_agent_status_response(
                        correlated_data=data,
                        summary=summary,
                        status_filter=_st,
                    )
                    if _sec:
                        _sections.append(_sec)
                if _sections:
                    return "\n\n".join(_sections)

            return self.formatter.format_agent_status_response(
                correlated_data=data,
                summary=summary,
                status_filter=status_filter
            )

        # Agent health check — use dedicated formatter
        elif tool_name == "check_agent_health":
            logger.info(f"[FORMATTER] → HEALTH CHECK formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No health data")
            return self.formatter.format_health_response(raw_text)

        # Alert tools
        elif tool_name == "get_wazuh_alerts":
            logger.info(f"[FORMATTER] → ALERT formatter")
            _time_range = (plan.primary_args or {}).get("time_range", "24h")
            return self.formatter.format_alert_response(
                correlated_data=data,
                summary=summary,
                time_range=_time_range
            )

        # Alert summary — dedicated formatter for aggregation data
        elif tool_name == "get_wazuh_alert_summary":
            logger.info(f"[FORMATTER] → ALERT SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No alert summary data")
            return self.formatter.format_alert_summary_response(raw_text)

        # Alert patterns — dedicated formatter for pattern data
        elif tool_name == "analyze_alert_patterns":
            logger.info(f"[FORMATTER] → ALERT PATTERNS formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No alert pattern data")
            return self.formatter.format_alert_patterns_response(raw_text)

        # Search security events (FIX: dedicated formatter)
        elif tool_name == "search_security_events":
            logger.info(f"[FORMATTER] → SEARCH SECURITY EVENTS formatter")
            search_query = plan.primary_args.get("query", "")
            time_range = plan.primary_args.get("time_range", "24h")
            return self.formatter.format_search_events_response(
                correlated_data=data,
                summary=summary,
                search_query=search_query,
                time_range=time_range
            )

        # Process / ports — use dedicated formatters with suspicious detection
        elif tool_name == "get_agent_processes":
            logger.info(f"[FORMATTER] → PROCESS formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No process data")
            return self.formatter.format_processes_response(raw_text)

        elif tool_name == "get_agent_ports":
            logger.info(f"[FORMATTER] → PORTS formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No port data")
            return self.formatter.format_ports_response(raw_text)

        elif tool_name == "get_agent_configuration":
            logger.info(f"[FORMATTER] → AGENT CONFIG formatter")
            raw_text = self._extract_raw_text(tool_response, "⚠️ No config data")
            return self.formatter.format_agent_configuration_response(raw_text)

        # ----------------------------------------------------------------
        # GROUP D: Round 6 Enterprise SOC Analysis Tools
        # ----------------------------------------------------------------

        elif tool_name == "get_rule_trigger_analysis":
            logger.info(f"[FORMATTER] → RULE TRIGGER ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No rule analysis data")
            return self.formatter.format_rule_trigger_response(raw_text)

        elif tool_name == "get_mitre_coverage":
            logger.info(f"[FORMATTER] → MITRE COVERAGE formatter")
            raw_text = self._extract_raw_text(tool_response, "No MITRE coverage data")
            return self.formatter.format_mitre_coverage_response(raw_text)

        elif tool_name == "get_alert_timeline":
            logger.info(f"[FORMATTER] → ALERT TIMELINE formatter")
            raw_text = self._extract_raw_text(tool_response, "No timeline data")
            return self.formatter.format_alert_timeline_response(raw_text)

        elif tool_name == "get_log_source_health":
            logger.info(f"[FORMATTER] → LOG SOURCE HEALTH formatter")
            raw_text = self._extract_raw_text(tool_response, "No log health data")
            return self.formatter.format_log_source_health_response(raw_text)

        elif tool_name == "get_decoder_analysis":
            logger.info(f"[FORMATTER] → DECODER ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No decoder data")
            return self.formatter.format_decoder_analysis_response(raw_text)

        elif tool_name == "get_fim_events":
            logger.info(f"[FORMATTER] → FIM EVENTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No FIM data")
            return self.formatter.format_fim_events_response(raw_text)

        elif tool_name == "get_cloudflare_http_summary":
            logger.info(f"[FORMATTER] → CLOUDFLARE HTTP SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare HTTP data")
            return self.formatter.format_cloudflare_http_summary_response(raw_text)

        elif tool_name == "get_cloudflare_http_errors":
            logger.info(f"[FORMATTER] → CLOUDFLARE HTTP ERRORS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare error data")
            return self.formatter.format_cloudflare_http_errors_response(raw_text)

        elif tool_name == "get_cloudflare_firewall_events":
            logger.info(f"[FORMATTER] → CLOUDFLARE FIREWALL EVENTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare firewall data")
            return self.formatter.format_cloudflare_firewall_events_response(raw_text)

        elif tool_name == "get_cloudflare_firewall_top_attackers":
            logger.info(f"[FORMATTER] → CLOUDFLARE TOP ATTACKERS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare attacker data")
            return self.formatter.format_cloudflare_firewall_top_attackers_response(raw_text)

        elif tool_name == "get_cloudflare_firewall_top_rules":
            logger.info(f"[FORMATTER] → CLOUDFLARE TOP RULES formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare rule data")
            return self.formatter.format_cloudflare_firewall_top_rules_response(raw_text)

        elif tool_name == "get_cloudflare_security_summary":
            logger.info(f"[FORMATTER] → CLOUDFLARE SECURITY SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare security data")
            return self.formatter.format_cloudflare_security_summary_response(raw_text)

        elif tool_name == "get_cloudflare_http_top_paths":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare path data")
            return self.formatter.format_cloudflare_http_top_paths_response(raw_text)

        elif tool_name == "get_cloudflare_http_top_clients":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare client data")
            return self.formatter.format_cloudflare_http_top_clients_response(raw_text)

        elif tool_name == "get_cloudflare_http_search":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare search results")
            return self.formatter.format_cloudflare_http_search_response(raw_text)

        elif tool_name == "get_cloudflare_cache_performance":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare cache data")
            return self.formatter.format_cloudflare_cache_performance_response(raw_text)

        elif tool_name == "get_cloudflare_cache_top_misses":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare cache miss data")
            return self.formatter.format_cloudflare_cache_top_misses_response(raw_text)

        elif tool_name == "get_cloudflare_dns_summary":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare DNS data")
            return self.formatter.format_cloudflare_dns_summary_response(raw_text)

        elif tool_name == "get_cloudflare_dns_top_queries":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare DNS query data")
            return self.formatter.format_cloudflare_dns_top_queries_response(raw_text)

        elif tool_name == "get_cloudflare_dns_errors":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare DNS error data")
            return self.formatter.format_cloudflare_dns_errors_response(raw_text)

        elif tool_name == "get_cloudflare_workers_summary":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare Workers data")
            return self.formatter.format_cloudflare_workers_summary_response(raw_text)

        elif tool_name == "get_cloudflare_workers_errors":
            raw_text = self._extract_raw_text(tool_response, "No Cloudflare Workers error data")
            return self.formatter.format_cloudflare_workers_errors_response(raw_text)

        elif tool_name == "get_sca_results":
            logger.info(f"[FORMATTER] → SCA RESULTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No SCA data")
            return self.formatter.format_sca_results_response(raw_text)

        elif tool_name == "get_agent_inventory":
            logger.info(f"[FORMATTER] → AGENT INVENTORY formatter")
            raw_text = self._extract_raw_text(tool_response, "No inventory data")
            return self.formatter.format_agent_inventory_response(raw_text)

        elif tool_name == "get_rootcheck_results":
            logger.info(f"[FORMATTER] → ROOTCHECK formatter")
            raw_text = self._extract_raw_text(tool_response, "No rootcheck data")
            return self.formatter.format_rootcheck_response(raw_text)

        elif tool_name == "correlate_alerts_vulnerabilities":
            logger.info(f"[FORMATTER] → ALERT-VULN CORRELATION formatter")
            raw_text = self._extract_raw_text(tool_response, "No correlation data")
            return self.formatter.format_alert_vuln_correlation_response(raw_text)

        elif tool_name == "get_behavioral_baseline":
            logger.info(f"[FORMATTER] → BEHAVIORAL BASELINE formatter")
            raw_text = self._extract_raw_text(tool_response, "No baseline data")
            return self.formatter.format_behavioral_baseline_response(raw_text)

        elif tool_name == "generate_wazuh_decoder":
            logger.info(f"[FORMATTER] → DECODER GENERATOR formatter")
            raw_text = self._extract_raw_text(tool_response, "No decoder data")
            return self.formatter.format_decoder_generator_response(raw_text)

        # ----------------------------------------------------------------
        # GROUP E: Suricata IDS Tools
        # ----------------------------------------------------------------

        elif tool_name == "get_suricata_alerts":
            logger.info(f"[FORMATTER] → SURICATA ALERTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata alert data")
            return self.formatter.format_suricata_alerts_response(raw_text)

        elif tool_name == "get_suricata_alert_summary":
            logger.info(f"[FORMATTER] → SURICATA ALERT SUMMARY formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata summary data")
            return self.formatter.format_suricata_alert_summary_response(raw_text)

        elif tool_name in [
            "get_suricata_critical_alerts",
            "get_suricata_high_alerts",
            "get_suricata_medium_alerts",
            "get_suricata_low_alerts",
        ]:
            # User's dashboard maps Critical=2, Alert(High)=1, Warning(Medium)=3, Other(Low)=4+
            # — opposite of standard Suricata for the top two tiers. Match that mapping
            # so chat output and dashboard cards agree.
            severity_map = {
                "get_suricata_critical_alerts": 2,    # was 1
                "get_suricata_high_alerts":     1,    # was 2
                "get_suricata_medium_alerts":   3,
                "get_suricata_low_alerts":      4,
            }
            expected_sev = severity_map[tool_name]
            logger.info(f"[FORMATTER] → SURICATA SEVERITY-{expected_sev} ALERTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata alert data")
            return self.formatter.format_suricata_severity_alerts_response(
                raw_text, expected_severity=expected_sev
            )

        elif tool_name == "get_suricata_network_analysis":
            logger.info(f"[FORMATTER] → SURICATA NETWORK ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata network data")
            return self.formatter.format_suricata_network_analysis_response(raw_text)

        elif tool_name == "search_suricata_alerts":
            logger.info(f"[FORMATTER] → SURICATA SEARCH formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata search results")
            return self.formatter.format_suricata_search_response(raw_text)

        elif tool_name == "get_suricata_top_signatures":
            logger.info(f"[FORMATTER] → SURICATA TOP SIGNATURES formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata signature data")
            return self.formatter.format_suricata_signatures_response(raw_text)

        elif tool_name == "get_suricata_top_attackers":
            logger.info(f"[FORMATTER] → SURICATA TOP ATTACKERS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata attacker data")
            # Reuse network analysis formatter for attacker data
            return self.formatter.format_suricata_network_analysis_response(raw_text)

        elif tool_name == "get_suricata_top_destinations":
            # M6c (2026-06-10) - dedicated destinations-focus renderer
            logger.info(f"[FORMATTER] → SURICATA TOP DESTINATIONS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata destination data")
            return self.formatter.format_suricata_top_destinations_response(raw_text)

        elif tool_name == "get_suricata_health":
            logger.info(f"[FORMATTER] → SURICATA HEALTH formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata health data")
            return self.formatter.format_suricata_health_response(raw_text)

        elif tool_name == "get_suricata_category_breakdown":
            logger.info(f"[FORMATTER] → SURICATA CATEGORY formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata category data")
            return self.formatter.format_suricata_category_breakdown_response(raw_text)

        # Suricata Deep Visibility formatters (HTTP, TLS, MITRE, JA3)
        elif tool_name == "get_suricata_http_analysis":
            logger.info(f"[FORMATTER] → SURICATA HTTP ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No HTTP analysis data")
            return self.formatter.format_suricata_http_analysis_response(raw_text)

        elif tool_name == "search_suricata_http":
            logger.info(f"[FORMATTER] → SURICATA HTTP SEARCH formatter")
            raw_text = self._extract_raw_text(tool_response, "No HTTP search results")
            return self.formatter.format_suricata_http_search_response(raw_text)

        elif tool_name == "get_suricata_tls_analysis":
            logger.info(f"[FORMATTER] → SURICATA TLS ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No TLS analysis data")
            return self.formatter.format_suricata_tls_analysis_response(raw_text)

        elif tool_name == "get_suricata_mitre_mapping":
            logger.info(f"[FORMATTER] → SURICATA MITRE MAPPING formatter")
            raw_text = self._extract_raw_text(tool_response, "No MITRE mapping data")
            return self.formatter.format_suricata_mitre_mapping_response(raw_text)

        elif tool_name == "get_suricata_ja3_fingerprints":
            logger.info(f"[FORMATTER] → SURICATA JA3 FINGERPRINTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No JA3 fingerprint data")
            return self.formatter.format_suricata_ja3_response(raw_text)

        elif tool_name == "get_suricata_suspicious_activity":
            logger.info(f"[FORMATTER] → SURICATA SUSPICIOUS ACTIVITY formatter")
            raw_text = self._extract_raw_text(tool_response, "No suspicious activity data")
            return self.formatter.format_suricata_suspicious_activity_response(raw_text)

        elif tool_name == "get_suricata_traffic_overview":
            logger.info(f"[FORMATTER] → SURICATA TRAFFIC OVERVIEW formatter")
            raw_text = self._extract_raw_text(tool_response, "No traffic overview data")
            return self.formatter.format_suricata_traffic_overview_response(raw_text)

        # --- Pallas 3.3: JA4 and Flow tools ---
        elif tool_name == "get_suricata_ja4_analysis":
            logger.info(f"[FORMATTER] → SURICATA JA4 ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No JA4 fingerprint data")
            return self.formatter.format_suricata_ja4_response(raw_text)

        elif tool_name == "get_suricata_flow_analysis":
            logger.info(f"[FORMATTER] → SURICATA FLOW ANALYSIS formatter")
            raw_text = self._extract_raw_text(tool_response, "No flow analysis data")
            return self.formatter.format_suricata_flow_analysis_response(raw_text)

        # ----------------------------------------------------------------
        # Unknown tool — safe fallback, never dump MCP wrapper JSON
        # ----------------------------------------------------------------
        else:
            logger.warning(f"[FORMATTER] → UNKNOWN tool: {tool_name} — using safe text extraction")
            text = self._extract_raw_text(tool_response, "")
            if text:
                # 2026-05-21: detect JSON-shaped fall-through and route through
                # format_raw_json_response so users see a pretty code block
                # instead of an inline JSON wall. Catches any future tool that
                # lands here without an explicit dispatch case.
                _stripped = text.lstrip()
                _looks_json = _stripped.startswith("{") or _stripped.startswith("[")
                _has_label = "\n" in text and text.split("\n", 1)[0].strip().endswith(":")
                if _looks_json or _has_label:
                    return self.formatter.format_raw_json_response(
                        text,
                        title=f"Query Results: {tool_name}",
                    )
                return f"## Query Results: {tool_name}\n\n{text}"
            return self.formatter.format_general_response(
                data=data,
                title=f"Query Results: {tool_name}"
            )

    def _format_cluster_health(self, tool_response: Dict[str, Any]) -> str:
        """Format cluster health into a readable SOC dashboard card."""
        raw = self._extract_raw_text(tool_response, "")
        if not raw:
            return "⚠️ No cluster health data available"

        # Try to parse the inner JSON from text like "Cluster Health:\n{...}"
        try:
            # Strip any leading label line
            json_start = raw.find("{")
            if json_start >= 0:
                health = json.loads(raw[json_start:])
            else:
                health = {}
        except Exception:
            # Can't parse — return the text as-is in a code block
            return f"## 🏥 Cluster Health\n\n```\n{raw}\n```"

        status = health.get("status", "unknown")
        error = health.get("error", "")
        note = health.get("note", "")

        status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(status.lower(), "⚪")

        lines = [
            "## 🏥 Wazuh Cluster Health",
            "",
            f"**Status:** {status_icon} {status.upper()}",
        ]

        if note:
            lines.append(f"**Note:** {note}")
        if error:
            lines.append(f"**Details:** {error}")

        # Show any other fields
        skip_keys = {"status", "error", "note"}
        for k, v in health.items():
            if k not in skip_keys and v is not None:
                lines.append(f"**{k.replace('_', ' ').title()}:** {v}")

        if status.lower() == "unknown" or error:
            lines.append("")
            lines.append("ℹ️ *Cluster mode may not be enabled. Single-node deployments report status as unknown.*")

        return "\n".join(lines)

    def _format_remoted_stats(self, tool_response: Dict[str, Any]) -> str:
        """Format remoted stats into a human-readable SOC dashboard card."""
        raw = self._extract_raw_text(tool_response, "")
        if not raw:
            return "⚠️ No remoted stats available"

        # Parse JSON from the text (format: "Remoted Statistics:\n{...}")
        try:
            json_start = raw.find("{")
            if json_start >= 0:
                parsed = json.loads(raw[json_start:])
            else:
                parsed = {}
        except Exception:
            return f"## 📡 Remoted Statistics\n\n```\n{raw}\n```"

        items = parsed.get("data", {}).get("affected_items", [])
        if not items:
            return f"## 📡 Remoted Statistics\n\n```\n{raw}\n```"

        s = items[0]  # Typically one node

        def fmt_bytes(b):
            try:
                b = float(b)
                if b >= 1024*1024: return f"{b/1024/1024:.1f} MB"
                if b >= 1024: return f"{b/1024:.1f} KB"
                return f"{b:.0f} B"
            except Exception:
                return str(b)

        lines = [
            "## 📡 Wazuh Remoted Statistics",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| TCP Sessions | {int(s.get('tcp_sessions', 0))} |",
            f"| Events Received | {int(s.get('evt_count', 0)):,} |",
            f"| Control Messages | {int(s.get('ctrl_msg_count', 0)):,} |",
            f"| Discarded Events | {int(s.get('discarded_count', 0))} |",
            f"| Bytes Sent | {fmt_bytes(s.get('sent_bytes', 0))} |",
            f"| Bytes Received | {fmt_bytes(s.get('recv_bytes', 0))} |",
            f"| Queue Size | {int(s.get('queue_size', 0))} / {int(s.get('total_queue_size', 0)):,} |",
            f"| Ctrl Queue Processed | {int(s.get('ctrl_msg_queue_processed', 0)):,} |",
        ]

        lines.append("")
        lines.append(f"*Data from Wazuh remoted daemon — agent communication layer*")
        return "\n".join(lines)

    def _format_generic_stats(self, tool_response: Dict[str, Any], title: str) -> str:
        """Generic stats formatter: extract text and display cleanly."""
        raw = self._extract_raw_text(tool_response, "")
        if not raw:
            return f"## {title}\n\n⚠️ No data available"

        # Try to parse and display as a clean table if it has affected_items
        try:
            json_start = raw.find("{")
            if json_start >= 0:
                parsed = json.loads(raw[json_start:])
                items = parsed.get("data", {}).get("affected_items", [])
                if items and isinstance(items[0], dict):
                    lines = [f"## {title}", "", "```json",
                             json.dumps(items[0], indent=2, default=str)[:2000], "```"]
                    return "\n".join(lines)
        except Exception:
            pass

        return f"## {title}\n\n```\n{raw[:3000]}\n```"

    def _format_m5_section(self, agg_data: Any, agg_tool: str, primary_total: Optional[int]) -> str:
        """M5 (2026-06-10) - render the aggregation result as a "Full-Scale
        Analysis" markdown sub-section to prepend above the sample table.
        Delegates rendering to the same summary formatter the analyst would
        get if they ran the aggregation tool directly; the top-level `## `
        heading is downgraded to `### ` so it nests cleanly inside the
        parent response. Returns "" on any failure (silently logs)."""
        try:
            method_name = _M5_FORMATTER_METHOD_MAP.get(agg_tool)
            if not method_name:
                return ""
            formatter_method = getattr(self.formatter, method_name, None)
            if not callable(formatter_method):
                return ""
            # Extract raw markdown text from the MCP envelope if present
            raw_text = self._extract_raw_text(
                agg_data if isinstance(agg_data, dict) else {"data": agg_data},
                ""
            )
            if not raw_text or not isinstance(raw_text, str):
                # Fallback: dump the dict as JSON for the formatter to parse
                try:
                    raw_text = json.dumps(agg_data, default=str)
                except Exception:
                    return ""
            agg_md = formatter_method(raw_text)
            if not agg_md or not isinstance(agg_md, str):
                return ""
            agg_md = agg_md.strip()
            if not agg_md:
                return ""
            # Downgrade `## ` headings to `### ` (and `### ` to `#### ` etc.)
            # so the section nests cleanly inside the parent response.
            downgraded_lines = []
            for line in agg_md.split("\n"):
                if line.startswith("## "):
                    downgraded_lines.append("#" + line)
                else:
                    downgraded_lines.append(line)
            downgraded = "\n".join(downgraded_lines)
            total_str = f"{primary_total:,}" if primary_total else "the full population"
            return (
                f"### Full-Scale Analysis (population: {total_str} events)\n\n"
                f"_Auto-aggregated from `{agg_tool}` so the analysis below reflects "
                f"the full population, not just the sample shown later._\n\n"
                f"{downgraded}"
            )
        except Exception as exc:
            logger.warning(f"[M5_AUGMENT] _format_m5_section crashed: {exc}", exc_info=True)
            return ""

    def _build_correlation_header(self, correlation_type: str, summary: Dict[str, Any]) -> str:
        """Build a contextual header for correlated multi-tool results."""
        headers = {
            # Wazuh correlations
            "vulnerability_with_agents": "## Vulnerabilities Correlated with Agent Details\n",
            "agents_with_vulnerabilities": "## Agents Ranked by Vulnerability Exposure\n",
            "top_agents_by_vuln_count": "## Top Agents by Vulnerability Count\n",
            "disconnected_agents_with_critical_vulns": "## Disconnected Agents with Critical Vulnerabilities\n",
            "active_agents_with_high_vulns": "## Active Agents with High/Critical Vulnerabilities\n",
            "active_agents_vulns_over_threshold": "## Active Agents Exceeding Vulnerability Threshold\n",
            "compare_vulns_active_vs_disconnected": "## Vulnerability Comparison: Active vs Disconnected\n",
            "alerts_with_agents": "## Alerts Enriched with Agent Context\n",
            "agent_posture_deep_dive": "## Agent Security Posture -- Deep Dive\n",
            "alerts_with_vulnerabilities": "## Combined Risk: Agents with Alerts AND Vulnerabilities\n",
            "rule_mitre_with_agents": "## MITRE ATT&CK Coverage with Agent Mapping\n",
            "fim_with_agent_posture": "## FIM Events with Agent and Process Context\n",
            "vulnerability_trend_by_severity": "## Vulnerability Distribution by Severity\n",
            # Suricata correlations
            "suricata_attacker_target_map": "## Suricata Attacker-Target Mapping\n",
            "suricata_signature_severity_analysis": "## Suricata Signature Severity Analysis\n",
            "suricata_network_threat_profile": "## Network Threat Profile\n",
            "suricata_http_alert_context": "## Suricata Alerts with HTTP Context\n",
            "suricata_tls_threat_detection": "## TLS Threat Detection Analysis\n",
            "suricata_recon_detection": "## Reconnaissance Detection Results\n",
            # Cross-platform correlations
            "combined_alert_view": "## Combined Alert View: Wazuh + Suricata\n",
            "wazuh_agents_suricata_detections": "## Agent-to-IDS Detection Mapping\n",
            "ioc_with_suricata_context": "## IoC Analysis with Suricata Context\n",
            "comprehensive_security_posture": "## Comprehensive Security Posture\n",
            "wazuh_suricata_mitre_unified": "## Unified MITRE ATT&CK Coverage\n",
            # Pallas 3.2 universal correlations
            "ip_investigation_pivot": "## IP Investigation Pivot\n",
            "full_agent_investigation": "## Full Agent Investigation\n",
            "cross_platform_ip_correlation": "## Cross-Platform IP Correlation\n",
            "unified_scanning_detection": "## Unified Scanning Detection\n",
            "attack_chain_analysis": "## Attack Chain Analysis\n",
            "unified_threat_summary": "## Unified Threat Summary\n",
            "vulnerability_exploit_correlation": "## Vulnerability-Exploit Correlation\n",
            "detection_coverage_gap": "## Detection Coverage Gap Analysis\n",
            "fim_alert_correlation": "## FIM-Alert Correlation\n",
            "port_exposure_risk": "## Port Exposure Risk Assessment\n",
            "top_risk_agents_composite": "## Top Risk Agents (Composite Score)\n",
            "alert_context_enrichment": "## Alert Context Enrichment\n",
            # Pallas 3.4: Agent health analytics
            "agent_health_overview": "## Agent Fleet Health Overview\n",
            "agent_event_volume": "## Agent Event Volume Analysis\n",
            # Pallas 3.5: Advanced SOC correlations
            "temporal_attack_sequence": "## Temporal Attack Sequence Analysis\n",
            "dynamic_risk_scoring": "## Dynamic Risk Assessment\n",
            "cross_platform_threat_correlation": "## Cross-Platform Threat Correlation\n",
            "data_exfiltration_detection": "## Data Exfiltration Detection\n",
            "off_hours_anomaly_detection": "## Off-Hours Anomaly Detection\n",
            "alert_noise_analysis": "## Alert Noise Analysis\n",
            "multi_asset_campaign_detection": "## Multi-Asset Campaign Detection\n",
        }
        return headers.get(correlation_type, "")

    def _format_agent_posture_deep_dive(self, data: list, summary: dict) -> str:
        """Format deep-dive correlation: agent + vulns + ports + processes."""
        if not data:
            return "No deep-dive data available."

        sections = []
        item = data[0] if isinstance(data, list) and len(data) >= 1 else data

        if not isinstance(item, dict):
            return "No deep-dive data available."

        # Agent info section
        agent = item.get("agent", item)
        if isinstance(agent, dict):
            sections.append("### Agent Information\n")
            sections.append("| Field | Value |")
            sections.append("|-------|-------|")
            for fld in ["id", "name", "ip", "status", "os", "version", "last_keep_alive"]:
                val = agent.get(fld, "N/A")
                if isinstance(val, dict):
                    val = val.get("name", str(val))
                sections.append(f"| {fld.replace('_', ' ').title()} | {val} |")

        # Vulnerability summary
        vuln_summary = item.get("vulnerability_summary", {})
        if vuln_summary and isinstance(vuln_summary, dict):
            sections.append("\n### Vulnerability Summary\n")
            total = vuln_summary.get("total", 0)
            parts = [f"**Total:** {total}"]
            for sev in ["critical", "high", "medium", "low"]:
                count = vuln_summary.get(sev, 0)
                if count > 0:
                    parts.append(f"**{sev.title()}:** {count}")
            sections.append(" | ".join(parts))

        # Top vulnerabilities table
        top_vulns = item.get("top_vulnerabilities", [])
        if top_vulns and isinstance(top_vulns, list):
            sections.append("\n\n| CVE ID | Severity | CVSS | Package |")
            sections.append("|--------|----------|------|---------|")
            for v in top_vulns[:10]:
                if isinstance(v, dict):
                    cve = v.get("cve_id", v.get("id", "N/A"))
                    sev = v.get("severity", "N/A")
                    cvss = v.get("cvss_score", v.get("cvss", "N/A"))
                    pkg = v.get("package", "N/A")
                    if isinstance(pkg, dict):
                        pkg = pkg.get("name", str(pkg))
                    sections.append(f"| {cve} | {sev} | {cvss} | {pkg} |")

        # Open ports
        ports = item.get("open_ports", [])
        if ports and isinstance(ports, list):
            sections.append("\n### Open Ports\n")
            sections.append("| Port | Protocol | State | Process |")
            sections.append("|------|----------|-------|---------|")
            for p in ports[:15]:
                if isinstance(p, dict):
                    local = p.get("local", {})
                    port_num = local.get("port", p.get("port", "N/A")) if isinstance(local, dict) else p.get("port", "N/A")
                    proto = p.get("protocol", "N/A")
                    state = p.get("state", "N/A")
                    proc = p.get("process", "N/A")
                    sections.append(f"| {port_num} | {proto} | {state} | {proc} |")
        elif item.get("ports_raw_text"):
            # Fallback: use pre-formatted port text from server.py
            sections.append("\n### Open Ports\n")
            sections.append(item["ports_raw_text"])

        # Running processes
        procs = item.get("running_processes", [])
        if procs and isinstance(procs, list):
            sections.append("\n### Running Processes\n")
            sections.append("| PID | Name | User | State |")
            sections.append("|-----|------|------|-------|")
            for proc in procs[:15]:
                if isinstance(proc, dict):
                    pid = proc.get("pid", "N/A")
                    name = proc.get("name", "N/A")
                    user = proc.get("euser", proc.get("user", "N/A"))
                    state = proc.get("state", "N/A")
                    sections.append(f"| {pid} | {name} | {user} | {state} |")
        elif item.get("procs_raw_text"):
            # Fallback: use pre-formatted process text from server.py
            sections.append("\n### Running Processes\n")
            sections.append(item["procs_raw_text"])

        # Risk score
        risk = item.get("risk_score")
        if risk is not None:
            level = "CRITICAL" if risk >= 75 else "HIGH" if risk >= 50 else "MEDIUM" if risk >= 25 else "LOW"
            sections.append(f"\n### Risk Assessment\n\n**Overall Risk Score:** {risk}/100 ({level})")

        return "\n".join(sections)

    def _safe_fallback_format(
        self,
        tool_name: str,
        tool_response: Dict[str, Any],
        data: List[Dict[str, Any]],
        error: Exception
    ) -> str:
        """
        Safe fallback mode - always returns something readable.
        Called when the normal formatter throws an exception.
        """
        logger.warning(f"[SAFE_FALLBACK] Activated for tool={tool_name}, error={error}")

        lines = [f"## ⚠️ Results (Safe Fallback Mode)"]
        lines.append(f"*Formatter error: {str(error)[:100]}*")
        lines.append("")

        # Try to extract raw text from MCP response
        if isinstance(tool_response, dict) and "content" in tool_response:
            content = tool_response.get("content", [])
            if content and isinstance(content, list):
                raw_text = content[0].get("text", "")
                if raw_text:
                    lines.append("**Raw Data:**")
                    lines.append(f"```\n{raw_text[:2000]}\n```")
                    return "\n".join(lines)

        # Try to show items as JSON
        if data:
            lines.append(f"**{len(data)} items found:**")
            lines.append(f"```json\n{json.dumps(data[:5], indent=2, default=str)[:2000]}\n```")
        else:
            lines.append("*No data could be extracted.*")

        return "\n".join(lines)

    def _format_error_response(self, result: ExecutionResult, plan: ToolPlan) -> str:
        """E1.3a (2026-06-11): friendly tool-error Markdown lives in
        orchestrator/formatting/error_responses.py now. This method
        stays as a thin instance-method delegate so every existing
        call site keeps working unchanged."""
        from .orchestrator.formatting.error_responses import format_friendly_error
        return format_friendly_error(result.tool_name or "", result.error or "")

    def _format_exception_response(self, exception: Exception) -> str:
        """E1.3a (2026-06-11): pipeline-exception Markdown lives in
        orchestrator/formatting/error_responses.py. Thin delegate."""
        from .orchestrator.formatting.error_responses import format_exception_response
        return format_exception_response(exception)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""

        return {
            "tool_selection": self.selector.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_orchestrator(mcp_tool_executor, suricata_enabled: bool = False) -> QueryOrchestrator:
    """
    Factory function to create and initialize orchestrator.

    Args:
        mcp_tool_executor: MCP tool executor function
        suricata_enabled: Whether Suricata IDS integration is active

    Returns:
        Initialized QueryOrchestrator with hybrid intelligence
    """

    import os

    # Cloud-only LLM routing (local Ollama removed 2026-04-15 per directive).
    # All LLM calls — including previously-sensitive SOC analysis and forensic
    # narratives — now go through the cloud provider.
    cloud_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
    cloud_model = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus")
    cloud_key = os.getenv("LLM_API_KEY", "")

    cloud_llm = LLMClient(base_url=cloud_url, model=cloud_model, api_key=cloud_key)
    llm_router = LLMRouter(None, cloud_llm)
    await llm_router.initialize()

    # Create orchestrator — passes router as the LLM client
    orchestrator = QueryOrchestrator(
        mcp_tool_executor=mcp_tool_executor,
        ollama_client=llm_router,
        suricata_enabled=suricata_enabled
    )

    logger.info("QueryOrchestrator initialized: CLOUD-ONLY LLM routing")
    logger.info(f"   CLOUD: {cloud_model} at {cloud_url} (handles ALL LLM workload)")
    logger.info(f"   Suricata: {'ENABLED' if suricata_enabled else 'DISABLED'}")

    return orchestrator


# ============================================================================
# LESSONS LEARNED / CHANGELOG
# ============================================================================
#
# --- SESSION 1 (2026-02-19) ---
# Root Cause:
#   1. search_security_events had NO keyword or regex pattern defined.
#      All "virustotal event" queries fell through to LLM → LLM picked
#      get_wazuh_alerts (wrong tool, no query filter) → 10k unfiltered alerts shown.
#   2. _format_response() had no routing case for search_security_events.
#   3. _build_arguments() never populated the "query" param for search_security_events.
#   4. alert_general keyword pattern had no exclude_keywords for "search"/"virustotal".
#   5. No safe fallback mode: when formatter threw, user got an exception dump.
#
# Fix:
#   - Added regex+keyword patterns for search_security_events.
#   - Added _build_search_events_args() to extract query term from NL.
#   - Added explicit routing in _route_formatter() for search_security_events.
#   - Added _safe_fallback_format() mode.
#   - alert_general now excludes ["search","virustotal","fim","syscheck"].
#
# --- SESSION 3 (2026-02-19) ---
# Root Cause:
#   1. "vulnerability summary" → get_wazuh_vulnerabilities (wrong tool, shows critical CVE list).
#      Root cause: vulnerability_general keyword pattern matched "vulnerabilit*" BEFORE
#      any vulnerability_summary pattern existed. No HIGH_CONFIDENCE regex for "summary" either.
#      Fix: Added vulnerability_summary_explicit to HIGH_CONFIDENCE (confidence=0.97) and
#      vulnerability_summary to KEYWORD_PATTERNS placed BEFORE vulnerability_general.
#
#   2. "please check cluster health" → check_agent_health guardrail fires.
#      Root cause: agent_health HIGH_CONFIDENCE regex was:
#        r"\b(check|validate|verify)\b.*\b(health|status)\b"
#      This matched "check cluster health" (no 'agent' required). Guardrail demanded agent_id.
#      Also: KEYWORD_PATTERNS agent_health used bare keyword "health" — matched everything.
#      Fix: All agent_health regex patterns now require the word 'agent' explicitly.
#      agent_health keywords changed from ["health","heartbeat","keepalive"] to
#      ["agent health","agent heartbeat","check agent",...] with exclude_keywords=["cluster"].
#      cluster_health keywords expanded to include "check cluster", "wazuh cluster", etc.
#
#   3. "show manager errors" → 400 Bad Request: limit 10000 > max 500.
#      Root cause: _build_arguments pulls schema default via ToolRegistry. ToolRegistry
#      for some tools registers limit default=10000 (from search_wazuh_manager_logs alias).
#      Wazuh API /manager/logs enforces maximum=500. Orchestrator passed 10000 → API rejected.
#      Fix: Added API_LIMIT_CAPS dict in _build_arguments() that silently caps any limit
#      exceeding the Wazuh API hard maximum for each tool, before the args leave orchestrator.
#
# Future Prevention:
#   - RULE: Every tool with a "summary" or "overview" variant MUST have:
#     (a) A dedicated HIGH_CONFIDENCE regex pattern (confidence≥0.95, before generic patterns)
#     (b) A keyword pattern placed BEFORE any general keyword pattern for the same domain
#   - RULE: Agent-specific tool regex patterns MUST include \bagent\b in ALL branches.
#     Never use bare \b(check|health|status)\b without 'agent' — too greedy.
#   - RULE: API_LIMIT_CAPS must be updated whenever a new tool is added.
#     Wazuh API maximums: /manager/logs=500, /alerts=??, /agents=500, /vulnerabilities=??
#   - CI regression tests:
#     "give me a vulnerability summary" → assert tool=get_wazuh_vulnerability_summary
#     "please check cluster health" → assert tool=get_wazuh_cluster_health (NOT check_agent_health)
#     "show manager errors" → assert limit≤500 in args
# Root Cause:
#   1. get_wazuh_cluster_health, get_wazuh_remoted_stats, get_wazuh_statistics,
#      get_wazuh_weekly_stats, get_wazuh_log_collector_stats, get_wazuh_rules_summary,
#      validate_wazuh_connection, get_wazuh_cluster_nodes had NO routing case in
#      _route_formatter(). They fell into the "statistics" substring check or the
#      else/general branch. Both branches called format_statistics_response() or
#      format_general_response() with the ENTIRE MCP wrapper dict → dashboard showed
#      raw {"content": [...]} JSON blob instead of human-readable output.
#   2. "maanger errors" (user typo) → no keyword match for manager_error_logs →
#      LLM fallback picked get_wazuh_alerts (wrong tool).
#   3. The old "raw response" paths used tool_response["content"][0].get("text")
#      inline per-case instead of a shared helper → inconsistent and error-prone.
#
# Fix:
#   - Added _extract_raw_text() universal helper used by ALL raw-response paths.
#   - Added explicit routing cases for ALL 29 tools — no tool reaches the else.
#   - Added _format_cluster_health(), _format_remoted_stats(), _format_generic_stats()
#     for human-readable dashboard cards instead of raw JSON blobs.
#   - Added typo-tolerant regex patterns for manager error logs:
#     r"\bma+na+ge?r?\b.*\berr" catches maanger, manger, maneger, etc.
#   - Added expanded keyword list for manager_error_logs with typo variants.
#   - Added manager_errors_regex to HIGH_CONFIDENCE_PATTERNS (fires before keywords).
#
# Future Prevention:
#   - RULE: Every new tool added to tools/list MUST have:
#     (a) at least one keyword/regex pattern in selector
#     (b) explicit routing case in _route_formatter() using _extract_raw_text()
#     (c) a formatter method or "raw" designation documented
#   - Use _extract_raw_text() for ALL raw-response tools. Never inline content[0].get().
#   - Add typo-tolerant regex to HIGH_CONFIDENCE_PATTERNS for all critical tool categories.
#   - CI regression: "please check cluster health" → assert formatted card output,
#     NOT raw JSON; "please share maanger errors" → assert tool=get_wazuh_manager_error_logs.
