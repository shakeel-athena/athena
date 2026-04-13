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
# ============================================================================

@dataclass
class ToolPlan:
    """Tool execution plan with confidence scoring"""
    primary_tool: str
    secondary_tools: List[str] = field(default_factory=list)
    primary_args: Dict[str, Any] = field(default_factory=dict)
    secondary_args: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0
    selection_method: str = "unknown"  # "rule_regex", "rule_keyword", "llm", "fallback"
    reasoning: str = ""
    requires_correlation: bool = False
    correlation_strategy: Optional[str] = None
    fallback_guidance: Optional[str] = None  # Intelligent guidance when query can't be resolved


@dataclass
class ExecutionResult:
    """Result of single tool execution"""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: float


@dataclass
class QueryContext:
    """Context about the query for logging/analytics and conversation memory"""
    original_query: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_summary: str = ""
    carried_entities: Dict[str, Any] = field(default_factory=dict)
    is_follow_up: bool = False
    previous_tool: Optional[str] = None
    previous_result_type: Optional[str] = None


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
                    r"\b(health|heartbeat)\b.*\bagent\b"
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
            # to prevent "security events" regex in alert_investigation from hijacking
            "search_security_events_explicit": {
                "patterns": [
                    r"\bsearch\b.*\b(event|events|security\s+event)\b",
                    r"\b(find|lookup|look\s+up)\b.*\b(event|events|security\s+event)\b",
                    r"\b(virustotal|ossec|rootcheck)\b.*\b(event|events|alert|alerts)\b",
                    r"\b(event|events)\b.*\b(virustotal|ossec|rootcheck)\b",
                    r"search\s+(for\s+)?(security\s+)?events?\b",
                ],
                "tool": "search_security_events",
                "secondary": [],
                "confidence": 0.92,
                "description": "Explicit search security events query"
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
                    r"\b(deep\s+dive|posture|full\s+assessment|comprehensive)\b.*\bagent\b",
                    r"\bagent\b.*\b(deep\s+dive|posture|full\s+assessment|security\s+posture)",
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
                    r"\bmitre\b.*\b(coverage|mapping|technique|tactic|heatmap)",
                    r"\b(att&ck|attack)\b.*\b(coverage|technique|mapping|matrix)",
                    r"\bdetection\s+(coverage|gap|blind\s+spot)",
                    r"\bwhich\s+mitre\b.*\b(technique|tactic)",
                ],
                "tool": "get_mitre_coverage",
                "secondary": [],
                "confidence": 0.93,
                "description": "MITRE ATT&CK coverage mapping"
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
                    "confidence": 0.93,
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
                        r"\bmitre\b.*\b(att.ck|attack|mapping|tactic|technique)\b",
                        r"\b(tactic|technique)\b.*\b(mapping|distribution|breakdown)\b",
                    ],
                    "tool": "get_suricata_mitre_mapping",
                    "secondary": [],
                    "confidence": 0.95,
                    "description": "Suricata MITRE ATT&CK mapping"
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
                        r"\battack\b.*\b(sequence|stages|path)\b",
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
        # ================================================================
        self.KEYWORD_PATTERNS = {

            # System / manager tools (placed FIRST - fire before generic patterns)
            "remoted_stats": {
                "keywords": ["remoted", "remoted stats", "remoted statistics",
                             "agent communication stats", "agent communication statistics"],
                "exclude_keywords": [],
                "tool": "get_wazuh_remoted_stats",
                "secondary": [],
                "confidence": 0.92,
                "description": "Remoted (agent communication) statistics"
            },

            "manager_error_logs": {
                # FIX: Added typo-tolerant variants (maanger, manger, maneger, etc.)
                # Also added standalone "errors" since users say "please share manager errors"
                "keywords": [
                    "manager error", "manager errors", "manager error logs",
                    "error logs", "wazuh errors", "wazuh error",
                    "manager err", "manger error", "manger errors",
                    "maanger error", "maanger errors", "maneger error",
                    "show errors", "get errors", "share errors",
                    "view errors", "manager log error",
                ],
                "exclude_keywords": [],
                "tool": "get_wazuh_manager_error_logs",
                "secondary": [],
                "confidence": 0.92,
                "description": "Manager error logs"
            },

            "manager_logs_search": {
                "keywords": ["manager logs", "manager log", "wazuh logs", "wazuh log",
                             "search logs", "search manager"],
                "exclude_keywords": ["error"],
                "tool": "search_wazuh_manager_logs",
                "secondary": [],
                "confidence": 0.88,
                "description": "Search manager logs"
            },

            "cluster_health": {
                "keywords": [
                    "cluster health", "cluster status", "cluster nodes", "cluster info",
                    "check cluster", "wazuh cluster", "cluster health check",
                    "cluster overview", "cluster state",
                ],
                "exclude_keywords": [],
                "tool": "get_wazuh_cluster_health",
                "secondary": [],
                "confidence": 0.92,
                "description": "Cluster health"
            },

            "weekly_stats": {
                "keywords": ["weekly stats", "weekly statistics", "week stats"],
                "exclude_keywords": [],
                "tool": "get_wazuh_weekly_stats",
                "secondary": [],
                "confidence": 0.88,
                "description": "Weekly statistics"
            },

            "log_collector_stats": {
                "keywords": ["log collector", "log collector stats", "logcollector stats",
                             "log collector statistics", "logcollector statistics"],
                "exclude_keywords": [],
                "tool": "get_wazuh_log_collector_stats",
                "secondary": [],
                "confidence": 0.90,
                "description": "Log collector statistics"
            },

            "rules_summary": {
                "keywords": ["rules summary", "wazuh rules", "rules statistics", "rules stats"],
                "exclude_keywords": [],
                "tool": "get_wazuh_rules_summary",
                "secondary": [],
                "confidence": 0.88,
                "description": "Rules summary"
            },

            # FIX: search_security_events keyword pattern - placed HIGH in priority
            # MUST be before alert_general to prevent hijacking
            "search_events_keyword": {
                "keywords": [
                    "search event", "search events", "search security event", "search security events",
                    "virustotal event", "virustotal events", "virustotal alert", "virustotal alerts",
                    "fim event", "syscheck event", "rootcheck event", "ossec event",
                    "search for event", "find event", "find events", "lookup event",
                    "search alert", "search specific",
                ],
                "exclude_keywords": ["manager logs", "manager log", "wazuh log"],
                "tool": "search_security_events",
                "secondary": [],
                "confidence": 0.90,
                "description": "Search security events keyword query"
            },

            # Compliance and risk (before agent_general to avoid hijack)
            "compliance_check_keyword": {
                "keywords": ["compliance", "compliant", "complaince", "complience", "complianse",
                             "pci", "hipaa", "gdpr", "nist", "sox", "audit"],
                "exclude_keywords": [],
                "tool": "run_compliance_check",
                "secondary": [],
                "confidence": 0.85,
                "description": "Compliance check keyword query"
            },

            "risk_assessment_keyword": {
                "keywords": ["risk assessment", "riskassessment", "risk level", "risk score"],
                "exclude_keywords": [],
                "tool": "perform_risk_assessment",
                "secondary": [],
                "confidence": 0.85,
                "description": "Risk assessment keyword query"
            },

            "vulnerability_summary": {
                # MUST be placed before vulnerability_general to avoid being hijacked.
                # "summary" + any vuln keyword → get_wazuh_vulnerability_summary, NOT get_wazuh_vulnerabilities
                "keywords": [
                    "vulnerability summary", "vulnerabilities summary",
                    "vuln summary", "vulnerability overview", "vulnerabilities overview",
                    "cve summary", "vulnerability report", "vulnerabilities report",
                    "give me a vulnerability summary", "share vulnerability summary",
                    "show vulnerability summary", "vulnerability statistics",
                    "vulnerability stats",
                ],
                "exclude_keywords": [],
                "tool": "get_wazuh_vulnerability_summary",
                "secondary": [],
                "confidence": 0.95,
                "description": "Vulnerability summary/overview (NOT detailed CVE list)"
            },

            "vulnerability_general": {
                "keywords": ["vulnerability", "vulnerabilities", "cve", "vuln", "patch", "all vulnerabilities"],
                "exclude_keywords": ["critical"],  # Exclude critical to prevent filtering
                "tool": "get_wazuh_vulnerabilities",
                "args": {},  # No severity filter
                "secondary": [],
                "confidence": 0.82,
                "description": "General vulnerability query"
            },

            "alert_general": {
                "keywords": ["alert", "alerts", "detection", "incident", "triggered"],
                "exclude_keywords": ["search", "virustotal", "fim", "syscheck", "pattern", "patterns", "summary"],
                "tool": "get_wazuh_alerts",
                "secondary": [],
                "confidence": 0.82,
                "description": "General alert query"
            },

            "agent_ports": {
                "keywords": ["port", "ports", "open port", "open ports", "pots"],
                "exclude_keywords": ["vulnerability", "cve", "alert", "report", "security report"],
                "tool": "get_agent_ports",
                "secondary": [],
                "confidence": 0.82,
                "description": "Agent ports keyword query (requires agent_id)"
            },

            "agent_processes": {
                "keywords": ["process", "processes", "running process", "running processes"],
                "exclude_keywords": ["vulnerability", "cve", "alert"],
                "tool": "get_agent_processes",
                "secondary": [],
                "confidence": 0.82,
                "description": "Agent processes keyword query (requires agent_id)"
            },

            "agent_configuration": {
                "keywords": ["configuration", "config"],
                "exclude_keywords": ["vulnerability", "cve", "alert"],
                "tool": "get_agent_configuration",
                "secondary": [],
                "confidence": 0.80,
                "description": "Agent configuration keyword query (requires agent_id)"
            },

            "agent_health": {
                "keywords": ["agent health", "agent heartbeat", "agent keepalive",
                             "health of agent", "check agent", "verify agent"],
                "exclude_keywords": ["cluster", "wazuh health", "system health", "vulnerability", "cve", "alert"],
                "tool": "check_agent_health",
                "secondary": [],
                "confidence": 0.78,
                "description": "Agent health keyword query (requires agent_id)"
            },

            "agent_general": {
                "keywords": ["agent", "agents", "endpoint", "host", "server", "machine"],
                "exclude_keywords": ["active", "disconnected", "offline", "vulnerability", "port", "ports",
                                     "process", "processes", "configuration", "config", "health",
                                     "heartbeat", "keepalive", "compliance", "risk", "assessment",
                                     "check", "error", "errors", "remoted", "cluster", "logs", "log"],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "confidence": 0.75,
                "description": "General agent query"
            },

            "security_report": {
                "keywords": ["security report", "generate report", "daily report", "weekly report"],
                "exclude_keywords": [],
                "tool": "generate_security_report",
                "secondary": [],
                "confidence": 0.85,
                "description": "Security report generation"
            },

            "threat_analysis": {
                "keywords": ["threat", "threats", "ioc", "indicator", "malware"],
                "exclude_keywords": ["reputation", "check ioc", "ioc reputation", "detect ioc", "scan ioc"],
                "tool": "get_top_security_threats",
                "secondary": [],
                "confidence": 0.82,
                "description": "Threat analysis"
            },

            # R6: Enterprise SOC keyword patterns
            "rule_analysis_kw": {
                "keywords": ["rule trigger", "noisy rule", "rule frequency", "firing rule", "rule analysis"],
                "tool": "get_rule_trigger_analysis",
                "secondary": [],
                "confidence": 0.85,
                "description": "Rule trigger analysis"
            },
            "mitre_kw": {
                "keywords": ["mitre", "att&ck", "attack technique", "detection coverage", "mitre coverage"],
                "exclude_keywords": ["vulnerability", "cve"],
                "tool": "get_mitre_coverage",
                "secondary": [],
                "confidence": 0.85,
                "description": "MITRE ATT&CK coverage"
            },
            "timeline_kw": {
                "keywords": ["timeline", "alert trend", "alert spike", "hourly alert"],
                "exclude_keywords": ["vulnerability", "agent"],
                "tool": "get_alert_timeline",
                "secondary": [],
                "confidence": 0.82,
                "description": "Alert timeline"
            },
            "log_health_kw": {
                "keywords": ["log source", "ingestion", "silent agent", "source health", "log health"],
                "exclude_keywords": ["manager log", "search log"],
                "tool": "get_log_source_health",
                "secondary": [],
                "confidence": 0.85,
                "description": "Log source health"
            },
            "decoder_kw": {
                "keywords": ["decoder", "decoders", "parsing", "log parser"],
                "tool": "get_decoder_analysis",
                "secondary": [],
                "confidence": 0.82,
                "description": "Decoder analysis"
            },
            "fim_kw": {
                "keywords": ["fim", "file integrity", "syscheck", "file change", "file monitor"],
                "exclude_keywords": ["vulnerability", "cve"],
                "tool": "get_fim_events",
                "secondary": [],
                "confidence": 0.85,
                "description": "FIM events"
            },
            "sca_kw": {
                "keywords": ["sca", "configuration assessment", "cis benchmark", "hardening"],
                "exclude_keywords": ["compliance"],
                "tool": "get_sca_results",
                "secondary": [],
                "confidence": 0.85,
                "description": "SCA results"
            },
            "inventory_kw": {
                "keywords": ["inventory", "packages", "hardware info", "installed software", "network interface"],
                "exclude_keywords": ["vulnerability", "cve", "alert"],
                "tool": "get_agent_inventory",
                "secondary": [],
                "confidence": 0.80,
                "description": "Agent inventory"
            },
            "rootcheck_kw": {
                "keywords": ["rootcheck", "rootkit", "malware scan"],
                "tool": "get_rootcheck_results",
                "secondary": [],
                "confidence": 0.85,
                "description": "Rootcheck results"
            },
            "alert_vuln_corr_kw": {
                "keywords": ["alert vulnerability correlation", "cross correlate", "risk matrix"],
                "tool": "correlate_alerts_vulnerabilities",
                "secondary": [],
                "confidence": 0.85,
                "description": "Alert-vulnerability correlation"
            },
            "baseline_kw": {
                "keywords": ["baseline", "deviation", "anomaly detection", "normal activity"],
                "exclude_keywords": ["vulnerability", "cve", "agent"],
                "tool": "get_behavioral_baseline",
                "secondary": [],
                "confidence": 0.80,
                "description": "Behavioral baseline"
            },
            "decoder_gen_kw": {
                "keywords": ["generate decoder", "create decoder", "build decoder", "decoder for log", "parse this log", "custom decoder"],
                "exclude_keywords": ["list decoder", "show decoder", "decoder analysis", "active decoder"],
                "tool": "generate_wazuh_decoder",
                "secondary": [],
                "confidence": 0.85,
                "description": "Decoder generation keywords"
            }
        }

        # ================================================================
        # SURICATA KEYWORD PATTERNS (only when Suricata is enabled)
        # ================================================================
        if self.suricata_enabled:
            self.KEYWORD_PATTERNS.update({
                "suricata_general_kw": {
                    "keywords": ["suricata", "ids", "intrusion detection"],
                    "exclude_keywords": ["wazuh", "agent", "vulnerability", "cve"],
                    "tool": "get_suricata_alerts",
                    "secondary": [],
                    "confidence": 0.75,
                    "description": "General Suricata query"
                },
                "suricata_network_kw": {
                    "keywords": ["network analysis", "top talkers", "traffic analysis", "network traffic"],
                    "exclude_keywords": ["wazuh", "agent", "vulnerability"],
                    "tool": "get_suricata_network_analysis",
                    "secondary": [],
                    "confidence": 0.78,
                    "description": "Suricata network analysis"
                },
                "suricata_signature_kw": {
                    "keywords": ["signature", "signatures", "ids rule", "suricata rule"],
                    "exclude_keywords": ["wazuh", "decoder", "vulnerability"],
                    "tool": "get_suricata_top_signatures",
                    "secondary": [],
                    "confidence": 0.75,
                    "description": "Suricata signatures"
                },
                "suricata_attacker_kw": {
                    "keywords": ["attacker", "attackers", "attacking ip", "source attacker"],
                    "exclude_keywords": ["wazuh", "agent", "vulnerability"],
                    "tool": "get_suricata_top_attackers",
                    "secondary": [],
                    "confidence": 0.75,
                    "description": "Suricata top attackers"
                },
                "suricata_category_kw": {
                    "keywords": ["alert category", "detection category", "ids category"],
                    "exclude_keywords": ["wazuh", "vulnerability"],
                    "tool": "get_suricata_category_breakdown",
                    "secondary": [],
                    "confidence": 0.78,
                    "description": "Suricata categories"
                },
                "combined_alerts_kw": {
                    "keywords": ["combined alerts", "unified alerts", "all platform alerts", "both alerts"],
                    "tool": "get_suricata_alerts",
                    "secondary": ["get_wazuh_alerts"],
                    "correlation": "combined_alert_view",
                    "confidence": 0.80,
                    "description": "Combined platform alerts"
                },
                # Suricata Deep Visibility keyword patterns
                "suricata_http_kw": {
                    "keywords": ["http traffic", "web traffic", "http analysis", "url analysis", "http events"],
                    "exclude_keywords": ["wazuh", "agent", "vulnerability"],
                    "tool": "get_suricata_http_analysis",
                    "secondary": [],
                    "confidence": 0.78,
                    "description": "Suricata HTTP analysis"
                },
                "suricata_tls_kw": {
                    "keywords": ["tls", "ssl", "certificate", "ja3", "ja4", "tls fingerprint"],
                    "exclude_keywords": ["wazuh", "agent", "vulnerability"],
                    "tool": "get_suricata_tls_analysis",
                    "secondary": [],
                    "confidence": 0.78,
                    "description": "Suricata TLS analysis"
                },
                "suricata_mitre_kw": {
                    "keywords": ["mitre", "att&ck", "tactic", "technique", "mitre mapping"],
                    "exclude_keywords": ["wazuh"],
                    "tool": "get_suricata_mitre_mapping",
                    "secondary": [],
                    "confidence": 0.75,
                    "description": "Suricata MITRE mapping"
                },
                "suricata_suspicious_kw": {
                    "keywords": ["suspicious activity", "scanner detection", "anomaly detection", "suspicious traffic"],
                    "exclude_keywords": ["wazuh", "vulnerability"],
                    "tool": "get_suricata_suspicious_activity",
                    "secondary": [],
                    "confidence": 0.78,
                    "description": "Suspicious activity detection"
                },
            })

        # Pallas 3.2: Universal keyword patterns (always active)
        self.KEYWORD_PATTERNS.update({
            "ip_investigation_kw": {
                "keywords": ["investigate ip", "ip lookup", "ip pivot", "ip investigation", "lookup ip"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts", "get_wazuh_agents"] if self.suricata_enabled else ["get_wazuh_agents"],
                "correlation": "ip_investigation_pivot",
                "confidence": 0.78,
                "description": "IP investigation pivot"
            },
            "attack_chain_kw": {
                "keywords": ["attack chain", "kill chain", "attack progression", "attack stages", "attack path"],
                "exclude_keywords": [],
                "tool": "get_suricata_mitre_mapping" if self.suricata_enabled else "get_mitre_coverage",
                "secondary": ["get_wazuh_alerts"],
                "correlation": "attack_chain_analysis",
                "confidence": 0.78,
                "description": "Attack chain analysis"
            },
            "threat_priority_kw": {
                "keywords": ["top risks", "biggest threats", "priorities", "what to worry about", "top priorities", "biggest risk"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alert_summary",
                "secondary": ["get_wazuh_vulnerability_summary", "get_log_source_health"],
                "correlation": "unified_threat_summary",
                "confidence": 0.75,
                "description": "Prioritized threat summary"
            },
            "detection_gap_kw": {
                "keywords": ["detection gap", "coverage gap", "blind spots", "missing detection", "uncovered tactics"],
                "exclude_keywords": [],
                "tool": "get_mitre_coverage",
                "secondary": ["get_suricata_mitre_mapping"] if self.suricata_enabled else [],
                "correlation": "detection_coverage_gap" if self.suricata_enabled else None,
                "confidence": 0.78,
                "description": "Detection coverage gap"
            },
            "brute_force_kw": {
                "keywords": ["brute force", "password spray", "credential stuffing", "login attempts", "credential attack"],
                "exclude_keywords": [],
                "tool": "search_security_events",
                "secondary": [],
                "confidence": 0.80,
                "description": "Brute force detection"
            },
            "risk_ranking_kw": {
                "keywords": ["risk ranking", "risk score", "highest risk", "most vulnerable", "riskiest agents", "agent risk"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts"],
                "correlation": "top_risk_agents_composite",
                "confidence": 0.75,
                "description": "Agent risk ranking"
            },
            # --- Pallas 3.3: New keyword patterns ---
            "attacker_target_kw": {
                "keywords": ["attacker target", "attacker map", "attack map", "source destination map",
                             "who targets", "attacker to target"],
                "exclude_keywords": [],
                "tool": "get_suricata_alerts",
                "secondary": ["get_suricata_network_analysis"],
                "correlation": "suricata_attacker_target_map",
                "confidence": 0.78,
                "description": "Suricata attacker-target mapping"
            },
            "network_threat_profile_kw": {
                "keywords": ["network threat profile", "threat profile", "network risk profile",
                             "network threat overview"],
                "exclude_keywords": [],
                "tool": "get_suricata_network_analysis",
                "secondary": ["get_suricata_alert_summary"],
                "correlation": "suricata_network_threat_profile",
                "confidence": 0.75,
                "description": "Network threat profile"
            },
            "agents_ids_kw": {
                "keywords": ["agent ids alerts", "agent suricata", "endpoint network detection",
                             "agent network alert", "agent ids detection"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": ["get_suricata_alerts"],
                "correlation": "wazuh_agents_suricata_detections",
                "confidence": 0.75,
                "description": "Agent-Suricata detection mapping"
            },
            "vuln_trend_kw": {
                "keywords": ["vulnerability trend", "vuln trend", "severity breakdown",
                             "vulnerability by severity", "vuln by severity", "vuln distribution"],
                "exclude_keywords": ["alert", "suricata"],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": ["get_wazuh_agents"],
                "correlation": "vulnerability_trend_by_severity",
                "confidence": 0.75,
                "description": "Vulnerability trend by severity"
            },
            "security_posture_kw": {
                "keywords": ["security posture", "overall posture", "environment posture",
                             "security overview", "environment security"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": (["get_wazuh_vulnerabilities", "get_wazuh_alerts", "get_suricata_alerts"]
                              if self.suricata_enabled else ["get_wazuh_vulnerabilities", "get_wazuh_alerts"]),
                "correlation": "comprehensive_security_posture" if self.suricata_enabled else "agents_with_vulnerabilities",
                "confidence": 0.75,
                "description": "Security posture overview"
            },
            "combined_alerts_kw": {
                "keywords": ["all alerts together", "endpoint and network alerts", "combined alerts",
                             "unified alerts", "every alert", "all alerts both"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts"] if self.suricata_enabled else [],
                "correlation": "combined_alert_view" if self.suricata_enabled else None,
                "confidence": 0.72,
                "description": "Combined alert view"
            },
            # --- Pallas 3.4: Additional keyword patterns ---
            "windows_security_kw": {
                "keywords": ["windows event", "windows security event", "security event log",
                             "windows alert", "windows security alert", "windows log"],
                "exclude_keywords": ["manager", "wazuh log", "suricata"],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "windows OR rule.groups:windows"},
                "confidence": 0.78,
                "description": "Windows security event search"
            },
            "cloud_security_kw": {
                "keywords": ["cloud security", "cloud alerts", "cloud events",
                             "cloud security alerts", "cloud monitoring"],
                "exclude_keywords": ["suricata", "agent health"],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "aws OR azure OR office365 OR cloudtrail OR rule.groups:aws OR rule.groups:azure"},
                "confidence": 0.75,
                "description": "Cloud security event search"
            },
            "user_monitoring_kw": {
                "keywords": ["user activity", "access monitoring", "user monitoring",
                             "user access events", "user behavior", "access events"],
                "exclude_keywords": ["suricata", "vulnerability"],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "logon OR login OR user OR authentication OR access"},
                "confidence": 0.75,
                "description": "User activity / access monitoring search"
            },
            "agent_health_overview_kw": {
                "keywords": ["agent health overview", "agent summary", "agent inventory overview",
                             "fleet health", "fleet status", "agent fleet", "agent overview"],
                "exclude_keywords": ["vulnerability", "alert"],
                "tool": "get_wazuh_agents",
                "secondary": [],
                "correlation": "agent_health_overview",
                "confidence": 0.78,
                "description": "Agent health overview with OS/version grouping"
            },
            "insider_threat_kw": {
                "keywords": ["insider threat", "insider risk", "internal threat",
                             "data exfiltration", "data leak", "data theft",
                             "unauthorized access internal"],
                "exclude_keywords": ["suricata", "vulnerability"],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "data exfiltration OR unauthorized access OR insider OR privilege abuse"},
                "confidence": 0.78,
                "description": "Insider threat / data exfiltration search"
            },
            "anomaly_detection_kw": {
                "keywords": ["anomaly", "abnormal behavior", "unusual activity",
                             "behavior anomaly", "anomalous activity", "outlier detection"],
                "exclude_keywords": ["suricata", "baseline"],
                "tool": "search_security_events",
                "secondary": [],
                "args": {"query": "anomaly OR abnormal OR unusual activity"},
                "confidence": 0.72,
                "description": "Anomaly / abnormal behavior search"
            },
            # --- Pallas 4.1: Keyword fallbacks for correlation strategies ---
            "dynamic_risk_scoring_kw": {
                "keywords": ["risk score", "risk scoring", "dynamic risk", "composite risk",
                             "calculate risk", "risk assessment agents"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": (["get_wazuh_vulnerabilities", "get_wazuh_alerts", "get_suricata_alerts"]
                              if self.suricata_enabled else
                              ["get_wazuh_vulnerabilities", "get_wazuh_alerts"]),
                "correlation": "dynamic_risk_scoring",
                "confidence": 0.78,
                "description": "Dynamic composite risk scoring for agents"
            },
            "full_agent_investigation_kw": {
                "keywords": ["full investigation agent", "deep dive agent", "thorough investigation",
                             "complete investigation", "agent investigation"],
                "exclude_keywords": ["ip", "alert"],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_agent_ports", "get_agent_processes",
                              "get_wazuh_alerts", "get_fim_events"],
                "correlation": "full_agent_investigation",
                "confidence": 0.78,
                "description": "Full agent security investigation"
            },
            "alerts_with_vulns_kw": {
                "keywords": ["alerts vulnerabilities", "correlate alerts vulnerabilities",
                             "alert vuln correlation", "alerts with cves",
                             "match alerts vulnerabilities"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_agents"],
                "correlation": "alerts_with_vulnerabilities",
                "confidence": 0.78,
                "description": "Alert-vulnerability correlation"
            },
            "agent_posture_kw": {
                "keywords": ["agent posture", "security posture agent", "posture deep dive",
                             "endpoint posture", "agent security assessment"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_alerts",
                              "get_agent_ports", "get_agent_processes"],
                "correlation": "agent_posture_deep_dive",
                "confidence": 0.78,
                "description": "Agent security posture deep dive"
            },
            "fim_correlation_kw": {
                "keywords": ["fim correlation", "file change alert", "file integrity alert",
                             "syscheck correlation", "file modification alert"],
                "exclude_keywords": [],
                "tool": "get_fim_events",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents"],
                "correlation": "fim_alert_correlation",
                "confidence": 0.78,
                "description": "FIM-alert correlation"
            },
            "alert_enrichment_kw": {
                "keywords": ["alert context", "enrich alert", "alert enrichment",
                             "explain alert", "alert details context", "alert investigation"],
                "exclude_keywords": ["fim", "file"],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_wazuh_agents", "get_wazuh_vulnerabilities", "get_suricata_alerts"]
                              if self.suricata_enabled else
                              ["get_wazuh_agents", "get_wazuh_vulnerabilities"]),
                "correlation": "alert_context_enrichment",
                "confidence": 0.76,
                "description": "Alert context enrichment"
            },
            "agent_event_volume_kw": {
                "keywords": ["event volume", "event count agent", "alert count agent",
                             "busiest agents", "most alerts agent", "agent activity volume"],
                "exclude_keywords": [],
                "tool": "get_wazuh_agents",
                "secondary": ["get_wazuh_alerts"],
                "correlation": "agent_event_volume",
                "confidence": 0.78,
                "description": "Agent event volume analysis"
            },
            "behavioral_baseline_kw": {
                "keywords": ["behavioral baseline", "baseline behavior", "normal behavior",
                             "behavior pattern", "anomaly baseline", "establish baseline"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_wazuh_agents"],
                "correlation": "off_hours_anomaly_detection",
                "confidence": 0.76,
                "description": "Behavioral baseline and anomaly detection"
            },
            "vuln_exploit_correlation_kw": {
                "keywords": ["exploit correlation", "vulnerability exploit", "active exploit",
                             "exploitable vulnerability", "exploited cve"],
                "exclude_keywords": [],
                "tool": "get_wazuh_vulnerabilities",
                "secondary": ["get_wazuh_alerts", "get_wazuh_agents"],
                "correlation": "vulnerability_exploit_correlation",
                "confidence": 0.78,
                "description": "Vulnerability-exploit correlation"
            },
            "port_exposure_kw": {
                "keywords": ["port exposure", "exposed ports", "port risk", "open port risk",
                             "risky ports", "port vulnerability"],
                "exclude_keywords": [],
                "tool": "get_agent_ports",
                "secondary": ["get_wazuh_vulnerabilities", "get_wazuh_agents"],
                "correlation": "port_exposure_risk",
                "confidence": 0.78,
                "description": "Port exposure risk analysis"
            },
            "campaign_detection_kw": {
                "keywords": ["campaign detection", "multi asset attack", "coordinated attack",
                             "attack campaign", "widespread attack", "multi host attack"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": (["get_wazuh_agents", "get_suricata_alerts"]
                              if self.suricata_enabled else ["get_wazuh_agents"]),
                "correlation": "multi_asset_campaign_detection",
                "confidence": 0.76,
                "description": "Multi-asset attack campaign detection"
            },
            "alert_noise_kw": {
                "keywords": ["alert noise", "false positive", "noisy rules", "alert fatigue",
                             "suppress alerts", "tune alerts", "alert tuning"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alert_summary",
                "secondary": ["get_wazuh_alerts"],
                "correlation": "alert_noise_analysis",
                "confidence": 0.76,
                "description": "Alert noise and false positive analysis"
            },
            "temporal_attack_kw": {
                "keywords": ["attack sequence", "attack timeline", "attack chain",
                             "kill chain timeline", "temporal attack", "attack progression"],
                "exclude_keywords": [],
                "tool": "search_security_events",
                "secondary": ["get_wazuh_agents"] + (["get_suricata_alerts"] if self.suricata_enabled else []),
                "correlation": "temporal_attack_sequence",
                "args": {"query": "authentication OR login OR privilege escalation OR SeDebugPrivilege OR 4672", "limit": 200},
                "confidence": 0.76,
                "description": "Temporal attack sequence analysis"
            },
            "data_exfiltration_kw": {
                "keywords": ["data exfiltration detection", "exfiltration indicators",
                             "data leaving network", "outbound data anomaly"],
                "exclude_keywords": ["insider"],
                "tool": "search_security_events",
                "secondary": (["get_suricata_alerts", "get_wazuh_agents"]
                              if self.suricata_enabled else ["get_wazuh_agents"]),
                "correlation": "data_exfiltration_detection",
                "args": {"query": "data exfiltration OR large transfer OR outbound anomaly"},
                "confidence": 0.76,
                "description": "Data exfiltration detection"
            },
            "cross_platform_threat_kw": {
                "keywords": ["cross platform threat", "wazuh suricata correlation",
                             "endpoint network correlation", "cross tool correlation"],
                "exclude_keywords": [],
                "tool": "get_wazuh_alerts",
                "secondary": ["get_suricata_alerts", "get_wazuh_agents"] if self.suricata_enabled else ["get_wazuh_agents"],
                "correlation": "cross_platform_threat_correlation" if self.suricata_enabled else None,
                "confidence": 0.76,
                "description": "Cross-platform threat correlation"
            },
        })

        # ================================================================
        # ENTITY EXTRACTION PATTERNS
        # ================================================================
        self.ENTITY_PATTERNS = {
            "severity": r"\b(critical|high|medium|low)\b",
            "agent_id": r"\bagent[_-]?(\d{1,6})\b",
            "cve_id": r"\b(CVE-\d{4}-\d{4,7})\b",
            "status": r"\b(active|disconnected|never_connected|online|offline)\b",
            "time_range": r"\b(?:last|past)\s+(\d+)\s+(hour|day|week|month)s?\b",
            "framework": r"\b(PCI-DSS|PCI|HIPAA|GDPR|NIST|SOX)\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "level": r"\blevel\s+(\d+)\b",
            "rule_id": r"\brule\s*(?:id)?\s*[:#\-]?\s*(\d{4,6})\b",
            "mitre_id": r"\b(T\d{4}(?:\.\d{3})?)\b",
            "mitre_tactic": r"\b(initial access|execution|persistence|privilege escalation|defense evasion|credential access|discovery|lateral movement|collection|command and control|exfiltration|impact|reconnaissance|resource development)\b",
        }

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

        # Extract entities (used by all strategies)
        entities = self._extract_entities(query)

        # Merge carried entities from conversation history (follow-up support)
        if context and context.carried_entities:
            for key, value in context.carried_entities.items():
                if key not in entities:
                    entities[key] = value
                    logger.info(f"[CONTEXT] Carried entity from previous turn: {key}={value}")

        # Store entities for process_query() access (conversation memory)
        self._last_entities = entities

        logger.info(f"Extracted entities: {entities}")

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
            # Separate compound (has secondary tools) from simple matches
            compound_matches = [(n, c) for n, c in all_regex_matches if c.get("secondary")]
            simple_matches = [(n, c) for n, c in all_regex_matches if not c.get("secondary")]

            if compound_matches:
                # Prefer compound: pick highest confidence among compound matches
                best_name, best_config = max(compound_matches, key=lambda x: x[1].get("confidence", 0.9))
            else:
                # No compound match: pick highest confidence among simple matches
                best_name, best_config = max(simple_matches, key=lambda x: x[1].get("confidence", 0.9))

            logger.info(f"✅ HIGH-CONFIDENCE MATCH: {best_name} (from {len(all_regex_matches)} candidates)")

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

            self.stats["rule_regex"] += 1
            self.stats["total"] += 1

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

                self.stats["rule_keyword"] += 1
                self.stats["total"] += 1

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

            self.stats["llm"] += 1
            self.stats["total"] += 1

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

            fallback_plan = ToolPlan(
                primary_tool="get_wazuh_agents",
                secondary_tools=[],
                primary_args={"limit": 50},
                confidence=0.3,
                selection_method="fallback",
                reasoning=f"All strategies failed, using safe default. Error: {str(e)}",
                fallback_guidance=self._build_fallback_guidance(query)
            )

            self.stats["fallback"] += 1
            self.stats["total"] += 1

            return fallback_plan

    def _build_fallback_guidance(self, query: str) -> str:
        """
        Build intelligent guidance when no tool can be selected for the query.
        Explains the limitation and suggests alternative queries the user can try.
        """
        query_lower = query.lower()

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

        return (
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

        # PRIORITY 2: MITRE technique keyword mapping
        MITRE_TECHNIQUES = {
            "brute force": "Brute Force",
            "phishing": "Phishing",
            "spearphishing": "Spearphishing Attachment",
            "credential dumping": "OS Credential Dumping",
            "pass the hash": "Pass the Hash",
            "kerberoasting": "Kerberoasting",
            "golden ticket": "Steal or Forge Kerberos Tickets",
            "valid accounts": "Valid Accounts",
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

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from query"""

        entities = {}

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if entity_type == "time_range":
                    # Parse time range into standard format
                    amount, unit = matches[0]
                    entities[entity_type] = f"{amount}{unit[0]}"  # e.g., "24h"

                elif entity_type == "framework":
                    # Normalize framework name
                    framework = matches[0].upper()
                    if framework == "PCI":
                        framework = "PCI-DSS"
                    entities[entity_type] = framework

                elif entity_type == "status":
                    # Normalize status
                    status = matches[0].lower()
                    if status in ["online", "connected"]:
                        status = "active"
                    elif status == "offline":
                        status = "disconnected"
                    entities[entity_type] = status

                elif entity_type == "mitre_tactic":
                    # Title-case for Elasticsearch matching (e.g., "credential access" → "Credential Access")
                    entities[entity_type] = matches[0].strip().title()

                elif entity_type == "mitre_id":
                    # Uppercase T-code (e.g., "t1110" → "T1110")
                    entities[entity_type] = matches[0].strip().upper()

                else:
                    entities[entity_type] = matches[0]

        # Extra agent_id extraction (numeric IDs like 001) when query mentions 'agent'
        if "agent_id" not in entities:
            m_id = re.search(r"\bagent\s*(?:id)?\s*[:#\-]?\s*(\d{1,6})\b", query, re.IGNORECASE)
            if m_id:
                raw = m_id.group(1)
                entities["agent_id"] = raw.zfill(3) if len(raw) <= 3 else raw

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
            "get_wazuh_alerts":             1000,
            "get_wazuh_vulnerabilities":    1000,
            "get_wazuh_critical_vulnerabilities": 1000,
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

        prompt = f"""AVAILABLE TOOLS:
{tools_text}

EXTRACTED ENTITIES: {json.dumps(entities)}

USER QUERY: "{query}"

Select the most appropriate tools to answer this query.

Respond in JSON format:
{{
    "primary_tool": "exact_tool_name",
    "secondary_tools": ["tool_name"],
    "correlation_strategy": "strategy_name_or_null",
    "reasoning": "why these tools",
    "confidence": 0.8
}}

JSON:"""

        try:
            # LOW sensitivity — tool selection only sees query text + tool names, no client data
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
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
            return self.stats

        return {
            **self.stats,
            "percentages": {
                "rule_regex": (self.stats["rule_regex"] / total) * 100,
                "rule_keyword": (self.stats["rule_keyword"] / total) * 100,
                "llm": (self.stats["llm"] / total) * 100,
                "fallback": (self.stats["fallback"] / total) * 100
            }
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

        # Validate arguments
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

        # Pallas 4.1: Response cache for expensive multi-tool queries
        # Key = normalized query string, Value = {"response": str, "turn_data": dict, "timestamp": float}
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 180  # 3 minutes — fresh enough for SOC, avoids redundant multi-tool executions
        self._cache_max_size = 30  # Max cached entries

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

    async def process_query(self, user_query: str, context: Optional[QueryContext] = None) -> str:
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

        # Pallas 4.1: Check response cache for identical recent queries
        cache_key = user_query.strip().lower()
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
            logger.info(f"[DEBUG] Selected primary_tool={plan.primary_tool}, args={plan.primary_args}")
            logger.info(f"[DEBUG] Secondary tools={plan.secondary_tools}, secondary_args={plan.secondary_args}")
            logger.info(f"[DEBUG] Correlation strategy={plan.correlation_strategy}, requires_correlation={plan.requires_correlation}")
            logger.info(f"   Confidence: {plan.confidence:.2f}, Method: {plan.selection_method}")

            # Guardrail: Very low confidence — return intelligent guidance
            if plan.confidence < 0.3 and plan.selection_method in ("llm", "fallback"):
                if plan.fallback_guidance:
                    return plan.fallback_guidance
                return self.selector._build_fallback_guidance(user_query)

            # Guardrail: Agent deep-dive tools require a specific agent_id
            if plan.primary_tool in ("get_agent_ports", "get_agent_processes", "get_agent_configuration", "check_agent_health"):
                if not plan.primary_args.get("agent_id"):
                    return (
                        "I can do that, but I need the **agent ID** (or exact agent name).\n\n"
                        "Examples:\n"
                        "- `Show open ports for agent 001`\n"
                        "- `List running processes for agent 015`\n"
                        "- `Get configuration for agent 001`\n"
                        "- `Check health for agent 001`"
                    )


            # Step 2: Execute primary tool
            primary_result = await self.executor.execute(
                plan.primary_tool,
                plan.primary_args
            )

            if not primary_result.success:
                return self._format_error_response(primary_result, plan)

            # Step 3: Execute secondary tools (parallel)
            secondary_results = []
            if plan.secondary_tools:
                tool_specs = list(zip(plan.secondary_tools, plan.secondary_args))
                secondary_results = await self.executor.execute_batch(tool_specs)

                # Log failures
                failed = [r for r in secondary_results if not r.success]
                if failed:
                    logger.warning(f"⚠️ {len(failed)} secondary tool(s) failed")

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

            # Step 5: Format response FIRST (before LLM — so LLM gets clean, filtered data)
            # v3: _format_response now returns (markdown, blocks) tuple
            response, response_blocks = self._format_response(
                plan=plan,
                primary_result=primary_result,
                secondary_results=secondary_results,
                correlated_data=correlated_data,
                llm_analysis=None,
                user_query=user_query
            )

            # Step 6: Generate SOC insight (query intro + analysis) using formatted response
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

            # v3: Metadata footer removed — debug info is in `metadata` field.
            logger.info(
                f"[RESPONSE] {plan.selection_method} ({plan.confidence:.0%}) "
                f"tool={primary_result.tool_name} ({primary_result.execution_time:.2f}s) "
                f"total={total_time:.2f}s"
            )

            # Store turn data for conversation memory + v2 metadata (server.py picks this up)
            try:
                normalized = self._normalize_tool_payload(primary_result.data)
                result_data = normalized.get("affected_items", [])
            except Exception:
                result_data = []

            # Build suggestion list (without markdown formatting) for v2 response
            v2_suggestions = self._get_suggestion_list(plan, correlated_data)

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
                    "strategy": plan.correlation_strategy or plan.primary_tool,
                    "primary_tool": plan.primary_tool,
                    "secondary_tools": list(plan.secondary_tools) if plan.secondary_tools else [],
                    "confidence": plan.confidence,
                    "execution_time_ms": int(total_time * 1000),
                    "correlation_type": correlated_data.correlation_type if correlated_data else None,
                    "selection_method": plan.selection_method,
                    # v3: enriched metadata for frontend header bar
                    "results_count": len(result_data),
                    "query": user_query,
                    "tools": {"primary": plan.primary_tool},
                    "execution_time": total_time,
                },
                "suggestions": v2_suggestions,
                # v3: typed blocks for direct frontend rendering
                "blocks": response_blocks if response_blocks else None,
            }

            # Pallas 4.1: Cache the response for identical follow-up queries
            if plan.confidence >= 0.5:  # Only cache confident results
                # Evict oldest if at capacity
                if len(self._response_cache) >= self._cache_max_size:
                    oldest_key = min(self._response_cache, key=lambda k: self._response_cache[k]["timestamp"])
                    del self._response_cache[oldest_key]
                self._response_cache[cache_key] = {
                    "response": final,
                    "turn_data": self._last_turn_data,
                    "timestamp": time.time(),
                }

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
        # Use formatted response (clean, filtered) as data summary
        # Higher limit for multi-tool correlation (more data sections to cover)
        summary_limit = 2500 if (plan.requires_correlation and plan.secondary_tools) else 1500
        data_summary = formatted_response[:summary_limit]
        if len(formatted_response) > summary_limit:
            data_summary += "\n... (truncated)"

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
        """
        v3: Query/Tools/Results metadata is now sent via the `metadata` field
        in the WebSocket response. The frontend header bar renders it — no need
        to embed it in the markdown body. Return empty string.
        """
        return ""

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
        """

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

        # Override with correlation ONLY if it has data (NEVER return empty from correlation)
        if correlated_data and hasattr(correlated_data, "correlated_data"):
            if correlated_data.correlated_data and len(correlated_data.correlated_data) > 0:
                logger.info(f"[FORMATTER] Using correlation: {len(correlated_data.correlated_data)} items")
                data = correlated_data.correlated_data
                summary = correlated_data.summary
            else:
                logger.info(f"[FORMATTER] Correlation empty → fallback to primary data: {len(data)} items")
                # Clear correlation strategy so _route_formatter uses tool-based routing
                # instead of sending raw primary data to a correlation formatter
                plan.correlation_strategy = None

        logger.info(f"[FORMATTER] Final data count before formatting: {len(data)} items")

        # Graceful empty data handling
        if not data or len(data) == 0:
            # Some tools return text content instead of affected_items (stats, logs, decoder)
            # Check if there's raw text content before declaring empty
            raw_text = self._extract_raw_text(tool_response, "")
            if not raw_text or len(raw_text.strip()) < 10:
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

        blocks = []
        try:
            result = self._route_formatter(
                tool_name=tool_name,
                tool_response=tool_response,
                data=data,
                summary=summary,
                plan=plan,
                normalized=normalized
            )
            # v3 hybrid: formatters can return (markdown, blocks) tuple or plain string
            if isinstance(result, tuple) and len(result) == 2:
                response, blocks = result
            else:
                response = result
        except Exception as fmt_err:
            logger.error(f"[FORMATTER] ❌ Formatter failed for {tool_name}: {fmt_err}", exc_info=True)
            response = self._safe_fallback_format(tool_name, tool_response, data, fmt_err)

        # Prepend correlation header if this was a correlated multi-tool response
        if correlation_header:
            response = correlation_header + "\n" + response

        # v3 auto-block extraction: if formatter didn't return typed blocks,
        # auto-generate them from raw data so every query gets graphical rendering.
        if not blocks:
            try:
                blocks = self._auto_extract_blocks(tool_name, data, summary, plan)
            except Exception as e:
                logger.warning(f"[BLOCKS] Auto-extraction failed for {tool_name}: {e}")
                blocks = []

        return response, blocks

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

        # Pallas 3.2: Universal correlation formatters
        # Pallas 3.3: Extended to cover ALL 36 strategies (was 12, now 35 + 1 special-cased)
        correlation_formatter_map = {
            # --- Pallas 3.2 originals (12) ---
            "ip_investigation_pivot": "format_ip_investigation_response",
            "full_agent_investigation": "format_full_agent_investigation_response",
            "cross_platform_ip_correlation": "format_cross_platform_ip_response",
            "unified_scanning_detection": "format_unified_scan_detection_response",
            "attack_chain_analysis": "format_attack_chain_response",
            "unified_threat_summary": "format_unified_threat_summary_response",
            "vulnerability_exploit_correlation": "format_vulnerability_exploit_response",
            "detection_coverage_gap": "format_detection_coverage_gap_response",
            "fim_alert_correlation": "format_fim_alert_correlation_response",
            "port_exposure_risk": "format_port_exposure_risk_response",
            "top_risk_agents_composite": "format_top_risk_agents_response",
            "alert_context_enrichment": "format_alert_enrichment_response",
            # --- Pallas 3.3: Wazuh correlation formatters (12) ---
            "vulnerability_with_agents": "format_vulnerability_with_agents_response",
            "agents_with_vulnerabilities": "format_agents_with_vulnerabilities_response",
            "disconnected_agents_with_critical_vulns": "format_disconnected_critical_vulns_response",
            "top_agents_by_vuln_count": "format_top_agents_vuln_count_response",
            "active_agents_vulns_over_threshold": "format_active_agents_threshold_response",
            "active_agents_with_high_vulns": "format_active_agents_high_vulns_response",
            "compare_vulns_active_vs_disconnected": "format_compare_vulns_status_response",
            "alerts_with_agents": "format_alerts_with_agents_response",
            "alerts_with_vulnerabilities": "format_alerts_with_vulns_response",
            "rule_mitre_with_agents": "format_rule_mitre_agents_response",
            "fim_with_agent_posture": "format_fim_agent_posture_response",
            "vulnerability_trend_by_severity": "format_vuln_trend_severity_response",
            # --- Pallas 3.3: Suricata correlation formatters (6) ---
            "suricata_attacker_target_map": "format_suricata_attacker_target_response",
            "suricata_signature_severity_analysis": "format_suricata_sig_severity_response",
            "suricata_network_threat_profile": "format_suricata_network_threat_response",
            "suricata_http_alert_context": "format_suricata_http_alert_response",
            "suricata_tls_threat_detection": "format_suricata_tls_threat_response",
            "suricata_recon_detection": "format_suricata_recon_response",
            # --- Pallas 3.3: Cross-platform correlation formatters (5) ---
            "combined_alert_view": "format_combined_alert_view_response",
            "wazuh_agents_suricata_detections": "format_agents_suricata_detections_response",
            "ioc_with_suricata_context": "format_ioc_suricata_context_response",
            "comprehensive_security_posture": "format_comprehensive_posture_response",
            "wazuh_suricata_mitre_unified": "format_unified_mitre_response",
            # --- Pallas 3.4: Agent health correlation formatters (2) ---
            "agent_health_overview": "format_agent_health_overview_response",
            "agent_event_volume": "format_agent_event_volume_response",
            # --- Pallas 3.5: Advanced SOC correlation formatters (7) ---
            "temporal_attack_sequence": "format_temporal_attack_sequence_response",
            "dynamic_risk_scoring": "format_dynamic_risk_scoring_response",
            "cross_platform_threat_correlation": "format_cross_platform_threat_response",
            "data_exfiltration_detection": "format_data_exfiltration_response",
            "off_hours_anomaly_detection": "format_off_hours_anomaly_response",
            "alert_noise_analysis": "format_alert_noise_analysis_response",
            "multi_asset_campaign_detection": "format_multi_asset_campaign_response",
        }
        if plan.correlation_strategy in correlation_formatter_map:
            method_name = correlation_formatter_map[plan.correlation_strategy]
            formatter_method = getattr(self.formatter, method_name, None)
            if formatter_method:
                logger.info(f"[FORMATTER] -> {plan.correlation_strategy} -> {method_name}")
                return formatter_method(
                    {"correlated_data": data, "summary": summary},
                    summary=summary
                )

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
            return self.formatter.format_vulnerability_response(
                correlated_data=data,
                summary=summary,
                include_remediation=True
            )

        # Agent tools
        elif tool_name in ["get_wazuh_agents", "get_wazuh_running_agents"]:
            logger.info(f"[FORMATTER] → AGENT formatter")
            status_filter = plan.primary_args.get("status")
            # get_wazuh_running_agents only returns active agents — force the filter
            if not status_filter and tool_name == "get_wazuh_running_agents":
                status_filter = "active"
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
            return self.formatter.format_alert_response(
                correlated_data=data,
                summary=summary
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
            logger.info(f"[FORMATTER] → AGENT CONFIG raw")
            return self._extract_raw_text(tool_response, "⚠️ No config data")

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

        elif tool_name in ["get_suricata_critical_alerts", "get_suricata_high_alerts"]:
            logger.info(f"[FORMATTER] → SURICATA CRITICAL/HIGH ALERTS formatter")
            raw_text = self._extract_raw_text(tool_response, "No Suricata alert data")
            return self.formatter.format_suricata_critical_alerts_response(raw_text)

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

    def _auto_extract_blocks(self, tool_name: str, data: list, summary: dict, plan) -> list:
        """
        v3 auto-block extractor — universal safety net for ALL formatters.
        When a formatter hasn't been manually migrated to emit typed blocks,
        this method auto-generates basic blocks from the raw data.
        """
        from .block_builder import BlockBuilder
        bb = BlockBuilder()

        if not data or not isinstance(data, list):
            return []

        total = len(data)
        tool_lower = tool_name.lower()
        sample = data[0] if data else {}

        # Detect data shape
        has_status = 'status' in sample
        has_agent_id = any(k in sample for k in ('id', 'agent_id'))
        has_cve = any(k in sample for k in ('cve', 'cve_id', 'name'))
        has_timestamp = any(k in sample for k in ('timestamp', 'time', '@timestamp'))
        has_os = isinstance(sample.get('os'), dict)

        # Severity counts
        sev_counts = {}
        for item in data:
            sev = None
            if isinstance(item.get('rule'), dict):
                lvl = item['rule'].get('level', 0)
                try:
                    lvl = int(lvl)
                except (ValueError, TypeError):
                    lvl = 0
                sev = 'critical' if lvl >= 12 else 'high' if lvl >= 8 else 'medium' if lvl >= 4 else 'low'
            elif 'severity' in item:
                sev = str(item['severity']).lower()
            if sev and sev in ('critical', 'high', 'medium', 'low', 'info'):
                sev_counts[sev] = sev_counts.get(sev, 0) + 1

        # KPI grid (always)
        kpi_items = [bb.kpi_item("Total Records", total)]
        for sev_key in ['critical', 'high', 'medium', 'low']:
            if sev_key in sev_counts:
                kpi_items.append(bb.kpi_item(sev_key.title(), sev_counts[sev_key]))
        bb.kpi_grid(None, kpi_items)

        # Severity breakdown
        if sev_counts:
            bb.severity_breakdown([
                bb.sev_segment(k.title(), v, k) for k, v in sev_counts.items()
                if k in ('critical', 'high', 'medium', 'low', 'info')
            ])

        # Agent table
        if has_agent_id and has_status and ('agent' in tool_lower or 'health' in tool_lower):
            status_counts = {}
            for item in data:
                st = str(item.get('status', 'unknown')).lower()
                status_counts[st] = status_counts.get(st, 0) + 1
            active = status_counts.get('active', 0)
            disconnected = status_counts.get('disconnected', 0)
            health_pct = (active / total * 100) if total > 0 else 0
            bb.health_status("Agent Health", round(health_pct, 1), total, active, disconnected,
                             status_counts.get('never_connected', 0))
            agent_rows = []
            for a in data[:50]:
                os_info = a.get('os', {})
                os_name = f"{os_info.get('name', '')} {os_info.get('version', '')}".strip() if isinstance(os_info, dict) else ''
                agent_rows.append({
                    "agent_id": str(a.get('id', a.get('agent_id', ''))),
                    "name": a.get('name', ''),
                    "ip": a.get('ip', ''),
                    "os": os_name or 'N/A',
                    "version": str(a.get('version', '')).replace('Wazuh v', ''),
                    "status": a.get('status', ''),
                    "last_keep_alive": str(a.get('lastKeepAlive', a.get('last_keep_alive', '')))[:19].replace('T', ' '),
                })
            if agent_rows:
                bb.agent_table(agent_rows)

        # Vulnerability table
        elif has_cve and ('vuln' in tool_lower or 'cve' in tool_lower):
            vuln_rows = []
            for v in data[:50]:
                vuln_rows.append({
                    "cve": v.get('cve', v.get('cve_id', v.get('name', ''))),
                    "severity": str(v.get('severity', v.get('condition', 'unknown'))).lower(),
                    "package": v.get('package', {}).get('name', '') if isinstance(v.get('package'), dict) else str(v.get('package', '')),
                    "version": v.get('package', {}).get('version', '') if isinstance(v.get('package'), dict) else str(v.get('version', '')),
                    "description": v.get('description', v.get('title', '')),
                    "agent": str(v.get('agent_id', v.get('agent', ''))),
                })
            if vuln_rows:
                bb.vulnerability_table(vuln_rows)

        # Alert timeline
        elif has_timestamp and ('alert' in tool_lower or 'event' in tool_lower or 'threat' in tool_lower):
            events = []
            ip_counts = {}
            for item in data[:30]:
                rule = item.get('rule', {}) if isinstance(item.get('rule'), dict) else {}
                ts = item.get('timestamp', item.get('@timestamp', item.get('time', '')))
                lvl = rule.get('level', item.get('level', 0))
                try:
                    lvl = int(lvl)
                except (ValueError, TypeError):
                    lvl = 0
                sev = 'critical' if lvl >= 12 else 'high' if lvl >= 8 else 'medium' if lvl >= 4 else 'low'
                events.append({
                    "timestamp": str(ts)[:19].replace('T', ' ') if ts else '',
                    "severity": sev,
                    "title": (
                        rule.get('description')
                        or item.get('rule_description')
                        or item.get('description')
                        or item.get('message')
                        or item.get('full_log', '')[:100]
                        or f"Rule {rule.get('id', 'unknown')} · Level {lvl}"
                    ),
                    "detail": ' · '.join(filter(None, [
                        f"rule {rule.get('id', '')}" if rule.get('id') else '',
                        f"level {lvl}" if lvl else '',
                        f"agent {item.get('agent', {}).get('name', item.get('agent', {}).get('id', ''))}" if isinstance(item.get('agent'), dict) and item.get('agent', {}).get('id') else '',
                        item.get('location', ''),
                    ])),
                    "rule_id": str(rule.get('id', '')),
                    "agent": str(item.get('agent', {}).get('id', '')) if isinstance(item.get('agent'), dict) else '',
                })
                src = item.get('data', {}).get('srcip', '') if isinstance(item.get('data'), dict) else ''
                if src:
                    ip_counts[src] = ip_counts.get(src, 0) + 1
            if events:
                bb.timeline(events, "Alert Timeline")
            if len(ip_counts) >= 2:
                top_ips = sorted(ip_counts.items(), key=lambda x: -x[1])[:15]
                bb.top_attackers("sources", [{"ip": ip, "count": c} for ip, c in top_ips])

        # Suricata
        elif 'suricata' in tool_lower:
            events = []
            for item in data[:30]:
                alert_data = item.get('alert', {}) if isinstance(item.get('alert'), dict) else {}
                ts = item.get('timestamp', item.get('@timestamp', ''))
                sev_num = alert_data.get('severity', 3)
                try:
                    sev_num = int(sev_num)
                except (ValueError, TypeError):
                    sev_num = 3
                sev = 'critical' if sev_num <= 1 else 'high' if sev_num == 2 else 'medium' if sev_num == 3 else 'low'
                events.append({
                    "timestamp": str(ts)[:19].replace('T', ' ') if ts else '',
                    "severity": sev,
                    "title": alert_data.get('signature', item.get('description', '')),
                    "detail": f"sid:{alert_data.get('signature_id', '')} cat:{alert_data.get('category', '')}",
                    "rule_id": str(alert_data.get('signature_id', '')),
                })
            if events:
                bb.timeline(events, "Suricata Alerts")
            if 'flow' in tool_lower or 'network' in tool_lower or 'traffic' in tool_lower:
                flows = []
                for item in data[:20]:
                    f_data = {"src_ip": item.get('src_ip', ''), "dst_ip": item.get('dest_ip', item.get('dst_ip', '')),
                              "proto": item.get('proto', ''), "action": item.get('event_type', '')}
                    if f_data['src_ip'] and f_data['dst_ip']:
                        flows.append(f_data)
                if flows:
                    bb.flow_diagram(flows, "Network Flows")

        # MITRE
        elif 'mitre' in tool_lower:
            tactics = {}
            for item in data:
                tactic_name = item.get('tactic', item.get('tactic_name', 'Uncategorized'))
                tech_id = item.get('technique_id', item.get('id', ''))
                tech_name = item.get('technique_name', item.get('name', ''))
                count = int(item.get('count', item.get('alerts', 1)))
                if tactic_name not in tactics:
                    tactics[tactic_name] = {"id": item.get('tactic_id', ''), "name": tactic_name, "techniques": []}
                if tech_id:
                    tactics[tactic_name]["techniques"].append({"id": tech_id, "name": tech_name, "count": count})
            if tactics:
                bb.mitre_matrix(list(tactics.values()),
                                sum(len(t["techniques"]) for t in tactics.values()),
                                sum(sum(tech["count"] for tech in t["techniques"]) for t in tactics.values()))

        # OS distribution
        if has_os and ('health' in tool_lower or 'inventory' in tool_lower or 'agent' in tool_lower):
            os_counts = {}
            for a in data:
                os_info = a.get('os', {})
                if isinstance(os_info, dict):
                    os_name = os_info.get('name', 'Unknown')
                    os_counts[os_name] = os_counts.get(os_name, 0) + 1
            if len(os_counts) >= 2:
                colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#64748b']
                segments = [{"label": os, "key": os.lower().replace(' ', '_'), "value": count,
                             "color": colors[i % len(colors)]}
                            for i, (os, count) in enumerate(sorted(os_counts.items(), key=lambda x: -x[1]))]
                bb.severity_breakdown(segments)

        return bb.to_list()

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
        """Format error response"""

        return f"""## ⚠️ Tool Execution Error

**Tool**: {result.tool_name}
**Arguments**: {result.arguments}
**Error**: {result.error}

**Selection Details**:
- Method: {plan.selection_method}
- Confidence: {plan.confidence:.0%}
- Reasoning: {plan.reasoning}

**Troubleshooting**:
1. Check Wazuh server connectivity
2. Verify Wazuh Indexer is configured (for vulnerability/alert data)
3. Ensure tool arguments are valid
4. Check server logs for details

Please try:
- Rephrasing your query
- Using a simpler query
- Checking system status: "validate wazuh connection"

If the issue persists, contact support."""

    def _format_exception_response(self, exception: Exception) -> str:
        """Format exception response"""

        return f"""## ❌ System Error

An unexpected error occurred while processing your query.

**Error**: {str(exception)}

**What to try**:
1. Simplify your query
2. Try a different phrasing
3. Check system connectivity
4. Review server logs

**Examples of working queries**:
- "Show me critical vulnerabilities"
- "List all active agents"
- "Get security alerts from last 24 hours"
- "Search virustotal events in last 24 hours"

If the problem continues, please contact support."""

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

    # Initialize Hybrid LLM Router (Local + Cloud)
    local_host = os.getenv("LOCAL_LLM_HOST", "http://localhost:11434")
    local_model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:14b-instruct")
    cloud_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    cloud_model = os.getenv("LLM_MODEL", "moonshotai/kimi-k2.5")
    cloud_key = os.getenv("LLM_API_KEY", "")

    local_llm = LocalLLMClient(host=local_host, model=local_model)
    cloud_llm = LLMClient(base_url=cloud_url, model=cloud_model, api_key=cloud_key)
    llm_router = LLMRouter(local_llm, cloud_llm)
    await llm_router.initialize()

    # Create orchestrator — passes router as the LLM client
    orchestrator = QueryOrchestrator(
        mcp_tool_executor=mcp_tool_executor,
        ollama_client=llm_router,
        suricata_enabled=suricata_enabled
    )

    logger.info("QueryOrchestrator initialized with hybrid LLM routing")
    logger.info(f"   LOCAL: {local_model} at {local_host} (sensitive data)")
    logger.info(f"   CLOUD: {cloud_model} at {cloud_url} (orchestration)")
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