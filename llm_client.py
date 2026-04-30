"""
LLM Client — Cloud-Only Architecture (local LLM removed 2026-04-15)
- LLMClient: Cloud (DeepSeek V3.1-Terminus via HuggingFace) — handles ALL LLM workload
- LocalLLMClient: DEPRECATED stub, forwards to cloud (kept for backward compat)
- LLMRouter: Cloud-only router. Sensitivity labels retained for audit logging only.

WARNING: Client-sensitive SOC data (IPs, CVEs, agent names, forensic narratives)
is now sent to the cloud provider. There is no on-premise fallback.
"""

import os
import httpx
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
# SHARED: System prompts and analysis methods (used by both clients)
# ════════════════════════════════════════════════════════

SOC_SYSTEM_PROMPT = """You are a senior SOC analyst with 15+ years of experience.
Your role is to analyze security data and provide actionable insights.

CRITICAL RULES:
1. ONLY reference data explicitly provided in the prompt — quote exact values from the data
2. NEVER invent or fabricate CVE IDs, IP addresses, agent names, or any identifiers not in the data
3. Be concise - 2-3 sentences maximum
4. Focus on business impact and urgency
5. Provide specific, actionable recommendations
6. If the data shows zero results or is empty, state that clearly"""

ANALYSIS_PROMPTS = {
    "vulnerability": """Analyze this vulnerability data and provide a concise summary:

{data}

Provide:
1. Overall risk assessment (1-2 sentences)
2. Most critical concern
3. Immediate action required

Keep response under 150 words.""",
    "alert": """Analyze these security alerts:

{data}

Provide:
1. Pattern assessment (1-2 sentences)
2. Potential attack indicators
3. Recommended investigation steps

Keep response under 150 words.""",
    "agent": """Analyze this agent status data:

{data}

Provide:
1. Infrastructure health assessment
2. Critical issues requiring attention
3. Recommended actions

Keep response under 150 words.""",
    "port": """Analyze these open ports for security risks:

{data}

Provide:
1. Exposure assessment — which ports pose the highest risk
2. Suspicious ports — any known malicious or unnecessary services
3. Firewall recommendations — what should be restricted

Keep response under 150 words.""",
    "process": """Analyze these running processes for security concerns:

{data}

Provide:
1. Suspicious process assessment — any known malware patterns or anomalies
2. Resource abuse indicators — crypto mining, excessive CPU/memory
3. Recommended actions — processes to investigate or terminate

Keep response under 150 words.""",
    "statistics": """Analyze these system statistics for operational health:

{data}

Provide:
1. Queue and throughput health — any bottlenecks or message loss
2. Capacity concerns — is the system under strain
3. Recommended actions — scaling or configuration changes needed

Keep response under 150 words.""",
    "general": """Analyze this security data:

{data}

Provide a concise summary highlighting key findings and recommended actions.
Keep response under 150 words.""",
}

SOC_INSIGHT_SYSTEM_PROMPT = """You are a senior SOC analyst providing executive-level security briefings.

CRITICAL RULES:
1. ONLY reference data explicitly provided below — NEVER invent metrics, IPs, CVE IDs, or agent names
2. Be precise and actionable — quote exact numbers and identifiers from the data
3. If the data is empty or shows zero results, say so clearly — do NOT make up findings
4. Do NOT hallucinate or fabricate any information not present in the data
5. When referencing specific CVEs, agent IDs, or IPs, they MUST appear verbatim in the data above"""

FOCUS_MAP = {
    "vulnerability": "Focus on CVE severity distribution, affected agent count, patch urgency, exploit availability.",
    "alert": "Focus on alert volume trends, rule concentration (is one rule dominating?), attack chain indicators.",
    "agent": "Focus on active/disconnected ratio, coverage gaps, version drift, infrastructure health.",
    "port": "Focus on exposed services, known-bad ports, lateral movement risk.",
    "process": "Focus on suspicious binaries, resource abuse, persistence mechanisms.",
    "statistics": "Focus on throughput health, queue saturation, capacity trends.",
    "correlation_vulnerability_with_agents": "Focus on which CVEs affect the most agents, severity hotspots, patch prioritization by agent exposure.",
    "correlation_agents_with_vulnerabilities": "Focus on which agents carry the highest vulnerability load, risk ranking, remediation priority order.",
    "correlation_alerts_with_agents": "Focus on which agents generate the most alerts, rule noise vs real threats, agent-specific investigation priorities.",
    "correlation_alerts_with_vulnerabilities": "Focus on agents appearing in BOTH alert and vulnerability data, combined risk scoring, attack surface overlap.",
    "correlation_agent_posture_deep_dive": "Focus on holistic agent security: vulnerability exposure, suspicious ports/processes, combined risk profile, immediate containment recommendations.",
    "correlation_top_agents_by_vuln_count": "Focus on the ranking of most vulnerable agents, whether top agents are internet-facing, remediation priority.",
    "correlation_disconnected_agents_with_critical_vulns": "Focus on the danger of unpatched offline agents, risk of reconnection without patching, immediate isolation recommendations.",
    "correlation_active_agents_with_high_vulns": "Focus on active agents with high/critical vulnerabilities, immediate patch requirements, exposure window.",
    "correlation_compare_vulns_active_vs_disconnected": "Focus on the vulnerability distribution gap between active and disconnected agents, patch deployment effectiveness.",
    "correlation_fim_with_agent_posture": "Focus on file integrity changes correlated with agent status, suspicious modification patterns, insider threat indicators.",
    "correlation_rule_mitre_with_agents": "Focus on MITRE ATT&CK technique coverage, detection gaps, which agents trigger which techniques.",
    "correlation": "Focus on cross-data relationships: which agents appear in multiple risk categories, combined risk assessment, prioritization.",
}


def _build_analysis_prompt(data_summary: str, analysis_type: str) -> str:
    template = ANALYSIS_PROMPTS.get(analysis_type, ANALYSIS_PROMPTS["general"])
    return template.format(data=data_summary)


def _build_soc_insight_prompt(user_query: str, data_summary: str, analysis_type: str, conversation_context: str) -> str:
    focus = FOCUS_MAP.get(analysis_type, "Focus on the most relevant security insights from this data.")
    context_block = ""
    if conversation_context:
        context_block = f"CONVERSATION CONTEXT (previous queries in this session):\n{conversation_context}\n\n"

    return f"""The SOC operator asked: "{user_query}"

{context_block}Retrieved data summary:
{data_summary}

{focus}

Provide exactly TWO sections with these markers:

=== QUERY INTERPRETATION ===
1-2 sentences: What the operator asked for, and a brief summary of key findings.
Be conversational but professional. Reference actual numbers from the data.

=== SOC ANALYSIS ===

**Risk Assessment:** [1-2 sentences with severity level]

**Key Findings:**
- [Finding referencing specific data]
- [Finding referencing specific data]

**Threat Indicators:**
- [Threat based on actual data patterns]

**Recommended Actions:**
1. [Specific actionable step]
2. [Specific actionable step]
3. [Specific actionable step]

**Priority:** [CRITICAL / HIGH / MEDIUM / LOW] - [justification]"""


def _parse_soc_insight(response: str) -> Dict[str, str]:
    result = {"intro": None, "analysis": None}
    if "=== QUERY INTERPRETATION ===" in response and "=== SOC ANALYSIS ===" in response:
        parts = response.split("=== SOC ANALYSIS ===")
        intro_part = parts[0].split("=== QUERY INTERPRETATION ===")
        if len(intro_part) > 1:
            result["intro"] = intro_part[1].strip()
        if len(parts) > 1:
            result["analysis"] = parts[1].strip()
    elif "=== SOC ANALYSIS ===" in response:
        parts = response.split("=== SOC ANALYSIS ===")
        result["analysis"] = parts[1].strip() if len(parts) > 1 else response.strip()
    else:
        result["analysis"] = response.strip()
    return result


# ════════════════════════════════════════════════════════
# LocalLLMClient — DEPRECATED cloud-forwarding stub
# Local LLM (Ollama qwen2.5:14b) removed per directive 2026-04-15.
# This class is kept to preserve existing imports. All calls now go to cloud.
# WARNING: Client-sensitive SOC data (IPs, CVEs, agent names, narratives)
# now flows to the cloud provider (HuggingFace / DeepSeek V3.1-Terminus).
# ════════════════════════════════════════════════════════

class LocalLLMClient:
    """
    DEPRECATED WRAPPER — kept for backward compatibility only.
    After local LLM removal, every call forwards to the cloud LLMClient.
    The host/model constructor args are accepted but ignored; the cloud target
    is read from LLM_BASE_URL / LLM_MODEL / LLM_API_KEY env vars.
    """

    def __init__(self, host: str = "", model: str = "", cloud_client: Optional["LLMClient"] = None):
        # Constructor args preserved for backward compatibility but ignored.
        self._cloud = cloud_client or LLMClient(
            base_url=os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus"),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        # Exposed attributes so legacy code reading .host/.model keeps working
        self.host = self._cloud.base_url
        self.model = self._cloud.model
        self._initialized = False

    async def initialize(self):
        await self._cloud.initialize()
        self._initialized = True
        logger.info(f"LocalLLMClient (deprecated stub) → forwarding to CLOUD {self._cloud.model}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        return await self._cloud.generate(prompt, max_tokens, temperature, system_prompt, json_mode)

    async def analyze_security_data(self, data_summary: str, analysis_type: str = "general") -> str:
        return await self._cloud.analyze_security_data(data_summary, analysis_type)

    async def generate_recommendations(self, findings: str, priority: str = "high") -> list[str]:
        return await self._cloud.generate_recommendations(findings, priority)

    async def generate_soc_insight(
        self,
        user_query: str,
        data_summary: str,
        analysis_type: str = "general",
        conversation_context: str = "",
    ) -> Dict[str, str]:
        return await self._cloud.generate_soc_insight(
            user_query, data_summary, analysis_type, conversation_context
        )

    async def close(self):
        await self._cloud.close()
        self._initialized = False


# ════════════════════════════════════════════════════════
# CLOUD LLM CLIENT — DeepSeek V3.1-Terminus via HuggingFace
# Handles ALL LLM workload (SOC analysis, narratives, tool selection, decoder gen).
# ════════════════════════════════════════════════════════

class LLMClient:
    """
    OpenAI-compatible cloud LLM client.
    Works with any /v1/chat/completions provider.
    Used for orchestration, tool selection, decoder generation.
    """

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "deepseek-ai/DeepSeek-V3.1-Terminus",
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.hf_org = os.getenv("HF_ORG_NAME", "")
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self._initialized = True
        logger.info(f"LLMClient initialized: {self.base_url} - {self.model} (CLOUD)")

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        await self._ensure_initialized()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["</response>", "\n\n\n"],
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.hf_org:
                headers["X-HF-Bill-To"] = self.hf_org

            # 1 retry on transient cloud LLM failures (timeout / 429 / 5xx).
            # HuggingFace router is flaky enough that a single retry cuts
            # user-visible error rate; the analyst already waited 30-180s on
            # the first attempt, so beyond 1 retry just wastes their time.
            transient_codes = {429, 502, 503, 504}
            retry_delay_sec = 2
            response = None
            last_timeout = None

            for attempt in (1, 2):
                try:
                    response = await self.client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                except httpx.TimeoutException as te:
                    last_timeout = te
                    if attempt == 1:
                        logger.warning("[CLOUD] LLM timeout on attempt %s — retrying once after %ss", attempt, retry_delay_sec)
                        await asyncio.sleep(retry_delay_sec)
                        continue
                    raise

                if response.status_code in transient_codes and attempt == 1:
                    logger.warning(
                        "[CLOUD] LLM transient error %s on attempt %s — retrying once after %ss",
                        response.status_code, attempt, retry_delay_sec,
                    )
                    await asyncio.sleep(retry_delay_sec)
                    continue
                break

            if response is None:
                raise last_timeout or httpx.TimeoutException("LLM request timed out")

            response.raise_for_status()

            data = response.json()
            generated_text = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"[CLOUD] Generated {len(generated_text)} chars")
            return generated_text

        except httpx.TimeoutException:
            logger.error("[CLOUD] LLM request timed out (after retry)")
            raise Exception("Cloud LLM generation timed out")
        except httpx.HTTPStatusError as e:
            # Surface the full response body so we can see WHY the server rejected it
            # (e.g. "model not found", "unsupported parameter", etc.)
            try:
                err_body = e.response.text[:500]
            except Exception:
                err_body = "<unable to read response body>"
            logger.error(
                f"[CLOUD] LLM HTTP {e.response.status_code} for model={self.model} "
                f"at {self.base_url} — response body: {err_body}"
            )
            raise Exception(f"Cloud LLM generation failed ({e.response.status_code}): {err_body}")
        except Exception as e:
            logger.error(f"[CLOUD] LLM generation error: {e}")
            raise

    async def analyze_security_data(self, data_summary: str, analysis_type: str = "general") -> str:
        prompt = _build_analysis_prompt(data_summary, analysis_type)
        return await self.generate(prompt=prompt, system_prompt=SOC_SYSTEM_PROMPT, max_tokens=800, temperature=0.3)

    async def generate_recommendations(self, findings: str, priority: str = "high") -> list[str]:
        system_prompt = "You are a security architect providing actionable remediation guidance.\nGenerate specific, technical recommendations that can be implemented immediately."
        prompt = f"Based on these security findings:\n\n{findings}\n\nGenerate 3-5 specific, actionable recommendations for {priority} priority issues.\n\nFormat as a numbered list:\n1. [Action]\n2. [Action]\n\nRecommendations:"
        response = await self.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=1200, temperature=0.4)
        recommendations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                rec = line.lstrip('0123456789.-) ').strip()
                if rec:
                    recommendations.append(rec)
        return recommendations[:5]

    async def generate_soc_insight(self, user_query: str, data_summary: str, analysis_type: str = "general", conversation_context: str = "") -> Dict[str, str]:
        prompt = _build_soc_insight_prompt(user_query, data_summary, analysis_type, conversation_context)
        try:
            response = await self.generate(prompt=prompt, system_prompt=SOC_INSIGHT_SYSTEM_PROMPT, max_tokens=2000, temperature=0.3)
            return _parse_soc_insight(response)
        except Exception as e:
            logger.warning(f"[CLOUD] SOC insight generation failed: {e}")
            return {"intro": None, "analysis": None}

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self._initialized = False


# ════════════════════════════════════════════════════════
# LLM ROUTER — Cloud-only (local LLM removed 2026-04-15 per directive)
# Sensitivity labels preserved for audit logging, but ALL calls go to cloud.
# ════════════════════════════════════════════════════════

class LLMRouter:
    """
    Cloud-only LLM router. Every call goes to the HuggingFace / DeepSeek cloud client.

    WARNING: Client-sensitive SOC data (IPs, CVE IDs, agent names, forensic narratives)
    is now sent to the cloud LLM provider. The on-premise Ollama path has been removed.

    Sensitivity labels are retained for telemetry/audit logging only — they do NOT
    gate routing anymore.
    """

    HIGH_SENSITIVITY = frozenset({
        "soc_analysis", "soc_insight", "forensic_narrative",
        "primary_analysis", "recommendations",
    })

    LOW_SENSITIVITY = frozenset({
        "tool_selection", "decoder_generation", "decoder_explanation",
        "query_rewriting",
    })

    def __init__(self, local_client, cloud_client: LLMClient):
        # local_client parameter retained for backward compat but ignored.
        self.cloud = cloud_client
        self.local = cloud_client  # alias so legacy .local references still work
        self._initialized = False
        self._cloud_available = False

    async def _check_cloud_health(self) -> bool:
        """Check if cloud LLM API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {}
                if self.cloud.api_key:
                    headers["Authorization"] = f"Bearer {self.cloud.api_key}"
                if self.cloud.hf_org:
                    headers["X-HF-Bill-To"] = self.cloud.hf_org
                resp = await client.get(f"{self.cloud.base_url}/models", headers=headers)
                return resp.status_code == 200
        except Exception:
            return False

    async def initialize(self):
        """Initialize cloud client and verify reachability."""
        await self.cloud.initialize()
        self._cloud_available = await self._check_cloud_health()
        self._initialized = True

        if self._cloud_available:
            logger.info(f"LLMRouter initialized: CLOUD-ONLY mode (model={self.cloud.model})")
        else:
            logger.error("LLMRouter initialized: CLOUD ✗ — LLM features UNAVAILABLE (no local fallback)")

    async def _refresh_availability(self):
        """Re-check cloud availability (called on errors)."""
        self._cloud_available = await self._check_cloud_health()

    def get_client(self, call_type: str = "general"):
        """
        Return the cloud client for any call. The sensitivity label is logged
        for audit/compliance visibility but no longer affects routing.
        """
        if call_type in self.HIGH_SENSITIVITY:
            logger.debug(f"[ROUTER] {call_type} → CLOUD (HIGH-SENSITIVITY data sent to cloud provider)")
        else:
            logger.debug(f"[ROUTER] {call_type} → CLOUD")
        return self.cloud

    # ─── Convenience methods (all cloud-only now) ───

    async def analyze_security_data(self, data_summary: str, analysis_type: str = "general") -> str:
        try:
            return await self.cloud.analyze_security_data(data_summary, analysis_type)
        except Exception as e:
            logger.error(f"[ROUTER] Cloud analyze failed: {e}")
            await self._refresh_availability()
            return ""

    async def generate_soc_insight(
        self,
        user_query: str,
        data_summary: str,
        analysis_type: str = "general",
        conversation_context: str = "",
    ) -> Dict[str, str]:
        try:
            return await self.cloud.generate_soc_insight(
                user_query, data_summary, analysis_type, conversation_context
            )
        except Exception as e:
            logger.error(f"[ROUTER] Cloud SOC insight failed: {e}")
            await self._refresh_availability()
            return {"intro": None, "analysis": None}

    async def generate_recommendations(self, findings: str, priority: str = "high") -> list[str]:
        try:
            return await self.cloud.generate_recommendations(findings, priority)
        except Exception as e:
            logger.error(f"[ROUTER] Cloud recommendations failed: {e}")
            return []

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        call_type: str = "general",
        json_mode: bool = False,
    ) -> str:
        """Cloud-only generate(). No local fallback."""
        # Log sensitivity label for audit visibility
        if call_type in self.HIGH_SENSITIVITY:
            logger.debug(f"[ROUTER] generate {call_type} → CLOUD (HIGH-SENSITIVITY)")

        try:
            return await self.cloud.generate(prompt, max_tokens, temperature, system_prompt, json_mode)
        except Exception as e:
            logger.warning(f"[ROUTER] Cloud generate failed for {call_type}: {e}")
            await self._refresh_availability()
            return ""

    async def close(self):
        await self.cloud.close()


# Backward compatibility aliases
OllamaClient = LLMRouter  # Code importing OllamaClient gets the router
