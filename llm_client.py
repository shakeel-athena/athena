"""
LLM Client — Hybrid Local + Cloud Architecture
- LocalLLMClient: Ollama (qwen2.5:14b) for sensitive data (stays on-premise)
- LLMClient: Cloud (Kimi-K2.5 via HuggingFace) for orchestration + code gen
- LLMRouter: Routes requests based on data sensitivity
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
# LOCAL LLM CLIENT — Ollama (qwen2.5:14b)
# For HIGH sensitivity data (IPs, CVEs, agent names)
# Data NEVER leaves the tenant network
# ════════════════════════════════════════════════════════

class LocalLLMClient:
    """
    Ollama-based local LLM client for sensitive data operations.
    Uses /api/generate endpoint (Ollama native format).
    Data never leaves the tenant network.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:14b-instruct"):
        self.host = host.rstrip("/")
        self.model = model
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self._initialized = True
        logger.info(f"LocalLLMClient initialized: {self.host} - {self.model} (ON-PREMISE)")

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        await self._ensure_initialized()
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["</response>", "\n\n\n"],
                },
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = await self.client.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()
            generated_text = data.get("response", "").strip()
            logger.debug(f"[LOCAL] Generated {len(generated_text)} chars")
            return generated_text

        except httpx.TimeoutException:
            logger.error("[LOCAL] Ollama request timed out")
            raise Exception("Local LLM generation timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"[LOCAL] Ollama HTTP error: {e.response.status_code}")
            raise Exception(f"Local LLM generation failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[LOCAL] Ollama generation error: {e}")
            raise

    async def analyze_security_data(self, data_summary: str, analysis_type: str = "general") -> str:
        prompt = _build_analysis_prompt(data_summary, analysis_type)
        return await self.generate(prompt=prompt, system_prompt=SOC_SYSTEM_PROMPT, max_tokens=200, temperature=0.3)

    async def generate_recommendations(self, findings: str, priority: str = "high") -> list[str]:
        system_prompt = "You are a security architect providing actionable remediation guidance.\nGenerate specific, technical recommendations that can be implemented immediately."
        prompt = f"Based on these security findings:\n\n{findings}\n\nGenerate 3-5 specific, actionable recommendations for {priority} priority issues.\n\nFormat as a numbered list:\n1. [Action]\n2. [Action]\n\nRecommendations:"
        response = await self.generate(prompt=prompt, system_prompt=system_prompt, max_tokens=300, temperature=0.4)
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
            response = await self.generate(prompt=prompt, system_prompt=SOC_INSIGHT_SYSTEM_PROMPT, max_tokens=500, temperature=0.3)
            return _parse_soc_insight(response)
        except Exception as e:
            logger.warning(f"[LOCAL] SOC insight generation failed: {e}")
            return {"intro": None, "analysis": None}

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self._initialized = False


# ════════════════════════════════════════════════════════
# CLOUD LLM CLIENT — Kimi-K2.5 via HuggingFace
# For LOW sensitivity data (query text, decoder code, tool names)
# Better reasoning, code gen, and structured output
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
        model: str = "moonshotai/kimi-k2.5",
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
            # user-visible error rate; analyst already waited 30-180s for
            # the first attempt, beyond 1 retry just wastes their time.
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
            logger.error(f"[CLOUD] LLM HTTP error: {e.response.status_code}")
            raise Exception(f"Cloud LLM generation failed: {e.response.status_code}")
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
# LLM ROUTER — Sensitivity-based routing
# ════════════════════════════════════════════════════════

class LLMRouter:
    """
    Routes LLM requests to local or cloud model based on data sensitivity.
    Automatically detects which models are available and falls back accordingly.

    Routing rules (when both available):
    - HIGH sensitivity (client data: IPs, CVEs, agent names) → Local (qwen2.5:14b via Ollama)
    - LOW sensitivity (query text, tool names, code gen) → Cloud (Kimi-K2.5 via HuggingFace)

    Fallback rules:
    - If LOCAL is down → LLM features disabled (sensitive data NEVER goes to cloud)
    - If CLOUD is down → Local handles everything (cloud tasks fall back to local)
    - If BOTH are down → LLM features disabled
    """

    HIGH_SENSITIVITY = frozenset({
        "soc_analysis", "soc_insight", "forensic_narrative",
        "primary_analysis", "recommendations",
    })

    LOW_SENSITIVITY = frozenset({
        "tool_selection", "decoder_generation", "decoder_explanation",
        "query_rewriting",
    })

    def __init__(self, local_client: LocalLLMClient, cloud_client: LLMClient):
        self.local = local_client
        self.cloud = cloud_client
        self._initialized = False
        self._local_available = False
        self._cloud_available = False

    async def _check_local_health(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.local.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

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
        """Initialize both clients and check availability.

        Both `self.local` and `self.cloud` are optional — production deployments
        without on-prem Ollama pass `local=None`, and the chat path can run
        cloud-only. Skip initialization on the missing side rather than
        crashing on `NoneType.initialize()`. Health checks below also already
        tolerate a missing client.
        """
        if self.local is not None:
            await self.local.initialize()
        if self.cloud is not None:
            await self.cloud.initialize()

        self._local_available = await self._check_local_health() if self.local is not None else False
        self._cloud_available = await self._check_cloud_health() if self.cloud is not None else False

        self._initialized = True

        if self._local_available and self._cloud_available:
            logger.info("LLMRouter initialized: LOCAL ✓ + CLOUD ✓ (hybrid routing active)")
        elif self._local_available and not self._cloud_available:
            logger.warning("LLMRouter initialized: LOCAL ✓ + CLOUD ✗ (all requests → LOCAL)")
        elif not self._local_available and self._cloud_available:
            logger.error("LLMRouter initialized: LOCAL ✗ + CLOUD ✓ (LLM DISABLED — local model required, cloud will NOT be used for sensitive data)")
        else:
            logger.error("LLMRouter initialized: LOCAL ✗ + CLOUD ✗ (NO LLM AVAILABLE)")

    async def _refresh_availability(self):
        """Periodically refresh model availability (called on errors)."""
        self._local_available = await self._check_local_health()
        self._cloud_available = await self._check_cloud_health()

    def get_client(self, call_type: str = "general"):
        """
        Get appropriate client based on data sensitivity + availability.

        Rules:
        1. Both available → route by sensitivity (local for sensitive, cloud for orchestration)
        2. Only local available → local for everything (cloud tasks fall back to local)
        3. Local down → NO LLM (sensitive data NEVER goes to cloud)
        """
        if self._local_available and self._cloud_available:
            # Normal hybrid routing
            if call_type in self.LOW_SENSITIVITY:
                logger.debug(f"[ROUTER] {call_type} → CLOUD (low sensitivity)")
                return self.cloud
            logger.debug(f"[ROUTER] {call_type} → LOCAL (high sensitivity)")
            return self.local

        elif self._local_available and not self._cloud_available:
            # Cloud is down — local handles everything
            logger.debug(f"[ROUTER] {call_type} → LOCAL (cloud unavailable, local handles all)")
            return self.local

        else:
            # Local is down — REFUSE to use cloud for sensitive data
            # Return None to signal LLM is unavailable
            logger.error(f"[ROUTER] {call_type} → BLOCKED (local model not running, refusing cloud for data privacy)")
            return None

    # ─── Convenience methods ───

    async def analyze_security_data(self, data_summary: str, analysis_type: str = "general") -> str:
        """HIGH sensitivity — local only. If local down, returns empty (never cloud)."""
        if not self._local_available:
            await self._refresh_availability()
        if not self._local_available:
            logger.error("[ROUTER] analyze_security_data BLOCKED — local model not running")
            return ""
        try:
            return await self.local.analyze_security_data(data_summary, analysis_type)
        except Exception as e:
            logger.error(f"[ROUTER] Local analyze failed: {e}")
            await self._refresh_availability()
            return ""

    async def generate_soc_insight(self, user_query: str, data_summary: str, analysis_type: str = "general", conversation_context: str = "") -> Dict[str, str]:
        """HIGH sensitivity — local only. If local down, returns empty (never cloud)."""
        if not self._local_available:
            await self._refresh_availability()
        if not self._local_available:
            logger.error("[ROUTER] generate_soc_insight BLOCKED — local model not running")
            return {"intro": None, "analysis": None}
        try:
            return await self.local.generate_soc_insight(user_query, data_summary, analysis_type, conversation_context)
        except Exception as e:
            logger.error(f"[ROUTER] Local SOC insight failed: {e}")
            await self._refresh_availability()
            return {"intro": None, "analysis": None}

    async def generate_recommendations(self, findings: str, priority: str = "high") -> list[str]:
        """MEDIUM sensitivity — local only. If local down, returns empty."""
        if not self._local_available:
            await self._refresh_availability()
        if not self._local_available:
            return []
        try:
            return await self.local.generate_recommendations(findings, priority)
        except Exception as e:
            logger.error(f"[ROUTER] Local recommendations failed: {e}")
            return []

    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3, system_prompt: Optional[str] = None, call_type: str = "general", json_mode: bool = False) -> str:
        """
        Route generate() based on call_type.
        - LOW sensitivity → cloud, falls back to local if cloud fails
        - HIGH sensitivity → local only, NEVER falls back to cloud
        """
        client = self.get_client(call_type)

        if client is None:
            # Local is down and this is sensitive — refuse
            logger.error(f"[ROUTER] generate BLOCKED for {call_type} — local model required but not running")
            return ""

        try:
            return await client.generate(prompt, max_tokens, temperature, system_prompt, json_mode)
        except Exception as e:
            logger.warning(f"[ROUTER] Primary generate failed for {call_type}: {e}")
            await self._refresh_availability()

            # Only fall back to local if cloud failed (NEVER fall back to cloud)
            if call_type in self.LOW_SENSITIVITY and self._local_available:
                logger.info(f"[ROUTER] Cloud failed for {call_type}, falling back to LOCAL")
                return await self.local.generate(prompt, max_tokens, temperature, system_prompt, json_mode)

            return ""

    async def close(self):
        await self.local.close()
        await self.cloud.close()


# Backward compatibility aliases
OllamaClient = LLMRouter  # Code importing OllamaClient gets the router
