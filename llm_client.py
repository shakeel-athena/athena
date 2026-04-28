"""
LLM Client - Cloud LLM integration for analysis and synthesis
Supports any OpenAI-compatible API (OpenRouter, Together AI, NVIDIA NIM, etc.)
Used ONLY for generating insights, NOT for tool selection
"""

import os
import httpx
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI-compatible LLM client.
    Works with any provider that supports /v1/chat/completions (OpenRouter, Together AI, etc.)
    Used exclusively for analysis generation, not tool orchestration.
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
        self.hf_org = os.getenv("HF_ORG_NAME", "")  # HuggingFace org for billing
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=180.0)
        self._initialized = True
        logger.info(f"LLMClient initialized: {self.base_url} - {self.model}")

    async def _ensure_initialized(self):
        """Ensure client is initialized"""
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
        """
        Generate text from prompt using OpenAI-compatible chat completions.

        Args:
            prompt: User prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            system_prompt: Optional system prompt
            json_mode: If True, enable JSON response format

        Returns:
            Generated text
        """
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

            # 1 retry on transient failures (timeout / 429 / 502 / 503 / 504).
            # Cloud LLM endpoints are flaky enough that a single retry
            # materially cuts user-visible error rate. Anything beyond 1
            # retry just wastes the analyst's time — the analyst already
            # waited 30-180s for the first attempt.
            transient_codes = {429, 502, 503, 504}
            retry_delay_sec = 2
            response = None
            last_timeout_exc = None

            for attempt in (1, 2):
                try:
                    response = await self.client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                except httpx.TimeoutException as te:
                    last_timeout_exc = te
                    if attempt == 1:
                        logger.warning("LLM timeout on attempt %s — retrying once after %ss", attempt, retry_delay_sec)
                        await asyncio.sleep(retry_delay_sec)
                        continue
                    raise

                if response.status_code in transient_codes and attempt == 1:
                    logger.warning(
                        "LLM transient error %s on attempt %s — retrying once after %ss",
                        response.status_code, attempt, retry_delay_sec,
                    )
                    await asyncio.sleep(retry_delay_sec)
                    continue
                break

            if response is None:
                # Both attempts timed out
                raise last_timeout_exc or httpx.TimeoutException("LLM request timed out")

            response.raise_for_status()

            data = response.json()
            generated_text = data["choices"][0]["message"]["content"].strip()

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except httpx.TimeoutException:
            logger.error("LLM request timed out (after retry)")
            raise Exception("LLM generation timed out")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Pull the upstream error body for diagnostics — providers like
            # HuggingFace return useful messages such as "User Access Token expired".
            try:
                err_body = e.response.json()
                err_detail = (
                    err_body.get("error")
                    or err_body.get("message")
                    or err_body.get("detail")
                    or str(err_body)
                )
            except Exception:
                err_detail = (e.response.text or "")[:300]

            if status in (401, 403):
                logger.error(
                    "LLM authentication failed (HTTP %s) — token is invalid, "
                    "expired, or lacks access to model '%s'. Upstream said: %s",
                    status,
                    getattr(self, "model", "?"),
                    err_detail,
                )
                raise Exception(
                    f"LLM auth failed ({status}): token invalid or expired. "
                    f"Upstream: {err_detail}"
                )

            logger.error(
                "LLM HTTP error %s. Upstream said: %s", status, err_detail
            )
            raise Exception(f"LLM generation failed: {status} — {err_detail}")
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    async def analyze_security_data(
        self,
        data_summary: str,
        analysis_type: str = "general"
    ) -> str:
        """
        Generate security analysis from data summary.

        Args:
            data_summary: Structured summary of security data
            analysis_type: Type of analysis (vulnerability, alert, agent, etc.)

        Returns:
            Analysis text
        """

        system_prompt = """You are a senior SOC analyst with 15+ years of experience.
Your role is to analyze security data and provide actionable insights.

CRITICAL RULES:
1. ONLY reference data explicitly provided in the prompt — quote exact values from the data
2. NEVER invent or fabricate CVE IDs, IP addresses, agent names, or any identifiers not in the data
3. Be concise - 2-3 sentences maximum
4. Focus on business impact and urgency
5. Provide specific, actionable recommendations
6. If the data shows zero results or is empty, state that clearly"""

        if analysis_type == "vulnerability":
            prompt = f"""Analyze this vulnerability data and provide a concise summary:

{data_summary}

Provide:
1. Overall risk assessment (1-2 sentences)
2. Most critical concern
3. Immediate action required

Keep response under 150 words."""

        elif analysis_type == "alert":
            prompt = f"""Analyze these security alerts:

{data_summary}

Provide:
1. Pattern assessment (1-2 sentences)
2. Potential attack indicators
3. Recommended investigation steps

Keep response under 150 words."""

        elif analysis_type == "agent":
            prompt = f"""Analyze this agent status data:

{data_summary}

Provide:
1. Infrastructure health assessment
2. Critical issues requiring attention
3. Recommended actions

Keep response under 150 words."""

        elif analysis_type == "port":
            prompt = f"""Analyze these open ports for security risks:

{data_summary}

Provide:
1. Exposure assessment — which ports pose the highest risk
2. Suspicious ports — any known malicious or unnecessary services
3. Firewall recommendations — what should be restricted

Keep response under 150 words."""

        elif analysis_type == "process":
            prompt = f"""Analyze these running processes for security concerns:

{data_summary}

Provide:
1. Suspicious process assessment — any known malware patterns or anomalies
2. Resource abuse indicators — crypto mining, excessive CPU/memory
3. Recommended actions — processes to investigate or terminate

Keep response under 150 words."""

        elif analysis_type == "statistics":
            prompt = f"""Analyze these system statistics for operational health:

{data_summary}

Provide:
1. Queue and throughput health — any bottlenecks or message loss
2. Capacity concerns — is the system under strain
3. Recommended actions — scaling or configuration changes needed

Keep response under 150 words."""

        else:
            prompt = f"""Analyze this security data:

{data_summary}

Provide a concise summary highlighting key findings and recommended actions.
Keep response under 150 words."""

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=800,
            temperature=0.3
        )

    async def generate_recommendations(
        self,
        findings: str,
        priority: str = "high"
    ) -> list[str]:
        """
        Generate actionable recommendations.

        Args:
            findings: Security findings summary
            priority: Priority level (critical, high, medium)

        Returns:
            List of recommendations
        """

        system_prompt = """You are a security architect providing actionable remediation guidance.
Generate specific, technical recommendations that can be implemented immediately."""

        prompt = f"""Based on these security findings:

{findings}

Generate 3-5 specific, actionable recommendations for {priority} priority issues.

Format as a numbered list:
1. [Action]
2. [Action]
...

Each recommendation should be:
- Specific and technical
- Immediately actionable
- Include relevant commands or tools where applicable

Recommendations:"""

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1200,
            temperature=0.4
        )

        # Parse numbered list
        recommendations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove number/bullet
                rec = line.lstrip('0123456789.-) ').strip()
                if rec:
                    recommendations.append(rec)

        return recommendations[:5]  # Max 5

    async def generate_soc_insight(
        self,
        user_query: str,
        data_summary: str,
        analysis_type: str = "general",
        conversation_context: str = ""
    ) -> Dict[str, str]:
        """
        Generate query interpretation + SOC analysis in one LLM call.

        Args:
            user_query: The SOC operator's query
            data_summary: Formatted data summary
            analysis_type: Type of analysis (vulnerability, alert, agent, etc.)
            conversation_context: Previous conversation summary for multi-turn context

        Returns:
            {"intro": "query interpretation text", "analysis": "SOC analysis text"}
        """

        system_prompt = """You are a senior SOC analyst providing executive-level security briefings.

CRITICAL RULES:
1. ONLY reference data explicitly provided below — NEVER invent metrics, IPs, CVE IDs, or agent names
2. Be precise and actionable — quote exact numbers and identifiers from the data
3. If the data is empty or shows zero results, say so clearly — do NOT make up findings
4. Do NOT hallucinate or fabricate any information not present in the data
5. When referencing specific CVEs, agent IDs, or IPs, they MUST appear verbatim in the data above"""

        # Analysis-type-specific focus guidance
        focus_map = {
            "vulnerability": "Focus on CVE severity distribution, affected agent count, patch urgency, exploit availability.",
            "alert": "Focus on alert volume trends, rule concentration (is one rule dominating?), attack chain indicators.",
            "agent": "Focus on active/disconnected ratio, coverage gaps, version drift, infrastructure health.",
            "port": "Focus on exposed services, known-bad ports, lateral movement risk.",
            "process": "Focus on suspicious binaries, resource abuse, persistence mechanisms.",
            "statistics": "Focus on throughput health, queue saturation, capacity trends.",
            # Correlation strategy-specific focus (matched by "correlation_{strategy_name}")
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
            # Generic fallback for unknown correlation types
            "correlation": "Focus on cross-data relationships: which agents appear in multiple risk categories, combined risk assessment, prioritization.",
        }

        focus = focus_map.get(analysis_type, "Focus on the most relevant security insights from this data.")

        # Build conversation context block if available
        context_block = ""
        if conversation_context:
            context_block = f"""CONVERSATION CONTEXT (previous queries in this session):
{conversation_context}

"""

        prompt = f"""The SOC operator asked: "{user_query}"

{context_block}Retrieved data summary:
{data_summary}

{focus}

Provide exactly TWO sections with these markers:

=== QUERY INTERPRETATION ===
1-2 sentences: What the operator asked for, and a brief summary of key findings.
Be conversational but professional. Reference actual numbers from the data.
Example: "You requested all active agents. Out of 48 total agents, only 4 are currently active - a significant coverage gap that needs investigation."

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

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.3
            )

            # Parse the two sections
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
                # No markers found — use entire response as analysis (backward compatible)
                result["analysis"] = response.strip()

            return result

        except Exception as e:
            logger.warning(f"SOC insight generation failed: {e}")
            return {"intro": None, "analysis": None}

    async def close(self):
        """Close HTTP client"""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self._initialized = False


# Backward compatibility alias — existing code that imports OllamaClient will still work
OllamaClient = LLMClient

