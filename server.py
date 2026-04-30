#!/usr/bin/env python3
"""
Wazuh MCP Server - Complete MCP-Compliant Remote Server
Full compliance with Model Context Protocol 2025-06-18 specification
Production-ready with Streamable HTTP, SSE transport, authentication, and hybrid intelligence
"""

# MCP Protocol Version Support
MCP_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

import os
import json
import asyncio
import secrets
import logging
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
import uuid

# ============================================================
# SIMPLE RESPONSE FORMATTERS
# ============================================================

def format_agent_response(data):
    """Format agent response as JSON"""
    import json
    return json.dumps(data, indent=2, default=str)

def format_alert_response(data):
    """Format alert response as JSON"""
    import json
    return json.dumps(data, indent=2, default=str)

def format_vulnerability_response(data):
    """Format vulnerability response as JSON"""
    import json
    return json.dumps(data, indent=2, default=str)

def format_statistics_response(data):
    """Format statistics response as Markdown table of top rules by frequency."""
    import json
    try:
        items = data.get("data", {}).get("affected_items", [])
        if not items:
            return "## Wazuh Statistics\n\nNo statistics available."
        # Aggregate rule counts across all hours
        rule_totals = {}
        for hour_data in items:
            for alert in hour_data.get("alerts", []):
                sid = alert.get("sigid", "?")
                lvl = alert.get("level", 0)
                cnt = alert.get("times", 0)
                if sid not in rule_totals:
                    rule_totals[sid] = {"level": lvl, "count": 0}
                rule_totals[sid]["count"] += cnt
        total_events = sum(v["count"] for v in rule_totals.values())
        top_rules = sorted(rule_totals.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        output = "## Wazuh Statistics\n\n"
        output += f"**Total Events Recorded:** {total_events:,}\n"
        output += f"**Unique Rules Triggered:** {len(rule_totals)}\n"
        output += f"**Hours of Data:** {len(items)}\n\n"
        output += "### Top 10 Rules by Frequency\n\n"
        output += "| Rule ID | Level | Event Count | % of Total |\n"
        output += "|---------|-------|-------------|------------|\n"
        for sid, info in top_rules:
            pct = (info["count"] / total_events * 100) if total_events else 0
            output += f"| {sid} | {info['level']} | {info['count']:,} | {pct:.1f}% |\n"
        return output
    except Exception:
        return json.dumps(data, indent=2, default=str)

def format_analysis_response(data):
    """Format analysis response as JSON"""
    import json
    return json.dumps(data, indent=2, default=str)



from fastapi import FastAPI, Request, Response, HTTPException, Header, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
import httpx

from wazuh_mcp_server.config import get_config, WazuhConfig
from wazuh_mcp_server.api.wazuh_client import WazuhClient
from wazuh_mcp_server.api.suricata_client import SuricataClient
from wazuh_mcp_server.api.wazuh_indexer import IndexerNotConfiguredError
from wazuh_mcp_server.llm_client import LLMClient, LocalLLMClient, LLMRouter, OllamaClient
from wazuh_mcp_server.auth import create_access_token, verify_token
from wazuh_mcp_server.security import RateLimiter, validate_input
from wazuh_mcp_server.monitoring import REQUEST_COUNT, REQUEST_DURATION, ACTIVE_CONNECTIONS
from wazuh_mcp_server.resilience import GracefulShutdown
from wazuh_mcp_server.session_store import create_session_store, SessionStore

# Pallas 4.3 - SOC Workflow Automation imports
from wazuh_mcp_server.incident_store import IncidentStore
from wazuh_mcp_server.jira_client import JiraClient

# ============================================================================
# IMPORT HYBRID QUERY ORCHESTRATOR (REPLACES CHAT HANDLER)
# ============================================================================
from wazuh_mcp_server.query_orchestrator import create_orchestrator, QueryOrchestrator, QueryContext
from wazuh_mcp_server.conversation_memory import ConversationMemory, ConversationTurn, ReferenceResolver

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# ============================================================================
_query_orchestrator: Optional[QueryOrchestrator] = None

# ============================================================================
# CONVERSATION MEMORY — Multi-turn SOC Chat context
# ============================================================================
_conversation_memory = ConversationMemory(max_turns=10, ttl_minutes=30)

# OAuth manager (initialized on startup if needed)
_oauth_manager = None


async def verify_authentication(authorization: Optional[str], config) -> bool:
    """
    Verify authentication based on configured auth mode.

    Returns True if authenticated, raises HTTPException if not.
    Supports: authless (none), bearer token, and OAuth modes.
    """
    # Authless mode - no authentication required
    if config.is_authless:
        return True

    # Authentication required
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # OAuth mode
    if config.is_oauth:
        global _oauth_manager
        if _oauth_manager:
            token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
            token_obj = _oauth_manager.validate_access_token(token)
            if token_obj:
                return True
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired OAuth token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Bearer token mode (default)
    try:
        from wazuh_mcp_server.auth import verify_bearer_token
        await verify_bearer_token(authorization)
        return True
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


# MCP Protocol Models
class MCPRequest(BaseModel):
    """MCP JSON-RPC 2.0 Request."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")
    method: str = Field(description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")


class MCPResponse(BaseModel):
    """MCP JSON-RPC 2.0 Response."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")
    result: Optional[Any] = Field(default=None, description="Result data")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error object")


class MCPError(BaseModel):
    """MCP JSON-RPC 2.0 Error object."""
    code: int = Field(description="Error code")
    message: str = Field(description="Error message")
    data: Optional[Any] = Field(default=None, description="Additional error data")


class MCPSession:
    """MCP Session Management for Remote MCP Server."""

    def __init__(self, session_id: str, origin: Optional[str] = None):
        self.session_id = session_id
        self.origin = origin
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        self.capabilities = {}
        self.client_info = {}
        self.authenticated = False

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session is expired."""
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "origin": self.origin,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "capabilities": self.capabilities,
            "client_info": self.client_info,
            "authenticated": self.authenticated
        }


# Session management with pluggable backend (serverless-ready)
class SessionManager:
    """
    Session manager with pluggable storage backend.
    Supports both in-memory (default) and Redis (serverless-ready) backends.
    """

    def __init__(self, store: SessionStore):
        self._store = store
        self._lock = threading.RLock()  # For synchronous operations
        logger.info(f"SessionManager initialized with {type(store).__name__}")

    def _session_from_dict(self, data: Dict[str, Any]) -> MCPSession:
        """Reconstruct MCPSession from dictionary."""
        session = MCPSession(data['session_id'], data.get('origin'))
        session.created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        session.last_activity = datetime.fromisoformat(data['last_activity'].replace('Z', '+00:00'))
        session.capabilities = data.get('capabilities', {})
        session.client_info = data.get('client_info', {})
        session.authenticated = data.get('authenticated', False)
        return session

    async def get(self, session_id: str) -> Optional[MCPSession]:
        """Get session by ID."""
        data = await self._store.get(session_id)
        if data:
            return self._session_from_dict(data)
        return None

    async def set(self, session_id: str, session: MCPSession) -> bool:
        """Store session."""
        return await self._store.set(session_id, session.to_dict())

    def __getitem__(self, session_id: str) -> MCPSession:
        """Synchronous dict-like access (blocks)."""
        loop = asyncio.get_event_loop()
        session = loop.run_until_complete(self.get(session_id))
        if session is None:
            raise KeyError(f"Session {session_id} not found")
        return session

    def __setitem__(self, session_id: str, session: MCPSession) -> None:
        """Synchronous dict-like access (blocks)."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.set(session_id, session))

    def __delitem__(self, session_id: str) -> None:
        """Synchronous delete (blocks)."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.remove(session_id))

    async def __contains__(self, session_id: str) -> bool:
        """Check if session exists."""
        return await self._store.exists(session_id)

    async def remove(self, session_id: str) -> bool:
        """Remove session by ID."""
        return await self._store.delete(session_id)

    def pop(self, session_id: str, default=None) -> Optional[MCPSession]:
        """Remove and return session (synchronous, blocks)."""
        loop = asyncio.get_event_loop()
        session = loop.run_until_complete(self.get(session_id))
        if session:
            loop.run_until_complete(self.remove(session_id))
            return session
        return default

    async def clear(self) -> bool:
        """Clear all sessions."""
        return await self._store.clear()

    def values(self) -> List[MCPSession]:
        """Get all session values (synchronous, blocks)."""
        loop = asyncio.get_event_loop()
        sessions_dict = loop.run_until_complete(self.get_all())
        return list(sessions_dict.values())

    def keys(self) -> List[str]:
        """Get all session keys (synchronous, blocks)."""
        loop = asyncio.get_event_loop()
        sessions_dict = loop.run_until_complete(self.get_all())
        return list(sessions_dict.keys())

    async def get_all(self) -> Dict[str, MCPSession]:
        """Get all sessions as dictionary."""
        data_dict = await self._store.get_all()
        return {sid: self._session_from_dict(data) for sid, data in data_dict.items()}

    async def cleanup_expired(self) -> int:
        """Remove expired sessions and return count."""
        return await self._store.cleanup_expired()


# Initialize session manager with pluggable backend
# Will use Redis if REDIS_URL is set, otherwise in-memory
_session_store = create_session_store()
sessions = SessionManager(_session_store)


async def get_or_create_session(session_id: Optional[str], origin: Optional[str]) -> MCPSession:
    """Get existing session or create new one."""
    if session_id:
        existing_session = await sessions.get(session_id)
        if existing_session:
            existing_session.update_activity()
            await sessions.set(session_id, existing_session)
            return existing_session

    # Create new session
    new_session_id = session_id or str(uuid.uuid4())
    session = MCPSession(new_session_id, origin)
    await sessions.set(new_session_id, session)

    # Cleanup expired sessions periodically
    try:
        expired_count = await sessions.cleanup_expired()
        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired sessions")
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")

    return session


# Initialize FastAPI app for MCP compliance
# All routes are prefixed with /ai-analyst for reverse proxy compatibility
API_PREFIX = "/ai-analyst"

app = FastAPI(
    title="Wazuh MCP Server",
    description="MCP-compliant remote server for Wazuh SIEM integration. Supports Streamable HTTP, SSE, OAuth, and authless modes.",
    version="4.0.3",
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# Create router with /ai-analyst prefix for all routes
router = APIRouter(prefix=API_PREFIX)

# Get configuration
config = get_config()

# Create Wazuh configuration from server config
wazuh_config = WazuhConfig(
    wazuh_host=config.WAZUH_HOST,
    wazuh_user=config.WAZUH_USER,
    wazuh_pass=config.WAZUH_PASS,
    wazuh_port=config.WAZUH_PORT,
    verify_ssl=config.WAZUH_VERIFY_SSL,
    wazuh_indexer_host=os.getenv("WAZUH_INDEXER_HOST"),
    wazuh_indexer_port=int(os.getenv("WAZUH_INDEXER_PORT", "9200")),
    wazuh_indexer_user=os.getenv("WAZUH_INDEXER_USER"),
    wazuh_indexer_pass=os.getenv("WAZUH_INDEXER_PASS"),
    abuseipdb_api_key=os.getenv("ABUSEIPDB_API_KEY"),
    suricata_enabled=os.getenv("SURICATA_ENABLED", "true").lower() == "true",
    suricata_index_pattern=os.getenv("SURICATA_INDEX_PATTERN", "suricata-*"),
)

# Initialize Wazuh client
wazuh_client = WazuhClient(wazuh_config)

# Initialize Suricata client (optional)
suricata_client = None
if getattr(wazuh_config, "suricata_enabled", False) and getattr(wazuh_config, "wazuh_indexer_host", None):
    try:
        suricata_client = SuricataClient(wazuh_config)
        logger.info(" Suricata client initialized")
    except Exception as e:
        logger.warning(f" Suricata client initialization failed: {e}")
else:
    logger.info(" Suricata integration disabled")

# Pallas 4.3 - Initialize SOC workflow clients
incident_store = IncidentStore()
jira_client_instance = JiraClient()
# Blocking is performed by Agelia (single control plane). Pallas no longer
# instantiates AWS clients — readiness is determined by AGELIA_BASE_URL below.
agelia_blocking_configured: bool = bool(os.getenv("AGELIA_BASE_URL", "").rstrip("/"))
logger.info(
    f"  SOC Workflow: Jira={'ON' if jira_client_instance.is_configured else 'OFF'}, "
    f"AgeliaBlocking={'ON' if agelia_blocking_configured else 'OFF'}"
)

# ─── Agelia Integration ───────────────────────────────────────────────────────
# AGELIA_BASE_URL is the primary integration point for JWT-forwarding proxy calls
# (analyst response actions: block, mute, enrich).
# AGELIA_API_URL + INTEGRATION_API_KEY are used only for the read-only
# machine-to-machine mute summary endpoint consumed by alert display filtering.
AGELIA_BASE_URL       = os.getenv('AGELIA_BASE_URL', '').rstrip('/')
AGELIA_API_URL        = os.getenv('AGELIA_API_URL', '').rstrip('/')
AGELIA_API_KEY        = os.getenv('INTEGRATION_API_KEY', '')
AGELIA_NOTIFY_ENABLED = bool(AGELIA_API_URL and AGELIA_API_KEY)
if AGELIA_BASE_URL:
    logger.info(f"  Agelia JWT proxy: ON ({AGELIA_BASE_URL})")
else:
    logger.info("  Agelia JWT proxy: OFF (set AGELIA_BASE_URL to enable)")
if AGELIA_NOTIFY_ENABLED:
    logger.info(f"  Agelia mute summary (M2M read-only): ON ({AGELIA_API_URL})")


# Shared async HTTP client for Agelia proxy calls (reused across requests)
_agelia_http_client: "httpx.AsyncClient | None" = None

def _get_agelia_client() -> "httpx.AsyncClient":
    global _agelia_http_client
    if _agelia_http_client is None or _agelia_http_client.is_closed:
        _agelia_http_client = httpx.AsyncClient(timeout=30.0)
    return _agelia_http_client


async def _call_agelia(method: str, path: str, request: Request,
                       body: dict = None) -> tuple[dict, int]:
    """Forward an analyst action to Agelia, propagating the Keycloak JWT.

    Returns (response_dict, status_code). Never raises — caller decides how to
    surface errors.
    """
    if not AGELIA_BASE_URL:
        return {"error": "Agelia integration not configured (AGELIA_BASE_URL missing)"}, 503
    analyst_token = request.headers.get("Authorization", "")
    headers = {
        "Content-Type": "application/json",
        "X-Initiated-From": "pallas",
    }
    if analyst_token:
        headers["Authorization"] = analyst_token
    try:
        client = _get_agelia_client()
        resp = await client.request(
            method,
            f"{AGELIA_BASE_URL}{path}",
            json=body,
            headers=headers,
        )
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        return data, resp.status_code
    except Exception as exc:
        logger.error("Agelia proxy call %s %s failed: %s", method, path, exc)
        return {"error": str(exc)}, 502


# Cached active mute rules summary (machine-to-machine read).
# Refreshes from Agelia at most once per _MUTE_SUMMARY_TTL window — alert
# display fetches no longer hit Agelia DB on every request. Pallas-initiated
# actions invalidate the cache immediately so they show up on the next fetch.
_mute_summary_cache: list = []
_mute_summary_fetched_at: float = 0.0
_MUTE_SUMMARY_TTL: float = 60.0  # 1 minute — short window so Agelia changes propagate quickly


async def _get_mute_summary() -> list:
    """Return cached active mute rules from Agelia, refreshing on TTL expiry.

    Uses the shared agelia client with X-Integration-Key (machine-to-machine,
    read-only). Returns stale cache on transient Agelia failure rather than
    failing the alert fetch.
    """
    import time  # local import — `time` is imported inside other handlers; keep that pattern
    global _mute_summary_cache, _mute_summary_fetched_at

    if not AGELIA_NOTIFY_ENABLED:
        return []

    now = time.monotonic()
    if now - _mute_summary_fetched_at < _MUTE_SUMMARY_TTL:
        return _mute_summary_cache

    try:
        client = _get_agelia_client()
        resp = await client.get(
            f"{AGELIA_API_URL}/api/integrations/mute/active-summary",
            headers={"X-Integration-Key": AGELIA_API_KEY},
            timeout=5.0,
        )
        if resp.status_code == 200:
            _mute_summary_cache = resp.json().get("data", {}).get("rules", [])
            _mute_summary_fetched_at = now
        else:
            logger.warning(
                "Mute summary refresh got HTTP %s (using stale cache, %d rules)",
                resp.status_code, len(_mute_summary_cache)
            )
    except Exception as exc:
        logger.warning("Mute summary refresh failed (using stale cache): %s", exc)

    return _mute_summary_cache


def _invalidate_mute_summary_cache() -> None:
    """Force the next _get_mute_summary() call to refetch from Agelia.

    Called after Pallas-initiated mute actions so the change reflects on the
    next alert list fetch (within ~1 second) rather than waiting for the
    60-second TTL.
    """
    global _mute_summary_fetched_at
    _mute_summary_fetched_at = 0.0


# ─── Blocked IPs cache (mirrors the mute summary pattern) ─────────────────────
# Pallas's alert list view annotates each alert with whether its src_ip is
# currently blocked at AWS. The set of blocked IPs is fetched from Agelia
# and cached here. Same TTL + invalidation pattern as the mute summary.
_blocked_ips_cache: set = set()
_blocked_ips_fetched_at: float = 0.0
_BLOCKED_IPS_TTL: float = 60.0


async def _get_blocked_ips_set() -> set:
    """Return cached set of currently-blocked IPs from Agelia.

    Used by the alert fetch handler to annotate alerts whose src_ip is
    currently blocked at AWS WAF / Network Firewall. 60-second TTL bounds
    drift; Pallas-initiated blocks invalidate immediately.
    """
    import time
    global _blocked_ips_cache, _blocked_ips_fetched_at

    if not AGELIA_NOTIFY_ENABLED:
        return set()

    now = time.monotonic()
    if now - _blocked_ips_fetched_at < _BLOCKED_IPS_TTL:
        return _blocked_ips_cache

    try:
        client = _get_agelia_client()
        resp = await client.get(
            f"{AGELIA_API_URL}/api/integrations/firewall/active-summary",
            headers={"X-Integration-Key": AGELIA_API_KEY},
            timeout=5.0,
        )
        if resp.status_code == 200:
            ips = resp.json().get("data", {}).get("ips", [])
            _blocked_ips_cache = set(ips)
            _blocked_ips_fetched_at = now
        else:
            logger.warning(
                "Blocked IPs refresh got HTTP %s (using stale cache, %d IPs)",
                resp.status_code, len(_blocked_ips_cache)
            )
    except Exception as exc:
        logger.warning("Blocked IPs refresh failed (using stale cache): %s", exc)

    return _blocked_ips_cache


def _invalidate_blocked_ips_cache() -> None:
    """Force the next _get_blocked_ips_set() call to refetch from Agelia."""
    global _blocked_ips_fetched_at
    _blocked_ips_fetched_at = 0.0


def _find_matching_mute_rule(alert: dict, mute_rules: list):
    """Return the FIRST matching mute rule (or None) for this alert.

    Returns the rule dict (including 'classification') so the UI can render
    "Muted (false_positive)" inline. Each rule's conditions are checked with
    AND logic — every condition must match for the rule to match.
    """
    field_map = {
        'rule_id':      lambda a: str(a.get('rule_id', '')),
        'signature_id': lambda a: str(a.get('rule_id', '')),
        'source_ip':    lambda a: a.get('source_ip', ''),
        'dest_ip':      lambda a: a.get('dest_ip', ''),
        'agent_name':   lambda a: a.get('agent_name', ''),
        'severity':     lambda a: a.get('severity', '').lower(),
        'category':     lambda a: a.get('category', ''),
    }
    for rule in mute_rules:
        conditions = rule.get('conditions', {})
        if not conditions:
            continue
        matched = all(
            field_map[k](alert) == str(v)
            for k, v in conditions.items()
            if k in field_map
        )
        if matched:
            return rule
    return None


def _alert_matches_agelia_mutes(alert: dict, mute_rules: list) -> bool:
    """Backwards-compatible wrapper — returns bool. Prefer _find_matching_mute_rule
    when you need the matched rule's classification.
    """
    return _find_matching_mute_rule(alert, mute_rules) is not None


async def _notify_agelia(action: str, analyst: str = '', **kwargs) -> None:
    """REMOVED. Use _call_agelia() with the appropriate Agelia endpoint directly.

    All analyst response actions (block, mute, enrich) now forward the analyst's
    Keycloak JWT via _call_agelia(). This function is a tombstone — calls log a
    warning and return without action. Will be deleted in the next minor release.
    """
    logger.warning(
        "_notify_agelia() is removed. Caller must migrate to _call_agelia(). action=%s",
        action,
    )
    return None


# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Initialize graceful shutdown manager
shutdown_manager = GracefulShutdown()
logger.info("Graceful shutdown manager initialized")


# CORS middleware for remote access with security
def validate_cors_origins(origins_config: str) -> List[str]:
    """Validate and parse CORS origins configuration."""
    if not origins_config or origins_config.strip() == "*":
        # Only allow wildcard in development
        if os.getenv("ENVIRONMENT") == "development":
            return ["*"]
        else:
            # In production, default to common Claude origins
            return ["https://claude.ai", "https://claude.anthropic.com"]

    origins = []
    for origin in origins_config.split(","):
        origin = origin.strip()
        # Validate origin format
        if origin.startswith(("http://", "https://")) or origin == "*":
            # Parse and validate URL structure
            if origin != "*":
                try:
                    parsed = urlparse(origin)
                    if parsed.netloc:
                        origins.append(origin)
                except Exception:
                    continue
            else:
                origins.append(origin)

    return origins if origins else ["https://claude.ai"]


allowed_origins = validate_cors_origins(config.ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "x-api-key",
        "MCP-Protocol-Version",
        "Mcp-Session-Id",
        "Last-Event-ID"
    ],
    expose_headers=["Mcp-Session-Id", "MCP-Protocol-Version", "Content-Type"],
    max_age=600
)

# MCP Protocol Error Codes
MCP_ERRORS = {
    "PARSE_ERROR": -32700,
    "INVALID_REQUEST": -32600,
    "METHOD_NOT_FOUND": -32601,
    "INVALID_PARAMS": -32602,
    "INTERNAL_ERROR": -32603,
    "TIMEOUT": -32001,
    "CANCELLED": -32002,
    "RESOURCE_NOT_FOUND": -32003
}


def create_error_response(request_id: Optional[Union[str, int]], code: int, message: str, data: Any = None) -> MCPResponse:
    """Create MCP error response."""
    error = MCPError(code=code, message=message, data=data)
    return MCPResponse(id=request_id, error=error.dict())


def create_success_response(request_id: Optional[Union[str, int]], result: Any) -> MCPResponse:
    """Create MCP success response."""
    return MCPResponse(id=request_id, result=result)


def validate_protocol_version(version: Optional[str]) -> str:
    """
    Validate and normalize MCP protocol version.
    Returns the validated version or defaults to 2025-03-26 for backwards compatibility.
    """
    if not version:
        return "2025-03-26"

    if version in SUPPORTED_PROTOCOL_VERSIONS:
        return version

    if version > MCP_PROTOCOL_VERSION:
        logger.warning(f"Client requested newer protocol version {version}, using {MCP_PROTOCOL_VERSION}")
        return MCP_PROTOCOL_VERSION

    logger.warning(f"Unsupported protocol version {version}, using 2025-03-26 for compatibility")
    return "2025-03-26"


# ============================================================================
# MCP PROTOCOL HANDLERS
# ============================================================================

async def handle_initialize(params: Dict[str, Any], session: MCPSession) -> Dict[str, Any]:
    """Handle MCP initialize method."""
    protocol_version = params.get("protocolVersion", "")
    capabilities = params.get("capabilities", {})
    client_info = params.get("clientInfo", {})

    session.capabilities = capabilities
    session.client_info = client_info

    server_capabilities = {
        "logging": {},
        "prompts": {
            "listChanged": True
        },
        "resources": {
            "subscribe": True,
            "listChanged": True
        },
        "tools": {
            "listChanged": True
        }
    }

    server_info = {
        "name": "Wazuh MCP Server",
        "version": "4.0.3",
        "vendor": "GenSec AI",
        "description": "MCP-compliant remote server for Wazuh SIEM integration with hybrid intelligence"
    }

    return {
        "protocolVersion": "2025-03-26",
        "capabilities": server_capabilities,
        "serverInfo": server_info,
        "instructions": "Connected to Wazuh MCP Server with hybrid tool selection. Use natural language queries for security analysis."
    }


async def handle_tools_list(params: Dict[str, Any], session: MCPSession) -> Dict[str, Any]:
    """Handle tools/list method - All 29 Wazuh Security Tools."""
    tools = [
        # Alert Management Tools (4 tools)
        {
            "name": "get_wazuh_alerts",
            "description": "Retrieve Wazuh security alerts with optional filtering",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
                    "rule_id": {"type": "string", "description": "Filter by specific rule ID"},
                    "level": {"type": "string", "description": "Filter by alert level (e.g., '12', '10+')"},
                    "agent_id": {"type": "string", "description": "Filter by agent ID"},
                    "timestamp_start": {"type": "string", "description": "Start timestamp (ISO format)"},
                    "timestamp_end": {"type": "string", "description": "End timestamp (ISO format)"}
                },
                "required": []
            }
        },
        {
            "name": "get_wazuh_alert_summary",
            "description": "Get a summary of Wazuh alerts grouped by specified field",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                    "group_by": {"type": "string", "default": "rule.level"}
                },
                "required": []
            }
        },
        {
            "name": "analyze_alert_patterns",
            "description": "Analyze alert patterns to identify trends and anomalies",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 5}
                },
                "required": []
            }
        },
        {
            "name": "search_security_events",
            "description": "Search for specific security events across all Wazuh data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or pattern"},
                    "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
                    "rule_id": {"type": "string", "description": "Exact rule ID filter (e.g., 80255)"},
                    "level": {"type": "integer", "description": "Exact rule level filter (e.g., 10)"},
                    "agent_id": {"type": "string", "description": "Filter by agent ID"},
                    "mitre_id": {"type": "string", "description": "MITRE technique ID (e.g., T1110)"},
                    "mitre_tactic": {"type": "string", "description": "MITRE tactic name (e.g., Credential Access)"},
                    "mitre_technique": {"type": "string", "description": "MITRE technique name (e.g., Brute Force)"}
                },
                "required": ["query"]
            }
        },

        # Agent Management Tools (6 tools)
        {
            "name": "get_wazuh_agents",
            "description": "Retrieve information about Wazuh agents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Specific agent ID to query"},
                    "status": {"type": "string", "enum": ["active", "disconnected", "never_connected"], "description": "Filter by agent status"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                },
                "required": []
            }
        },
        {
            "name": "get_wazuh_running_agents",
            "description": "Get list of currently running/active Wazuh agents",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "check_agent_health",
            "description": "Check the health status of a specific Wazuh agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "ID of the agent to check"}
                },
                "required": ["agent_id"]
            }
        },
        {
            "name": "get_agent_processes",
            "description": "Get running processes from a specific Wazuh agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "ID of the agent"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                },
                "required": ["agent_id"]
            }
        },
        {
            "name": "get_agent_ports",
            "description": "Get open ports from a specific Wazuh agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "ID of the agent"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                },
                "required": ["agent_id"]
            }
        },
        {
            "name": "get_agent_configuration",
            "description": "Get configuration details for a specific Wazuh agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "ID of the agent"}
                },
                "required": ["agent_id"]
            }
        },

        # Vulnerability Management Tools (3 tools)
        {
            "name": "get_wazuh_vulnerabilities",
            "description": "Retrieve vulnerability information from Wazuh Indexer (requires WAZUH_INDEXER_HOST configuration)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Filter by specific agent ID"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Filter by severity level"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 5000}
                },
                "required": []
            }
        },
        {
            "name": "get_wazuh_critical_vulnerabilities",
            "description": "Get critical vulnerabilities from Wazuh Indexer (requires WAZUH_INDEXER_HOST configuration)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 500}
                },
                "required": []
            }
        },
        {
            "name": "get_wazuh_vulnerability_summary",
            "description": "Get vulnerability summary statistics from Wazuh Indexer (requires WAZUH_INDEXER_HOST configuration)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "enum": ["1d", "7d", "30d"], "default": "7d"}
                },
                "required": []
            }
        },

        # Security Analysis Tools (6 tools)
        {
            "name": "analyze_security_threat",
            "description": "Analyze a security threat indicator using AI-powered analysis",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "The threat indicator to analyze (IP, hash, domain)"},
                    "indicator_type": {"type": "string", "enum": ["ip", "hash", "domain", "url"], "default": "ip"}
                },
                "required": ["indicator"]
            }
        },
        {
            "name": "check_ioc_reputation",
            "description": "Check reputation of an Indicator of Compromise (IoC)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "The IoC to check (IP, domain, hash, etc.)"},
                    "indicator_type": {"type": "string", "enum": ["ip", "domain", "hash", "url"], "default": "ip"}
                },
                "required": ["indicator"]
            }
        },
        {
            "name": "perform_risk_assessment",
            "description": "Perform comprehensive risk assessment for agents or the entire environment",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Specific agent ID to assess (if None, assess entire environment)"}
                },
                "required": []
            }
        },
        {
            "name": "get_top_security_threats",
            "description": "Get top security threats based on alert frequency and severity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                },
                "required": []
            }
        },
        {
            "name": "generate_security_report",
            "description": "Generate system-wide comprehensive security report for all agents or the entire environment",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["daily", "weekly", "monthly", "incident"], "default": "daily"},
                    "include_recommendations": {"type": "boolean", "default": True}
                },
                "required": []
            }
        },
        {
            "name": "run_compliance_check",
            "description": "Run compliance check against security frameworks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "framework": {"type": "string", "enum": ["PCI-DSS", "HIPAA", "SOX", "GDPR", "NIST"], "default": "PCI-DSS"},
                    "agent_id": {"type": "string", "description": "Specific agent ID to check (if None, check entire environment)"}
                },
                "required": []
            }
        },

        # System Monitoring Tools (10 tools)
        {
            "name": "get_wazuh_statistics",
            "description": "Get comprehensive Wazuh statistics and metrics",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_weekly_stats",
            "description": "Get weekly statistics from Wazuh including alerts, agents, and trends",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_cluster_health",
            "description": "Get Wazuh cluster health information",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_cluster_nodes",
            "description": "Get information about Wazuh cluster nodes",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_rules_summary",
            "description": "Get summary of Wazuh rules and their effectiveness",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_remoted_stats",
            "description": "Get Wazuh remoted (agent communication) statistics",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_wazuh_log_collector_stats",
            "description": "Get Wazuh log collector statistics",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "search_wazuh_manager_logs",
            "description": "Search Wazuh manager logs for specific patterns",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query/pattern"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_wazuh_manager_error_logs",
            "description": "Get recent error logs from Wazuh manager",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                },
                "required": []
            }
        },
        {
            "name": "validate_wazuh_connection",
            "description": "Validate connection to Wazuh server and return status",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
    ]

    # Suricata IDS Tools (10 tools) — only listed when SURICATA_ENABLED=true
    if suricata_client is not None:
        tools.extend([
            {
                "name": "get_suricata_alerts",
                "description": "Retrieve Suricata IDS alerts with optional filtering by severity, category, signature, or IP",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
                        "severity": {"type": "string", "description": "Filter by severity (1=Critical, 2=High, 3=Medium, 4+=Low)"},
                        "category": {"type": "string", "description": "Filter by alert category"},
                        "signature": {"type": "string", "description": "Filter by signature name"},
                        "src_ip": {"type": "string", "description": "Filter by source IP"},
                        "dest_ip": {"type": "string", "description": "Filter by destination IP"},
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_alert_summary",
                "description": "Get Suricata IDS alert summary with severity breakdown, top categories, top signatures, and timeline",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_critical_alerts",
                "description": "Get critical severity (1) Suricata IDS alerts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_high_alerts",
                "description": "Get high severity (2) Suricata IDS alerts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_network_analysis",
                "description": "Get Suricata network analysis: top source IPs, destination IPs, services, and hostnames",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
                    },
                    "required": []
                }
            },
            {
                "name": "search_suricata_alerts",
                "description": "Search Suricata IDS alerts by signature, category, or IP text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (signature, category, IP)"},
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_suricata_top_signatures",
                "description": "Get top firing Suricata IDS signatures ranked by count",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_top_attackers",
                "description": "Get top source IPs generating Suricata IDS alerts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_health",
                "description": "Check Suricata Elasticsearch cluster health and connection status",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_suricata_category_breakdown",
                "description": "Get Suricata IDS alert breakdown by category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            # Suricata Deep Visibility Tools (7 tools) — HTTP, TLS, MITRE, JA3
            {
                "name": "get_suricata_http_analysis",
                "description": "Analyze Suricata HTTP traffic: top URLs, methods, status codes, user agents, and hostnames",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "Number of top items per category"}
                    },
                    "required": []
                }
            },
            {
                "name": "search_suricata_http",
                "description": "Search Suricata HTTP events by URL, method, status code, user agent, or hostname",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Filter by URL path (partial match)"},
                        "method": {"type": "string", "description": "Filter by HTTP method (GET, POST, etc.)"},
                        "status_code": {"type": "integer", "description": "Filter by HTTP status code (200, 404, etc.)"},
                        "user_agent": {"type": "string", "description": "Filter by user agent string (partial match)"},
                        "hostname": {"type": "string", "description": "Filter by target hostname"},
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_tls_analysis",
                "description": "Analyze Suricata TLS/SSL traffic: version distribution, JA3/JA3S/JA4 fingerprints, and destination services",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "Number of top items per category"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_mitre_mapping",
                "description": "Get MITRE ATT&CK tactics and techniques from Suricata alert metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_ja3_fingerprints",
                "description": "Deep JA3/JA3S TLS fingerprint analysis with associated source IPs, destination IPs, and services",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "Number of top fingerprints"},
                        "ja3_hash": {"type": "string", "description": "Filter by specific JA3 hash for detailed investigation"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_suspicious_activity",
                "description": "Detect suspicious activity: scanner user agents, legacy TLS versions, unusual HTTP methods, and HTTP errors",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_traffic_overview",
                "description": "Get Suricata traffic overview: event type distribution, protocol breakdown, service map, and traffic locality",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_ja4_analysis",
                "description": "Deep JA4 TLS fingerprint analysis with associated source IPs, destination IPs, services, and TLS versions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "Number of top fingerprints"},
                        "ja4_fingerprint": {"type": "string", "description": "Filter by specific JA4 fingerprint for detailed investigation"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_suricata_flow_analysis",
                "description": "Analyze network flow/conversation patterns between IP pairs: alert counts, signatures, protocols, and ports per IP pair",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d"], "default": "24h"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20, "description": "Number of top conversations"},
                        "src_ip": {"type": "string", "description": "Filter by source IP"},
                        "dest_ip": {"type": "string", "description": "Filter by destination IP"}
                    },
                    "required": []
                }
            },
        ])

    return {"tools": tools}


async def handle_tools_call(params: Dict[str, Any], session: MCPSession) -> Dict[str, Any]:
    """Handle tools/call method - Execute MCP tools."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if not tool_name:
        raise ValueError("Tool name is required")

    validate_input(tool_name, max_length=100)

    try:
        # Alert Management Tools
        if tool_name == "get_wazuh_alerts":
            limit = arguments.get("limit", 100)
            rule_id = arguments.get("rule_id")
            level = arguments.get("level")
            agent_id = arguments.get("agent_id")
            timestamp_start = arguments.get("timestamp_start")
            timestamp_end = arguments.get("timestamp_end")
            result = await wazuh_client.get_alerts(
                limit=limit, rule_id=rule_id, level=level,
                agent_id=agent_id, timestamp_start=timestamp_start,
                timestamp_end=timestamp_end
            )
            formatted_response = format_alert_response(result)
            return {"content": [{"type": "text", "text": formatted_response}]}

        elif tool_name == "get_wazuh_alert_summary":
            time_range = arguments.get("time_range", "24h")
            group_by = arguments.get("group_by", "rule.level")
            result = await wazuh_client.get_alert_summary(time_range, group_by)
            return {"content": [{"type": "text", "text": f"Alert Summary:\n{json.dumps(result, indent=2)}"}]}

        elif tool_name == "analyze_alert_patterns":
            time_range = arguments.get("time_range", "24h")
            min_frequency = arguments.get("min_frequency", 5)
            result = await wazuh_client.analyze_alert_patterns(time_range, min_frequency)
            return {"content": [{"type": "text", "text": f"Alert Patterns:\n{json.dumps(result, indent=2)}"}]}

        elif tool_name == "search_security_events":
            query = arguments.get("query")
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 100)
            extra = {}
            for k in ("rule_id", "level", "agent_id", "mitre_id", "mitre_tactic", "mitre_technique"):
                if arguments.get(k):
                    extra[k] = arguments[k]
            result = await wazuh_client.search_security_events(query, time_range, limit, **extra)
            return {"content": [{"type": "text", "text": f"Security Events:\n{json.dumps(result, indent=2)}"}]}

        # Agent Management Tools
        elif tool_name == "get_wazuh_agents":
            agent_id = arguments.get("agent_id")
            status = arguments.get("status")
            limit = arguments.get("limit", 100)

            # Normalize/clean
            if agent_id in (None, "", "null", "None"):
                agent_id = None
            if status in (None, "", "null", "None"):
                status = None

            # Only pass status if present
            kwargs = {"limit": limit}
            if status:
                kwargs["status"] = status

            result = await wazuh_client.get_agents(agent_id=agent_id, **kwargs)
            formatted_response = format_agent_response(result)
            # Include raw data alongside text so correlation engine can extract agent items
            return {"content": [{"type": "text", "text": formatted_response}], "data": result.get("data", {})}

        elif tool_name == "get_wazuh_running_agents":
            result = await wazuh_client.get_running_agents()
            formatted_text = format_agent_response(result)
            return {"content": [{"type": "text", "text": formatted_text}], "data": result.get("data", {})}

        elif tool_name == "check_agent_health":
            agent_id = arguments.get("agent_id")
            logger.error(f" Health check for agent: {agent_id}")
            result = await wazuh_client.check_agent_health(agent_id)

            # Extract agent data properly
            agent_data = result.get("data", {})
            affected_items = agent_data.get("affected_items", [])

            if not affected_items:
                return {"content": [{"type": "text", "text": f" Agent {agent_id} not found or no data available"}]}

            agent = affected_items[0]

            health_report = f"""## Agent Health Report: {agent_id}

**Agent Name:** {agent.get('name', 'N/A')}
**Status:** {agent.get('status', 'unknown')}
**IP:** {agent.get('ip', 'N/A')}
**OS:** {agent.get('os', {}).get('name', 'N/A')} {agent.get('os', {}).get('version', '')}
**Version:** {agent.get('version', 'N/A')}
**Last Keep Alive:** {agent.get('lastKeepAlive', 'N/A')}
**Registration Date:** {agent.get('dateAdd', 'N/A')}

### Configuration Status
- **Config Sum:** {agent.get('configSum', 'N/A')}
- **Merged Sum:** {agent.get('mergedSum', 'N/A')}

### System Info
- **Architecture:** {agent.get('os', {}).get('arch', 'N/A')}
- **Platform:** {agent.get('os', {}).get('platform', 'N/A')}
"""

 # Add logcollector stats if available formatted as readable summary
            if 'logcollector_stats' in agent_data:
                try:
                    lc_stats = agent_data['logcollector_stats']
                    lc_items = lc_stats.get('affected_items', lc_stats if isinstance(lc_stats, list) else [lc_stats])
                    if lc_items and isinstance(lc_items, list) and len(lc_items) > 0:
                        lc = lc_items[0] if isinstance(lc_items[0], dict) else {}
                        global_stats = lc.get('global', {})
                        global_files = global_stats.get('files', [])
                        total_events = sum(f.get('events', 0) for f in global_files if isinstance(f, dict))
                        total_bytes = sum(f.get('bytes', 0) for f in global_files if isinstance(f, dict))
                        interval_stats = lc.get('interval', {})

                        health_report += "\n### Log Collector Stats\n\n"
                        health_report += "| Metric | Value |\n|--------|-------|\n"
                        health_report += f"| **Files Monitored** | {len(global_files)} |\n"
                        health_report += f"| **Total Events** | {total_events:,} |\n"
                        health_report += f"| **Total Bytes** | {total_bytes:,} |\n"
                        if interval_stats:
                            interval_files = interval_stats.get('files', [])
                            interval_events = sum(f.get('events', 0) for f in interval_files if isinstance(f, dict))
                            health_report += f"| **Interval Events** | {interval_events:,} |\n"
                    else:
                        health_report += "\n### Log Collector Stats\n- *No log collector data available*\n"
                except Exception as lc_err:
                    logger.warning(f"Logcollector stats formatting failed: {lc_err}")
                    health_report += "\n### Log Collector Stats\n- *Formatting unavailable*\n"

            return {"content": [{"type": "text", "text": health_report}]}

        elif tool_name == "get_agent_processes":
            agent_id = arguments.get("agent_id")
            limit = arguments.get("limit", 100)
            result = await wazuh_client.get_agent_processes(agent_id, limit)

            processes = result.get("data", {}).get("affected_items", [])
            if not processes:
                return {"content": [{"type": "text", "text": f" No processes for agent {agent_id}"}]}

            output = f"## Processes on Agent {agent_id}\n\n**Total:** {len(processes)}\n\n| PID | Name | User | State | Memory | Command |\n|-----|------|------|-------|--------|---------|\n"
            for p in processes[:20]:
                pid = p.get('pid', 'N/A')
                name = p.get('name', 'N/A')
                user = p.get('euser', 'N/A')
                state = p.get('state', 'N/A')
                mem = f"{p.get('resident', 0):,} KB"
                cmd = (p.get('cmd', 'N/A')[:30] + '...') if len(p.get('cmd', '')) > 30 else p.get('cmd', 'N/A')
                output += f"| {pid} | {name} | {user} | {state} | {mem} | {cmd} |\n"

            output += f"\n*Top 20 of {len(processes)} processes*"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_agent_ports":
            agent_id = arguments.get("agent_id")
            result = await wazuh_client.get_agent_ports(agent_id, limit=100)

            ports = result.get("data", {}).get("affected_items", [])
            if not ports:
                return {"content": [{"type": "text", "text": f" No ports for agent {agent_id}"}]}

            listening = [p for p in ports if p.get('state') == 'listening']
            output = f"## Open Ports on Agent {agent_id}\n\n**Listening Ports:** {len(listening)}\n\n| Protocol | IP | Port | Process | PID |\n|----------|----------|------|---------|-----|\n"
            for port in listening:
                protocol = port.get('protocol', 'N/A')
                ip = port.get('local', {}).get('ip', 'N/A')
                port_num = port.get('local', {}).get('port', 'N/A')
                process = port.get('process', 'N/A')
                pid = port.get('pid', 'N/A')
                output += f"| {protocol} | {ip} | {port_num} | {process} | {pid} |\n"

            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_agent_configuration":
            agent_id = arguments.get("agent_id")
            result = await wazuh_client.get_agent_configuration(agent_id)
            return {"content": [{"type": "text", "text": f"Agent Configuration:\n{json.dumps(result, indent=2)}"}]}

        # Vulnerability Management Tools
        elif tool_name == "get_wazuh_vulnerabilities":
            agent_id = arguments.get("agent_id")
            severity = arguments.get("severity")
            limit = arguments.get("limit", 10000)
            logger.error(f"TOOL CALL: get_wazuh_vulnerabilities")
            logger.error(f"   Arguments received: {arguments}")
            logger.error(f"   agent_id={agent_id}, severity={severity}, limit={limit}")
            result = await wazuh_client.get_vulnerabilities(agent_id=agent_id, severity=severity, limit=limit)
            formatted_response = format_vulnerability_response(result)
            return {"content": [{"type": "text", "text": formatted_response}]}

        elif tool_name == "get_wazuh_critical_vulnerabilities":
            limit = arguments.get("limit", 10000)
            result = await wazuh_client.get_critical_vulnerabilities(limit)
            formatted_response = format_vulnerability_response(result)
            return {"content": [{"type": "text", "text": formatted_response}]}

        elif tool_name == "get_wazuh_vulnerability_summary":
            time_range = arguments.get("time_range", "7d")
            result = await wazuh_client.get_vulnerability_summary(time_range)
 # Pass the raw data as JSON text the response_formatter will handle formatting
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        # Security Analysis Tools
        elif tool_name == "analyze_security_threat":
            indicator = arguments.get("indicator")
            indicator_type = arguments.get("indicator_type", "ip")
            result = await wazuh_client.analyze_security_threat(indicator, indicator_type)
            return {"content": [{"type": "text", "text": f"Threat Analysis:\n{json.dumps(result, indent=2)}"}]}

        elif tool_name == "check_ioc_reputation":
            indicator = arguments.get("indicator")
            indicator_type = arguments.get("indicator_type", "ip")
            result = await wazuh_client.check_ioc_reputation(indicator, indicator_type)
            return {"content": [{"type": "text", "text": f"IoC Reputation:\n{json.dumps(result, indent=2)}"}]}

        elif tool_name == "perform_risk_assessment":
            agent_id = arguments.get("agent_id")
            try:
                result = await wazuh_client.perform_risk_assessment(agent_id)

                # Format risk assessment into readable output
                if agent_id:
                    output = f"## Risk Assessment Agent {agent_id}\n\n"
                else:
                    output = f"## Risk Assessment Environment Overview\n\n"

                # Extract risk score and level
                risk_score = result.get("risk_score", "N/A")
                risk_level = result.get("risk_level", "Unknown")
                risk_icons = {"critical": "[CRIT]", "high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}
                icon = risk_icons.get(str(risk_level).lower(), "[?]")

                output += f"**Overall Risk:** {icon} **{risk_level}** (Score: {risk_score})\n\n"

                # Agent info
                agent_info = result.get("agent_info", result.get("agent", {}))
                if agent_info:
                    output += "### Agent Details\n\n"
                    output += "| Property | Value |\n|----------|-------|\n"
                    output += f"| **Status** | {agent_info.get('status', 'N/A')} |\n"
                    output += f"| **Name** | {agent_info.get('name', 'N/A')} |\n"
                    output += f"| **IP** | {agent_info.get('ip', 'N/A')} |\n"
                    output += f"| **Version** | {agent_info.get('version', 'N/A')} |\n\n"

                # Risk factors
                factors = result.get("risk_factors", result.get("factors", {}))
                if factors:
                    output += "### Risk Factors\n\n"
                    output += "| Factor | Value |\n|--------|-------|\n"
                    for key, val in factors.items():
                        output += f"| **{key.replace('_', ' ').title()}** | {val} |\n"
                    output += "\n"

                # Recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    output += "### Recommended Actions\n\n"
                    for i, rec in enumerate(recommendations, 1):
                        output += f"{i}. {rec}\n"

                return {"content": [{"type": "text", "text": output}]}

            except Exception as e:
                logger.error(f"Risk assessment failed for agent {agent_id}: {e}")
                if agent_id:
                    output = f"## Risk Assessment Agent {agent_id}\n\n"
                else:
                    output = f"## Risk Assessment Environment Overview\n\n"
                    output += f" Assessment could not be completed: {str(e)}\n\n"
                output += "### Alternative Steps\n"
                output += f"- Try `check agent {agent_id} health` for basic health info\n"
                output += "- Try `show vulnerabilities` for vulnerability overview\n"
                output += "- Verify the agent ID is correct and the agent is registered\n"
                return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_top_security_threats":
            limit = arguments.get("limit", 10)
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client.get_top_security_threats(limit, time_range)

            # Extract and format threats
            threats = result.get("top_threats", {}).get("data", {}).get("affected_items", [])

            output = f"## Top {limit} Security Threats (Last {time_range})\n\n"

            for i, threat in enumerate(threats[:limit], 1):
                rule = threat.get("rule", {})
                data = threat.get("data", {})
                output += f"### {i}. {rule.get('description', 'Unknown Threat')}\n"
                output += f"- **Level:** {rule.get('level')} "
                output += f"- **Rule ID:** {rule.get('id')}\n"
                output += f"- **Agent:** {threat.get('agent', {}).get('name', 'N/A')}\n"
                if data.get('srcuser'):
                    output += f"- **User:** {data.get('srcuser')}\n"
                if data.get('command'):
                    output += f"- **Command:** {data.get('command')}\n"
                mitre = rule.get('mitre', {})
                if mitre.get('technique'):
                    output += f"- **MITRE Technique:** {', '.join(mitre.get('technique', []))}\n"
                output += f"- **Timestamp:** {threat.get('timestamp', 'N/A')}\n\n"

            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "generate_security_report":
            report_type = arguments.get("report_type", "daily")
            include_recommendations = arguments.get("include_recommendations", True)
            result = await wazuh_client.generate_security_report(report_type, include_recommendations)

            # Output as JSON so formatter can parse properly
            output = json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "run_compliance_check":
            framework = arguments.get("framework", "PCI-DSS")
            agent_id = arguments.get("agent_id")
            result = await wazuh_client.run_compliance_check(framework, agent_id)
            return {"content": [{"type": "text", "text": f"Compliance Check:\n{json.dumps(result, indent=2)}"}]}

        # System Monitoring Tools
        elif tool_name == "get_wazuh_statistics":
            result = await wazuh_client.get_wazuh_statistics()
            formatted_response = format_statistics_response(result)
            return {"content": [{"type": "text", "text": formatted_response}]}

        elif tool_name == "get_wazuh_weekly_stats":
            result = await wazuh_client.get_weekly_stats()
            try:
                items = result.get("data", {}).get("affected_items", [])
                # Weekly stats return per-hour data like get_wazuh_statistics
                rule_totals = {}
                for hour_data in items:
                    for alert in (hour_data.get("alerts", []) if isinstance(hour_data, dict) else []):
                        sid = alert.get("sigid", "?")
                        cnt = alert.get("times", 0)
                        rule_totals[sid] = rule_totals.get(sid, 0) + cnt
                total_events = sum(rule_totals.values())
                top_rules = sorted(rule_totals.items(), key=lambda x: x[1], reverse=True)[:10]
                output = "## Weekly Statistics\n\n"
                output += f"**Total Events (7 days):** {total_events:,}\n"
                output += f"**Unique Rules Triggered:** {len(rule_totals)}\n\n"
                if top_rules:
                    output += "### Top Rules This Week\n\n"
                    output += "| Rule ID | Event Count |\n|---------|-------------|\n"
                    for sid, cnt in top_rules:
                        output += f"| {sid} | {cnt:,} |\n"
            except Exception as e:
                logger.warning(f"Weekly stats formatting error: {e}")
                output = f"Weekly Statistics:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_wazuh_cluster_health":
            result = await wazuh_client.get_cluster_health()
            try:
                data = result.get("data", result)
                output = "## Cluster Health\n\n"
                output += "| Field | Value |\n|-------|-------|\n"
                def _flatten(d, prefix=""):
                    rows = []
                    for k, v in d.items():
                        if k in ("affected_items", "total_affected_items", "failed_items", "total_failed_items"):
                            continue
                        label = (prefix + k).replace("_", " ").title()
                        if isinstance(v, dict):
                            rows.extend(_flatten(v, prefix=k + " "))
                        else:
                            rows.append((label, v))
                    return rows
                for label, val in _flatten(data):
                    output += f"| {label} | {val} |\n"
            except Exception as e:
                logger.warning(f"Cluster health formatting error: {e}")
                output = f"Cluster Health:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_wazuh_cluster_nodes":
            result = await wazuh_client.get_cluster_nodes()
            return {"content": [{"type": "text", "text": f"Cluster Nodes:\n{json.dumps(result, indent=2)}"}]}

        elif tool_name == "get_wazuh_rules_summary":
            result = await wazuh_client.get_rules_summary()
            return {"content": [{"type": "text", "text": f"Rules Summary:\n{json.dumps(result, indent=2, default=str)}"}]}

        elif tool_name == "get_wazuh_remoted_stats":
            result = await wazuh_client.get_remoted_stats()
            try:
                s = result.get("data", {}).get("affected_items", [{}])[0]
                def _fmt_bytes(b):
                    b = float(b or 0)
                    if b >= 1_048_576: return f"{b/1_048_576:.1f} MB"
                    if b >= 1_024: return f"{b/1_024:.1f} KB"
                    return f"{b:.0f} B"
                    output = "## Remoted Statistics (Agent Communication)\n\n"
                output += "| Metric | Value |\n|--------|-------|\n"
                output += f"| TCP Sessions Active | {int(s.get('tcp_sessions', 0))} |\n"
                output += f"| Events Received | {int(s.get('evt_count', 0)):,} |\n"
                output += f"| Control Messages | {int(s.get('ctrl_msg_count', 0)):,} |\n"
                output += f"| Discarded Messages | {int(s.get('discarded_count', 0))} |\n"
                output += f"| Bytes Sent | {_fmt_bytes(s.get('sent_bytes', 0))} |\n"
                output += f"| Bytes Received | {_fmt_bytes(s.get('recv_bytes', 0))} |\n"
                output += f"| Queue Size | {int(s.get('queue_size', 0))} / {int(s.get('total_queue_size', 0)):,} |\n"
                output += f"| Ctrl Queue Inserted | {int(s.get('ctrl_msg_queue_inserted', 0)):,} |\n"
                output += f"| Ctrl Queue Processed | {int(s.get('ctrl_msg_queue_processed', 0)):,} |\n"
                output += f"| Ctrl Queue Replaced | {int(s.get('ctrl_msg_queue_replaced', 0))} |\n"
                output += f"| Dequeued After Close | {int(s.get('dequeued_after_close', 0))} |\n"
            except Exception as e:
                logger.warning(f"Remoted stats formatting error: {e}")
                output = f"Remoted Statistics:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_wazuh_log_collector_stats":
            result = await wazuh_client.get_log_collector_stats()
            try:
                lc_items = result.get("data", {}).get("affected_items", [])
                lc = lc_items[0] if lc_items else {}
                global_files   = lc.get("global",   {}).get("files", [])
                interval_files = lc.get("interval", {}).get("files", [])
                def _fmt_b(b):
                    b = float(b or 0)
                    if b >= 1_048_576: return f"{b/1_048_576:.1f} MB"
                    if b >= 1_024:     return f"{b/1_024:.1f} KB"
                    return f"{b:.0f} B"
                total_events   = sum(f.get("events", 0) for f in global_files)
                total_bytes    = sum(f.get("bytes",  0) for f in global_files)
                total_drops    = sum(f.get("drops",  0) for f in global_files)
                interval_evts  = sum(f.get("events", 0) for f in interval_files)
                output = "## Log Collector Statistics\n\n"
                output += "| Metric | Value |\n|--------|-------|\n"
                output += f"| Total Events | {total_events:,} |\n"
                output += f"| Total Data Collected | {_fmt_b(total_bytes)} |\n"
                output += f"| Dropped Events | {total_drops} |\n"
                output += f"| Events (last interval) | {interval_evts:,} |\n"
                top = sorted(global_files, key=lambda x: x.get("events", 0), reverse=True)[:5]
                if top:
                    output += "\n**Top Log Sources (by events):**\n"
                    output += "| Source | Events | Data |\n|--------|--------|------|\n"
                    for f in top:
                        name = f.get("location", f.get("target", "unknown"))
                        output += f"| {name} | {f.get('events', 0):,} | {_fmt_b(f.get('bytes', 0))} |\n"
            except Exception as e:
                logger.warning(f"Log collector stats formatting error: {e}")
                output = f"Log Collector Statistics:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "search_wazuh_manager_logs":
            query = arguments.get("query")
            limit = min(int(arguments.get("limit", 50)), 500)
            result = await wazuh_client.search_manager_logs(query, limit)
            try:
                items = result.get("data", {}).get("affected_items", [])
                total = result.get("data", {}).get("total_affected_items", len(items))
                if not items:
                    output = f"## Manager Logs\n\nNo logs found matching query: `{query}`"
                else:
                    output = f"## Manager Logs `{query}` ({len(items)} of {total} shown)\n\n"
                    for i, log in enumerate(items, 1):
                        ts   = log.get("timestamp", "N/A")
                        tag  = log.get("tag", "N/A")
                        lvl  = log.get("level", "info")
                        desc = log.get("description", "N/A").strip()
                        icon = "[ERR]" if lvl == "error" else "[WARN]" if lvl == "warning" else "[INFO]"
                        output += f"{icon} **{i}. [{ts}]** `{tag}` ({lvl})\n> {desc}\n\n"
            except Exception as e:
                logger.warning(f"Manager logs formatting error: {e}")
                output = f"Manager Logs:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_wazuh_manager_error_logs":
            limit = min(int(arguments.get("limit", 50)), 500)
            result = await wazuh_client.get_manager_error_logs(limit)
            try:
                items = result.get("data", {}).get("affected_items", [])
                total = result.get("data", {}).get("total_affected_items", len(items))
                if not items:
                    output = "## Manager Error Logs\n\nNo error logs found."
                else:
                    output = f"## Manager Error Logs ({len(items)} of {total} shown)\n\n"
                    for i, log in enumerate(items, 1):
                        ts   = log.get("timestamp", "N/A")
                        tag  = log.get("tag", "N/A")
                        desc = log.get("description", "N/A").strip()
                        output += f"**{i}. [{ts}]** `{tag}`\n> {desc}\n\n"
            except Exception as e:
                logger.warning(f"Manager error logs formatting error: {e}")
                output = f"Manager Error Logs:\n{json.dumps(result, indent=2)}"
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "validate_wazuh_connection":
            result = await wazuh_client.validate_connection()
            formatted_response = format_connection_validation_response(result)
            return {"content": [{"type": "text", "text": formatted_response}]}

        # ============================================================
        # ROUND 6: ENTERPRISE SOC ANALYSIS TOOLS
        # ============================================================

        elif tool_name == "get_rule_trigger_analysis":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            result = await wazuh_client.get_rule_trigger_analysis(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_mitre_coverage":
            time_range = arguments.get("time_range", "7d")
            result = await wazuh_client.get_mitre_coverage(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_alert_timeline":
            time_range = arguments.get("time_range", "24h")
            interval = arguments.get("interval", "1h")
            result = await wazuh_client.get_alert_timeline(time_range, interval)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_log_source_health":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client.get_log_source_health(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_decoder_analysis":
            limit = arguments.get("limit", 100)
            result = await wazuh_client.get_decoder_analysis(limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_fim_events":
            agent_id = arguments.get("agent_id")
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 50)
            result = await wazuh_client.get_fim_events(agent_id, time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_sca_results":
            agent_id = arguments.get("agent_id")
            if not agent_id:
                return {"content": [{"type": "text", "text": "Agent ID is required for SCA results. Example: 'sca results for agent 011'"}]}
            result = await wazuh_client.get_sca_results(agent_id)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_agent_inventory":
            agent_id = arguments.get("agent_id")
            if not agent_id:
                return {"content": [{"type": "text", "text": "Agent ID is required for inventory. Example: 'show inventory for agent 011'"}]}
            result = await wazuh_client.get_agent_inventory(agent_id)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_rootcheck_results":
            agent_id = arguments.get("agent_id")
            if not agent_id:
                return {"content": [{"type": "text", "text": "Agent ID is required for rootcheck. Example: 'rootcheck results for agent 011'"}]}
            result = await wazuh_client.get_rootcheck_results(agent_id)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "correlate_alerts_vulnerabilities":
            time_range = arguments.get("time_range", "24h")
            min_severity = arguments.get("min_severity", "high")
            result = await wazuh_client.correlate_alerts_vulnerabilities(time_range, min_severity)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_behavioral_baseline":
            agent_id = arguments.get("agent_id")
            baseline_days = arguments.get("baseline_days", 7)
            result = await wazuh_client.get_behavioral_baseline(agent_id, baseline_days)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "generate_wazuh_decoder":
            raw_log = arguments.get("raw_log")
            if not raw_log:
                return {"content": [{"type": "text", "text": "Error: raw_log is required for decoder generation"}]}
            log_format = arguments.get("log_format", "syslog")
            device_type = arguments.get("device_type")
            vendor = arguments.get("vendor")
            expected_fields = arguments.get("expected_fields")
            # Initialize LLM client for decoder generation
            llm = LLMClient(
                base_url=os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1"),
                model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus"),
                api_key=os.getenv("LLM_API_KEY", ""),
            )
            await llm.initialize()
            try:
                result = await wazuh_client.generate_decoder(
                    raw_log=raw_log, log_format=log_format,
                    device_type=device_type, vendor=vendor,
                    expected_fields=expected_fields, llm_client=llm
                )
                output = json.dumps(result, indent=2, default=str)
                return {"content": [{"type": "text", "text": output}]}
            finally:
                await llm.close()

        # ================================================================
        # Suricata IDS Tools (10 tools)
        # ================================================================
        elif tool_name == "get_suricata_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            limit = arguments.get("limit", 100)
            severity = arguments.get("severity")
            category = arguments.get("category")
            signature = arguments.get("signature")
            src_ip = arguments.get("src_ip")
            dest_ip = arguments.get("dest_ip")
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_alerts(
                limit=limit, severity=severity, category=category,
                signature=signature, src_ip=src_ip, dest_ip=dest_ip,
                time_range=time_range
            )
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_alert_summary":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_alert_summary(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_critical_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            limit = arguments.get("limit", 50)
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_critical_alerts(limit=limit, time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_high_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            limit = arguments.get("limit", 50)
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_high_alerts(limit=limit, time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_network_analysis":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 10)
            result = await suricata_client.get_network_analysis(time_range=time_range, limit=limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "search_suricata_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            query = arguments.get("query")
            if not query:
                return {"content": [{"type": "text", "text": "Error: query is required for Suricata search. Example: 'search suricata for DNS'"}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 100)
            result = await suricata_client.search_alerts(query=query, time_range=time_range, limit=limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_top_signatures":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_alert_summary(time_range)
            # Extract just the top signatures from the summary
            signatures_data = {
                "top_signatures": result.get("top_signatures", []),
                "total_alerts": result.get("total_alerts", 0),
                "time_range": time_range
            }
            output = json.dumps(signatures_data, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_top_attackers":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 10)
            result = await suricata_client.get_network_analysis(time_range=time_range, limit=limit)
            # Extract just the top source IPs
            attackers_data = {
                "top_attackers": result.get("top_source_ips", []),
                "time_range": time_range
            }
            output = json.dumps(attackers_data, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_health":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            result = await suricata_client.health_check()
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_category_breakdown":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_alert_summary(time_range)
            # Extract just the category breakdown
            category_data = {
                "top_categories": result.get("top_categories", []),
                "severity_breakdown": result.get("severity_breakdown", {}),
                "total_alerts": result.get("total_alerts", 0),
                "time_range": time_range
            }
            output = json.dumps(category_data, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        # ================================================================
        # Suricata Deep Visibility Tools (7 tools)
        # ================================================================
        elif tool_name == "get_suricata_http_analysis":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 10)
            result = await suricata_client.get_http_analysis(time_range=time_range, limit=limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "search_suricata_http":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 100)
            url = arguments.get("url")
            method = arguments.get("method")
            status_code = arguments.get("status_code")
            user_agent = arguments.get("user_agent")
            hostname = arguments.get("hostname")
            result = await suricata_client.search_http_events(
                time_range=time_range, limit=limit, url=url, method=method,
                status_code=status_code, user_agent=user_agent, hostname=hostname
            )
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_tls_analysis":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 10)
            result = await suricata_client.get_tls_analysis(time_range=time_range, limit=limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_mitre_mapping":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_mitre_analysis(time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_ja3_fingerprints":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            ja3_hash = arguments.get("ja3_hash")
            result = await suricata_client.get_ja3_analysis(time_range=time_range, limit=limit, ja3_hash=ja3_hash)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_suspicious_activity":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_suspicious_activity(time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_traffic_overview":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_traffic_overview(time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        # --- Pallas 3.3: New Suricata tools ---
        elif tool_name == "get_suricata_ja4_analysis":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            ja4_fingerprint = arguments.get("ja4_fingerprint")
            result = await suricata_client.get_ja4_analysis(time_range=time_range, limit=limit, ja4_fingerprint=ja4_fingerprint)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_flow_analysis":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            src_ip = arguments.get("src_ip")
            dest_ip = arguments.get("dest_ip")
            result = await suricata_client.get_flow_analysis(time_range=time_range, limit=limit, src_ip=src_ip, dest_ip=dest_ip)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    except IndexerNotConfiguredError as e:
        logger.warning(f"Indexer not configured for tool {tool_name}: {e}")
        return {"content": [{"type": "text", "text": f" Configuration Error:\n\n{str(e)}"}], "isError": True}

    except Exception as e:
        logger.error(f"Tool execution error in {tool_name}: {e}")
        return {"content": [{"type": "text", "text": f" Error: {str(e)}\n\nTool: {tool_name}\n\nPlease try again or contact support."}], "isError": True}


# MCP Method Registry
MCP_METHODS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


async def process_mcp_request(request: MCPRequest, session: MCPSession) -> MCPResponse:
    """Process individual MCP request."""
    try:
        if request.method not in MCP_METHODS:
            return create_error_response(
                request.id,
                MCP_ERRORS["METHOD_NOT_FOUND"],
                f"Method '{request.method}' not found"
            )

        handler = MCP_METHODS[request.method]
        result = await handler(request.params or {}, session)

        return create_success_response(request.id, result)

    except ValueError as e:
        return create_error_response(
            request.id,
            MCP_ERRORS["INVALID_PARAMS"],
            str(e)
        )
    except Exception as e:
        logger.error(f"Internal error processing {request.method}: {e}")
        return create_error_response(
            request.id,
            MCP_ERRORS["INTERNAL_ERROR"],
            "Internal server error"
        )


async def generate_sse_events(session: MCPSession):
    """Generate Server-Sent Events for MCP."""
    yield f"event: session\ndata: {json.dumps(session.to_dict())}\n\n"
    yield f"event: capabilities\ndata: {json.dumps({'tools': True, 'resources': True})}\n\n"

    while True:
        yield f"event: keepalive\ndata: {json.dumps({'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
        await asyncio.sleep(30)


# ============================================================================
# MCP TOOL EXECUTOR - Used by Query Orchestrator
# ============================================================================

async def execute_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Execute an MCP tool and return the result.
    This wrapper allows the QueryOrchestrator to call MCP tools.
    """
    # Create a dummy session for tool execution
    session = MCPSession("orchestrator_session", None)
    session.authenticated = True

    try:
        result = await handle_tools_call({"name": tool_name, "arguments": arguments}, session)
        return result
    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {e}")
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}


# ============================================================================
# WEBSOCKET CHAT ENDPOINT - Uses Hybrid Query Orchestrator
# ============================================================================

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for browser-based SOC Chat.
    Uses QueryOrchestrator with hybrid intelligence (rules + LLM).
    Supports multi-turn conversation with ConversationMemory.
    """
    global _query_orchestrator, _conversation_memory

    await websocket.accept()

    # Generate session ID for this WebSocket connection
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket chat session started: {session_id[:8]} from {websocket.client}")

    # Track the currently running query task so it can be cancelled
    active_task: Optional[asyncio.Task] = None

    try:
        # Check if orchestrator is initialized
        if _query_orchestrator is None:
            await websocket.send_json({
                "role": "bot",
                "message": " Chat system is initializing. Please wait a moment and try again.",
                "warn": True
            })
            await websocket.close()
            return

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Handle JSON control messages (cancel, ping)
            try:
                msg = json.loads(data)
                if isinstance(msg, dict):
                    msg_type = msg.get("type", "")

                    # Cancel active query
                    if msg_type == "cancel":
                        if active_task and not active_task.done():
                            active_task.cancel()
                            logger.info(f"Chat [{session_id[:8]}]: Query cancelled by client")
                            await websocket.send_json({
                                "role": "bot",
                                "message": "Query cancelled.",
                                "warn": False,
                                "cancelled": True,
                            })
                        continue

                    # Pong for heartbeat
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON — treat as plain-text query

            logger.info(f"Chat [{session_id[:8]}]: {data[:100]}...")

            try:
                # Conversation memory DISABLED — each query is independent
                # (Memory infrastructure kept for future re-enablement)
                context = QueryContext(
                    original_query=data,
                    timestamp=datetime.utcnow(),
                    session_id=session_id,
                    conversation_summary="",
                    carried_entities={},
                    is_follow_up=False,
                    previous_tool=None,
                    previous_result_type=None,
                )

                # Process query in a cancellable task
                active_task = asyncio.get_running_loop().create_task(
                    _query_orchestrator.process_query(data, context)
                )
                try:
                    response = await active_task
                except asyncio.CancelledError:
                    logger.info(f"Chat [{session_id[:8]}]: Query task was cancelled")
                    continue  # Cancel response already sent above
                finally:
                    active_task = None

                # Build v2 structured response with metadata
                ws_response = {
                    "role": "bot",
                    "message": response,
                    "warn": False
                }

                # Extract v2 metadata from orchestrator's last turn data
                turn_data = getattr(_query_orchestrator, '_last_turn_data', None)
                if turn_data:
                    if turn_data.get("metadata"):
                        ws_response["metadata"] = turn_data["metadata"]
                    if turn_data.get("suggestions"):
                        ws_response["suggestions"] = turn_data["suggestions"]

                    # Extract SOC Analysis section from response for separate rendering
                    soc_markers = ["## SOC ANALYSIS", "## 🔍 SOC ANALYSIS"]
                    for marker in soc_markers:
                        if marker in response:
                            parts = response.split(marker, 1)
                            # Find the end of SOC analysis (before suggested queries or metadata footer)
                            soc_text = parts[1]
                            # Strip trailing suggested queries and metadata footer
                            for end_marker in ["**Suggested Follow-up Queries:**", "\n---\n*Selection:"]:
                                end_idx = soc_text.find(end_marker)
                                if end_idx != -1:
                                    soc_text = soc_text[:end_idx]
                            ws_response["soc_analysis"] = soc_text.strip()
                            break

                # Send structured response back to client
                await websocket.send_json(ws_response)

            except Exception as e:
                logger.error(f"Error processing chat message: {e}", exc_info=True)
                await websocket.send_json({
                    "role": "bot",
                    "message": f" An error occurred while processing your query:\n\n{str(e)}\n\nPlease try rephrasing your question or contact support.",
                    "warn": True
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket chat session ended: {session_id[:8]}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cancel any in-flight query on disconnect
        if active_task and not active_task.done():
            active_task.cancel()
        logger.info(f"WebSocket session cleanup: {session_id[:8]}")


# ============================================================================
# STANDARD MCP ENDPOINTS
# ============================================================================

@router.get("/")
@router.post("/")
async def mcp_endpoint(
    request: Request,
    origin: Optional[str] = Header(None),
    accept: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")
):
    """Main MCP protocol endpoint supporting both GET and POST."""
    REQUEST_COUNT.labels(method=request.method, endpoint="/", status_code=200).inc()
    ACTIVE_CONNECTIONS.inc()

    try:
        if not origin:
            raise HTTPException(status_code=403, detail="Origin header required")

        client_ip = request.client.host if request.client else "unknown"
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            headers = {"Retry-After": str(retry_after)} if retry_after else {}
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

        session = await get_or_create_session(mcp_session_id, origin)

        if request.method == "GET":
            if accept and "text/event-stream" in accept:
                response = StreamingResponse(
                    generate_sse_events(session),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Mcp-Session-Id": session.session_id,
                        "Access-Control-Expose-Headers": "Mcp-Session-Id"
                    }
                )
                return response
            else:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "serverInfo": {
                                "name": "Wazuh MCP Server",
                                "version": "4.0.3"
                            },
                            "session": session.to_dict()
                        }
                    },
                    headers={
                        "Mcp-Session-Id": session.session_id,
                        "Access-Control-Expose-Headers": "Mcp-Session-Id"
                    }
                )

        elif request.method == "POST":
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    content=create_error_response(
                        None,
                        MCP_ERRORS["PARSE_ERROR"],
                        "Invalid JSON"
                    ).dict(),
                    status_code=400
                )

            if isinstance(body, list):
                if not body:
                    return JSONResponse(
                        content=create_error_response(
                            None,
                            MCP_ERRORS["INVALID_REQUEST"],
                            "Empty batch request"
                        ).dict(),
                        status_code=400
                    )

                responses = []
                for item in body:
                    try:
                        mcp_request = MCPRequest(**item)
                        response = await process_mcp_request(mcp_request, session)
                        responses.append(response.dict())
                    except ValidationError as e:
                        responses.append(create_error_response(
                            item.get("id") if isinstance(item, dict) else None,
                            MCP_ERRORS["INVALID_REQUEST"],
                            f"Invalid request format: {e}"
                        ).dict())

                return JSONResponse(
                    content=responses,
                    headers={
                        "Mcp-Session-Id": session.session_id,
                        "Access-Control-Expose-Headers": "Mcp-Session-Id"
                    }
                )
            else:
                try:
                    mcp_request = MCPRequest(**body)
                    response = await process_mcp_request(mcp_request, session)
                    return JSONResponse(
                        content=response.dict(),
                        headers={
                            "Mcp-Session-Id": session.session_id,
                            "Access-Control-Expose-Headers": "Mcp-Session-Id"
                        }
                    )
                except ValidationError as e:
                    return JSONResponse(
                        content=create_error_response(
                            body.get("id") if isinstance(body, dict) else None,
                            MCP_ERRORS["INVALID_REQUEST"],
                            f"Invalid request format: {e}"
                        ).dict(),
                        status_code=400
                    )

        else:
            raise HTTPException(status_code=405, detail="Method not allowed")

    finally:
        ACTIVE_CONNECTIONS.dec()


@router.get("/sse")
async def mcp_sse_endpoint(
    request: Request,
    authorization: str = Header(None),
    origin: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")
):
    """Official MCP SSE endpoint following Anthropic standards."""
    await verify_authentication(authorization, config)

    if not origin:
        raise HTTPException(status_code=403, detail="Origin header required")

    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        headers = {"Retry-After": str(retry_after)} if retry_after else {}
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

    REQUEST_COUNT.labels(method="GET", endpoint="/sse", status_code=200).inc()
    ACTIVE_CONNECTIONS.inc()

    try:
        session = await get_or_create_session(mcp_session_id, origin)
        session.authenticated = True

        response = StreamingResponse(
            generate_sse_events(session),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Mcp-Session-Id": session.session_id,
                "Access-Control-Expose-Headers": "Mcp-Session-Id"
            }
        )
        return response

    except Exception as e:
        logger.error(f"SSE endpoint error: {e}")
        raise HTTPException(status_code=50000, detail="SSE stream error")

    finally:
        ACTIVE_CONNECTIONS.dec()


@router.post("/mcp")
@router.get("/mcp")
async def mcp_streamable_http_endpoint(
    request: Request,
    authorization: str = Header(None),
    origin: Optional[str] = Header(None),
    mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    accept: Optional[str] = Header("application/json"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")
):
    """Standard MCP endpoint using Streamable HTTP transport (2025-06-18 spec)."""
    protocol_version = validate_protocol_version(mcp_protocol_version)
    await verify_authentication(authorization, config)

    if not origin:
        raise HTTPException(status_code=403, detail="Origin header required")

    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        headers = {"Retry-After": str(retry_after)} if retry_after else {}
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

    REQUEST_COUNT.labels(method=request.method, endpoint="/mcp", status_code=200).inc()
    ACTIVE_CONNECTIONS.inc()

    try:
        session = await get_or_create_session(mcp_session_id, origin)
        session.authenticated = True

        response_headers = {
            "Mcp-Session-Id": session.session_id,
            "MCP-Protocol-Version": protocol_version,
            "Access-Control-Expose-Headers": "Mcp-Session-Id, MCP-Protocol-Version"
        }

        if request.method == "GET":
            if accept and "text/event-stream" in accept:
                response = StreamingResponse(
                    generate_sse_events(session),
                    media_type="text/event-stream",
                    headers={
                        **response_headers,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
                return response
            else:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "result": {
                            "protocolVersion": protocol_version,
                            "serverInfo": {
                                "name": "Wazuh MCP Server",
                                "version": "4.0.3"
                            },
                            "capabilities": {
                                "tools": True,
                                "resources": True,
                                "prompts": True,
                                "logging": True
                            },
                            "session": session.to_dict()
                        }
                    },
                    headers=response_headers
                )

        elif request.method == "POST":
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    content=create_error_response(
                        None,
                        MCP_ERRORS["PARSE_ERROR"],
                        "Invalid JSON"
                    ).dict(),
                    status_code=400,
                    headers=response_headers
                )

            try:
                mcp_request = MCPRequest(**body) if isinstance(body, dict) else None
            except ValidationError as e:
                return JSONResponse(
                    content=create_error_response(
                        None,
                        MCP_ERRORS["INVALID_REQUEST"],
                        f"Invalid MCP request: {str(e)}"
                    ).dict(),
                    status_code=400,
                    headers=response_headers
                )

            if mcp_request:
                mcp_response = await process_mcp_request(mcp_request, session)
                return JSONResponse(
                    content=mcp_response.dict(),
                    headers=response_headers
                )
            else:
                return JSONResponse(
                    content=create_error_response(
                        None,
                        MCP_ERRORS["INVALID_REQUEST"],
                        "Invalid request format"
                    ).dict(),
                    status_code=400,
                    headers=response_headers
                )

        else:
            raise HTTPException(status_code=405, detail="Method not allowed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        raise HTTPException(status_code=5000, detail="Internal server error")

    finally:
        ACTIVE_CONNECTIONS.dec()


@router.delete("/mcp")
async def close_mcp_session(
    mcp_session_id: str = Header(..., alias="Mcp-Session-Id"),
    authorization: str = Header(None)
):
    """Close MCP session explicitly (2025-06-18 spec)."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        from wazuh_mcp_server.auth import verify_bearer_token
        await verify_bearer_token(authorization)
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        await sessions.remove(mcp_session_id)
        logger.info(f"Session {mcp_session_id} closed via DELETE")
        return Response(status_code=204)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    try:
        wazuh_status = "healthy"
        try:
            await wazuh_client.get_manager_info()
        except Exception as e:
            wazuh_status = f"unhealthy: {str(e)}"

        indexer_status = "not_configured"
        if wazuh_client._indexer_client:
            try:
                health = await wazuh_client._indexer_client.health_check()
                if health.get("status") in ("green", "yellow"):
                    indexer_status = "healthy"
                elif health.get("status") == "red":
                    indexer_status = "degraded"
                else:
                    indexer_status = health.get("status", "unknown")
            except Exception as e:
                indexer_status = f"unhealthy: {str(e)}"

        suricata_status = "not_configured"
        if suricata_client:
            try:
                health = await suricata_client.health_check()
                if health.get("status") in ("green", "yellow"):
                    suricata_status = "healthy"
                elif health.get("status") == "red":
                    suricata_status = "degraded"
                else:
                    suricata_status = health.get("status", "unknown")
            except Exception as e:
                suricata_status = f"unhealthy: {str(e)}"

        all_sessions = await sessions.get_all()
        active_sessions = len([s for s in all_sessions.values() if not s.is_expired()])

        # Get orchestrator statistics
        orchestrator_stats = None
        if _query_orchestrator:
            try:
                orchestrator_stats = _query_orchestrator.get_statistics()
            except Exception:
                pass

        auth_info = {
            "mode": config.AUTH_MODE,
            "bearer_enabled": config.is_bearer,
            "oauth_enabled": config.is_oauth,
            "authless": config.is_authless,
        }
        if config.is_oauth:
            auth_info["oauth_dcr"] = config.OAUTH_ENABLE_DCR
            auth_info["oauth_endpoints"] = ["/oauth/authorize", "/oauth/token", "/oauth/register"]
            auth_info["oauth_discovery"] = "/.well-known/oauth-authorization-server"

        health_response = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.0.3",
            "mcp_protocol_version": MCP_PROTOCOL_VERSION,
            "supported_protocol_versions": SUPPORTED_PROTOCOL_VERSIONS,
            "transport": {
                "streamable_http": "enabled",
                "legacy_sse": "enabled"
            },
            "authentication": auth_info,
            "services": {
                "wazuh_manager": wazuh_status,
                "wazuh_indexer": indexer_status,
                "suricata_ids": suricata_status,
                "mcp": "healthy",
                "query_orchestrator": "initialized" if _query_orchestrator else "not_initialized"
            },
            "vulnerability_tools": {
                "available": wazuh_client._indexer_client is not None,
                "note": "Vulnerability tools require Wazuh Indexer (4.8.0+). Set WAZUH_INDEXER_HOST to enable." if not wazuh_client._indexer_client else "Wazuh Indexer configured"
            },
            "metrics": {
                "active_sessions": active_sessions,
                "total_sessions": len(all_sessions)
            },
            "endpoints": {
                "recommended": "/mcp (Streamable HTTP - 2025-06-18)",
                "legacy": "/sse (SSE only)",
                "chat": "/ws/chat (WebSocket with hybrid intelligence)",
                "authentication": "/auth/token" if config.is_bearer else ("/oauth/token" if config.is_oauth else None),
                "monitoring": ["/health", "/metrics"]
            }
        }

        if orchestrator_stats:
            health_response["orchestrator_statistics"] = orchestrator_stats

        return health_response

    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            },
            status_code=503
        )


# ============================================================================
# PALLAS DASHBOARD REST ENDPOINTS
# These serve the frontend plugin's api-client.ts requests
# ============================================================================

@router.get("/api/meta")
async def get_meta(request: Request):
    """
    Meta endpoint for Pallas Dashboard.
    Returns server stats, config, and integration health for the SOC Overview.
    """
    try:
        # Gather stats from available sources
        stats = {
            "events_total": 0,
            "events_in_window": 0,
            "events_kept": 0,
            "cases": 0,
            "vulns": 0,
        }

        # Try to get alert counts from indexer
        indexer = wazuh_client._indexer_client
        if indexer:
            try:
                await indexer._ensure_initialized()
                # Count total alerts
                alert_url = f"{indexer.base_url}/wazuh-alerts-*/_count"
                resp = await indexer.client.get(alert_url)
                if resp.status_code == 200:
                    stats["events_total"] = resp.json().get("count", 0)

                # Count alerts in last 24h
                from_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                count_body = {
                    "query": {
                        "range": {
                            "timestamp": {"gte": from_time}
                        }
                    }
                }
                resp24 = await indexer.client.post(
                    alert_url, json=count_body,
                    headers={"Content-Type": "application/json"}
                )
                if resp24.status_code == 200:
                    stats["events_in_window"] = resp24.json().get("count", 0)
                    stats["events_kept"] = stats["events_in_window"]

                # Count vulnerabilities
                vuln_url = f"{indexer.base_url}/wazuh-states-vulnerabilities-*/_count"
                resp_vuln = await indexer.client.get(vuln_url)
                if resp_vuln.status_code == 200:
                    stats["vulns"] = resp_vuln.json().get("count", 0)
            except Exception as e:
                logger.warning(f"Meta stats collection partial failure: {e}")

        # Check LLM availability
        llm_enabled = False
        llm_base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
        llm_api_key = os.getenv("LLM_API_KEY", "")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
                llm_resp = await client.get(f"{llm_base_url}/models", headers=headers)
                llm_enabled = llm_resp.status_code == 200
        except Exception:
            pass

        # Check Suricata/CTI status
        cti_enabled = suricata_client is not None
        cti_state = "disabled"
        if suricata_client:
            try:
                health = await suricata_client.health_check()
                cti_state = "connected" if health.get("status") in ("green", "yellow") else "degraded"
                cti_enabled = True
            except Exception:
                cti_state = "error"

        meta_response = {
            "ui_version": "3.5.0",
            "last_update_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "remote": config.WAZUH_HOST,
                "remote_root": None,
                "files_used": [],
                "days": 7,
                "archive_root": "",
            },
            "stats": stats,
            "limits": {
                "max_events_in_memory": 50000,
                "max_bundles": 100,
                "reload_interval_seconds": 300,
                "bundle_window_minutes": 60,
                "keep_hours": 24,
                "store_raw": False,
                "alert_level_threshold": 3,
                "public_ip_malicious_enabled": bool(getattr(wazuh_config, "abuseipdb_api_key", None)),
            },
            "cti_enabled": cti_enabled,
            "cti_state": cti_state,
            "ollama_enabled": llm_enabled,  # kept key name for frontend compat
            "retention_window": {
                "keep_hours": 24,
            },
        }

        return JSONResponse(content=meta_response)

    except Exception as e:
        logger.error(f"Meta endpoint error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@router.get("/api/vulns")
async def get_vulns(
    request: Request,
    offset: int = 0,
    limit: int = 80,
    q: str = "",
):
    """
    Vulnerability list endpoint for the Pallas Vulns Dashboard.
    Returns paginated vulnerability data from Wazuh Indexer.
    """
    indexer = wazuh_client._indexer_client
    if not indexer:
        return JSONResponse(
            content={
                "total": 0,
                "offset": offset,
                "limit": limit,
                "items_count": 0,
                "items": [],
                "message": "Wazuh Indexer not configured. Set WAZUH_INDEXER_HOST to enable vulnerability queries.",
            }
        )

    try:
        await indexer._ensure_initialized()

        # Build OpenSearch query
        if q:
            query = {
                "bool": {
                    "should": [
                        {"wildcard": {"vulnerability.id": f"*{q.upper()}*"}},
                        {"wildcard": {"vulnerability.severity": f"*{q}*"}},
                        {"wildcard": {"package.name": f"*{q}*"}},
                        {"wildcard": {"agent.name": f"*{q.lower()}*"}},
                        {"match_phrase_prefix": {"vulnerability.description": q}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        else:
            query = {"match_all": {}}

        url = f"{indexer.base_url}/wazuh-states-vulnerabilities-*/_search"
        body = {
            "query": query,
            "size": limit,
            "from": offset,
            "sort": [{"vulnerability.detected_at": {"order": "desc", "unmapped_type": "date"}}],
        }

        response = await indexer.client.post(
            url, json=body,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()

        hits = result.get("hits", {})
        total = hits.get("total", {})
        total_count = total.get("value", 0) if isinstance(total, dict) else total

        items = []
        for hit in hits.get("hits", []):
            src = hit.get("_source", {})
            vuln = src.get("vulnerability", {})
            agent = src.get("agent", {})
            pkg = src.get("package", {})
            items.append({
                "id": hit.get("_id", ""),
                "cve": vuln.get("id", "N/A"),
                "package": f"{pkg.get('name', 'unknown')} {pkg.get('version', '')}".strip(),
                "severity": vuln.get("severity", "Unknown"),
                "title": vuln.get("description", vuln.get("id", "No description"))[:200],
                "agent_name": agent.get("name", "Unknown"),
                "agent_id": agent.get("id", ""),
                "timestamp": vuln.get("detected_at", src.get("@timestamp", "")),
            })

        return JSONResponse(content={
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "items_count": len(items),
            "items": items,
        })

    except Exception as e:
        logger.error(f"Vulns endpoint error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@router.get("/api/query_stats")
async def get_query_stats():
    """
    Tool-selection telemetry. Returns:
      - counts per strategy (regex / keyword / LLM / fallback)
      - percentages
      - recent queries that hit LLM fallback (candidates to promote to regex)
      - recent queries that hit safe-default fallback (gaps in coverage)

    Use this weekly to decide which regex patterns to add next.
    """
    global _query_orchestrator
    if not _query_orchestrator:
        return JSONResponse(
            content={"error": "Query orchestrator not initialized"},
            status_code=503,
        )
    try:
        stats = _query_orchestrator.get_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


@router.get("/api/ollama_stats")
async def get_llm_stats():
    """
    LLM stats endpoint for the Pallas Dashboard.
    Returns connection status and basic metrics.
    Kept at /api/ollama_stats for frontend backward compatibility.
    """
    llm_base_url = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
    llm_model = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus")
    llm_api_key = os.getenv("LLM_API_KEY", "")

    stats = {
        "enabled": False,
        "url": llm_base_url,
        "model": llm_model,
        "timeout_s": 180,
        "provider": "cloud",
        "total_requests": 0,
        "in_flight": 0,
        "success_count": 0,
        "error_count": 0,
        "json_ok_count": 0,
        "json_fail_count": 0,
        "last_latency_ms": None,
        "avg_latency_ms": None,
        "last_ok_utc": None,
        "last_error_utc": None,
        "last_error_message": None,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
            resp = await client.get(f"{llm_base_url}/models", headers=headers)
            if resp.status_code == 200:
                stats["enabled"] = True
    except Exception as e:
        stats["last_error_message"] = str(e)
        stats["last_error_utc"] = datetime.now(timezone.utc).isoformat()

    return JSONResponse(content=stats)


@router.post("/api/reload")
async def reload_data():
    """
    Trigger a data reload / cache refresh.
    """
    try:
        # Re-test Wazuh connectivity
        await wazuh_client.get_manager_info()
        return JSONResponse(content={
            "status": "ok",
            "message": "Data reload triggered successfully",
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": f"Reload failed: {str(e)}",
        }, status_code=500)


@router.post("/api/decoder/generate")
async def generate_decoder_endpoint(request: Request):
    """REST endpoint for Decoder Lab — generate Wazuh decoder from raw log."""
    try:
        body = await request.json()
        raw_log = body.get("raw_log")
        if not raw_log:
            return JSONResponse(content={"success": False, "error": "raw_log is required"}, status_code=400)

        hints = body.get("hints", {})
        log_format = hints.get("format", "syslog")
        device_type = hints.get("device_type")
        vendor = hints.get("vendor")
        expected_fields = hints.get("expected_fields")

        llm = LLMClient(
            base_url=os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus"),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        await llm.initialize()
        try:
            result = await wazuh_client.generate_decoder(
                raw_log=raw_log, log_format=log_format,
                device_type=device_type, vendor=vendor,
                expected_fields=expected_fields, llm_client=llm
            )
            return JSONResponse(content=result)
        finally:
            await llm.close()
    except Exception as e:
        logger.error(f"Decoder generation failed: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


# ============================================================================
# ATTACK NARRATIVE — Alert Browser + On-Demand AI Narrative
# ============================================================================


def _extract_source_ip(alert: dict) -> str:
    """Extract source IP from a Wazuh alert, checking common field locations."""
    data = alert.get("data", {})
    for field in ("srcip", "src_ip", "srcIp"):
        if data.get(field):
            return data[field]
    predecoder = alert.get("predecoder", {})
    if predecoder.get("srcip"):
        return predecoder["srcip"]
    return ""


def _wazuh_severity_label(level: int) -> str:
    """Map Wazuh rule level to severity label."""
    if level >= 14:
        return "critical"
    elif level >= 10:
        return "high"
    elif level >= 7:
        return "medium"
    return "low"


def _suricata_severity_label(severity: int) -> str:
    """Map Suricata alert.severity to severity label. 1=Critical, 2=High, 3=Medium."""
    if severity <= 1:
        return "critical"
    elif severity == 2:
        return "high"
    elif severity == 3:
        return "medium"
    return "low"


def _normalize_wazuh_alert(alert: dict, idx: int) -> dict:
    """Normalize a Wazuh alert into unified card format."""
    rule = alert.get("rule", {})
    agent = alert.get("agent", {})
    mitre = rule.get("mitre", {})
    data = alert.get("data", {})
    ts = alert.get("@timestamp") or alert.get("timestamp", "")
    full_log = alert.get("full_log", "")
    data_snippet = full_log[:500] if full_log else json.dumps(data, default=str)[:500]

    # Extract network flow fields from data object
    dest_ip = data.get("dstip", "") or data.get("dst_ip", "") or data.get("dstIp", "")
    src_port = str(data.get("srcport", "") or data.get("src_port", "") or "")
    dest_port = str(data.get("dstport", "") or data.get("dst_port", "") or "")
    protocol = data.get("protocol", "") or data.get("proto", "")
    action = data.get("action", "")

    # Decoder and log source metadata
    decoder_name = alert.get("decoder", {}).get("name", "")
    location = alert.get("location", "")
    rule_groups = rule.get("groups", [])
    firedtimes = rule.get("firedtimes", 0)

    return {
        "id": f"wz_{idx}_{rule.get('id', '')}",
        "source": "wazuh",
        "timestamp": ts,
        "severity": _wazuh_severity_label(rule.get("level", 0)),
        "severity_num": rule.get("level", 0),
        "rule_id": str(rule.get("id", "")),
        "rule_description": rule.get("description", ""),
        "agent_id": agent.get("id", ""),
        "agent_name": agent.get("name", ""),
        "source_ip": _extract_source_ip(alert),
        "dest_ip": dest_ip,
        "mitre_tactics": mitre.get("tactic", []),
        "mitre_techniques": mitre.get("technique", []),
        "category": ", ".join(rule_groups[:3]),
        "data_snippet": data_snippet,
        # Wazuh enrichment fields
        "src_port": src_port,
        "dest_port": dest_port,
        "protocol": protocol,
        "action": action,
        "decoder_name": decoder_name,
        "location": location,
        "rule_groups": rule_groups,
        "firedtimes": firedtimes,
    }


def _normalize_suricata_alert(alert: dict, idx: int) -> dict:
    """Normalize a Suricata alert into unified card format."""
    alert_data = alert.get("alert", {})
    ts = alert.get("@timestamp") or alert.get("timestamp", "")
    severity = alert_data.get("severity", 3)

    # Extract HTTP metadata (critical for identifying real attacker IPs behind proxies)
    http_data = alert.get("http", {})
    xff = http_data.get("xff", "") or http_data.get("x_forwarded_for", "")
    http_hostname = http_data.get("hostname", "")
    http_url = http_data.get("url", "")
    http_method = http_data.get("http_method", "") or http_data.get("method", "")
    http_user_agent = http_data.get("http_user_agent", "") or http_data.get("user_agent", "")
    http_status = http_data.get("status", "")

    # Also check top-level fields (some Suricata index mappings flatten these)
    if not xff:
        xff = alert.get("xff", "") or alert.get("x_forwarded_for", "")
    if not http_hostname:
        http_hostname = alert.get("hostname", "") or alert.get("http_hostname", "")

    return {
        "id": f"sr_{idx}_{alert_data.get('signature_id', '')}",
        "source": "suricata",
        "timestamp": ts,
        "severity": _suricata_severity_label(severity),
        "severity_num": severity,
        "rule_id": str(alert_data.get("signature_id", "")),
        "rule_description": alert_data.get("signature", ""),
        "agent_id": "",
        "agent_name": "",
        "source_ip": alert.get("src_ip", "") or alert.get("source", {}).get("ip", ""),
        "dest_ip": alert.get("dest_ip", "") or alert.get("destination", {}).get("ip", ""),
        "mitre_tactics": [],
        "mitre_techniques": [],
        "category": alert_data.get("category", ""),
        "data_snippet": "",
        # HTTP enrichment fields
        "http_xff": xff,
        "http_hostname": http_hostname,
        "http_url": http_url,
        "http_method": http_method,
        "http_user_agent": http_user_agent,
        "http_status": str(http_status) if http_status else "",
    }


@router.post("/api/narrative/generate")
async def generate_narrative(request: Request):
    """Fetch high-severity alerts from Wazuh + Suricata for the narrative browser."""
    import time
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        body = {}

    time_range = body.get("time_range", "24h")
    min_level = body.get("min_level", 12)
    max_alerts = body.get("max_alerts", 200)

    # ── Filter + pagination params ─────────────────────────────────────
    # All optional. Defaults preserve legacy "give me everything" behavior
    # except for pagination, where we now default to 25/page since the
    # frontend renders the list with proper pagination controls.
    severity_filter = (body.get("severity") or "all").strip().lower()
    source_filter   = (body.get("source")   or "all").strip().lower()
    status_filter   = (body.get("status")   or "all").strip().lower()
    # Frontend resolves "me" → actual Keycloak username before sending,
    # so this is just a literal-match filter on incident_store.assigned_to.
    assigned_filter = (body.get("assigned_to") or "").strip()
    search_query    = (body.get("search") or "").strip().lower()
    try:
        page = max(1, int(body.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        # Clamp to [10, 100] to bound server work
        page_size = min(100, max(10, int(body.get("page_size", 25))))
    except (TypeError, ValueError):
        page_size = 25

    unified_alerts = []
    wazuh_count = 0
    suricata_count = 0
    errors = []

    # ── Fetch Wazuh alerts ──────────────────────────────────────────────
    indexer = wazuh_client._indexer_client if wazuh_client else None
    if indexer:
        try:
            result = await indexer.search_alerts(
                time_range=time_range,
                size=max_alerts,
                min_level=min_level,
                sort=[{"@timestamp": "desc"}]
            )
            wazuh_alerts = result.get("data", {}).get("affected_items", [])
            wazuh_count = len(wazuh_alerts)
            for i, alert in enumerate(wazuh_alerts):
                unified_alerts.append(_normalize_wazuh_alert(alert, i))
        except Exception as e:
            logger.warning(f"Narrative: Wazuh alert fetch failed: {e}")
            errors.append(f"Wazuh: {str(e)}")
    else:
        errors.append("Wazuh Indexer not configured")

    # ── Fetch Suricata alerts (severity 1=Critical + 2=High only) ──────
    if suricata_client is not None:
        try:
            # Fetch critical (severity 1) alerts
            crit_result = await suricata_client.get_alerts(
                time_range=time_range,
                limit=max_alerts,
                severity="1",
            )
            crit_alerts = crit_result.get("data", {}).get("affected_items", [])

            # Fetch high (severity 2) alerts
            high_result = await suricata_client.get_alerts(
                time_range=time_range,
                limit=max_alerts,
                severity="2",
            )
            high_alerts = high_result.get("data", {}).get("affected_items", [])

            suricata_alerts = crit_alerts + high_alerts
            suricata_count = len(suricata_alerts)
            for i, alert in enumerate(suricata_alerts):
                unified_alerts.append(_normalize_suricata_alert(alert, i))
        except Exception as e:
            logger.warning(f"Narrative: Suricata alert fetch failed: {e}")
            errors.append(f"Suricata: {str(e)}")

    # ── Merge and sort by timestamp (most recent first) ─────────────────
    unified_alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

    # Truncate to max_alerts
    unified_alerts = unified_alerts[:max_alerts]

    # ── Annotate alerts with Agelia state (mute + block) ────────────────
    # Don't filter at the annotate step — the frontend renders badges and
    # offers a "Show muted" toggle. List view always reflects current Agelia
    # truth (within 60-second cache window). Per-action UI badges still come
    # from the pallas-incidents mirror columns.
    mute_rules = await _get_mute_summary()
    blocked_ips = await _get_blocked_ips_set()
    agelia_muted_count = 0
    agelia_blocked_count = 0

    # Batch-load incident state for the alerts we have so far so the status
    # and assigned_to filters can run without N+1 fetches.
    alert_ids = [a.get('id') for a in unified_alerts if a.get('id')]
    incident_states = {}
    if alert_ids:
        try:
            incident_states = await incident_store.get_incidents_batch(alert_ids) or {}
        except Exception as exc:
            logger.warning("Batch incident-state load failed (continuing without status filter data): %s", exc)
            incident_states = {}

    for alert in unified_alerts:
        # Attach incident state for downstream filter/UI use (frontend ignores
        # this if not needed; backend reads from alert['_incident'] in the filter).
        alert['_incident'] = incident_states.get(alert.get('id') or '', {}) or {}

        # Mute annotation
        matched_rule = _find_matching_mute_rule(alert, mute_rules) if mute_rules else None
        if matched_rule:
            alert['_agelia_muted'] = True
            alert['_agelia_mute_classification'] = matched_rule.get('classification')
            agelia_muted_count += 1
        else:
            alert['_agelia_muted'] = False
            alert['_agelia_mute_classification'] = None

        # Block annotation — check the source IP fields in priority order
        src_ip = (
            (alert.get('http_xff') or '').split(',')[0].strip()
            or alert.get('source_ip')
            or alert.get('src_ip')
            or ''
        )
        if src_ip and src_ip in blocked_ips:
            alert['_waf_blocked'] = True
            agelia_blocked_count += 1
        else:
            alert['_waf_blocked'] = False

    # ── Apply filters BEFORE pagination ────────────────────────────────
    # Each filter is opt-in: if the param defaults to "all" / empty, the
    # filter passes through. The combined predicate is AND across all dims.
    def _alert_matches_filters(alert):
        # Severity (normalized field — both Wazuh and Suricata go through normalize)
        if severity_filter != "all":
            if (alert.get("severity") or "").lower() != severity_filter:
                return False
        # Source
        if source_filter != "all":
            if (alert.get("source") or "").lower() != source_filter:
                return False
        # Incident lifecycle status
        if status_filter != "all":
            inc = alert.get("_incident") or {}
            current_status = (inc.get("status") or "open").lower()
            if current_status != status_filter:
                return False
        # "Mine" / assigned-to filter. The frontend resolves "me" → the analyst's
        # Keycloak username before sending. Match broadly: an analyst is
        # "involved" with an incident if they appear in ANY actor field
        # (acknowledged_by, closed_by, fp_dismissed_by, assigned_by). The
        # `assigned_to` field stores the Jira display name (e.g. "Shakeel
        # Zaman") rather than the Keycloak username (`shakeelzaman`), so we
        # also do a case-insensitive substring check there to catch the
        # common case where the display name contains the username.
        if assigned_filter:
            inc = alert.get("_incident") or {}
            af_lower = assigned_filter.lower()
            actor_fields = (
                inc.get("acknowledged_by"),
                inc.get("closed_by"),
                inc.get("fp_dismissed_by"),
                inc.get("assigned_by"),
            )
            assigned_to_value = (inc.get("assigned_to") or "")
            matches_actor = any(
                (v or "").lower() == af_lower for v in actor_fields
            )
            matches_assigned = (
                assigned_to_value
                and (af_lower in assigned_to_value.lower() or assigned_to_value.lower() == af_lower)
            )
            if not (matches_actor or matches_assigned):
                return False
        # Search across SID / IP / agent / description
        if search_query:
            haystack = " ".join([
                str(alert.get("rule_id") or ""),
                str(alert.get("signature_id") or ""),
                str(alert.get("source_ip") or alert.get("src_ip") or ""),
                str(alert.get("dest_ip") or ""),
                str(alert.get("agent_name") or ""),
                str(alert.get("rule_description") or alert.get("description") or alert.get("alert_msg") or ""),
            ]).lower()
            if search_query not in haystack:
                return False
        return True

    raw_total = len(unified_alerts)
    filtered_alerts = [a for a in unified_alerts if _alert_matches_filters(a)]
    total_after_filter = len(filtered_alerts)

    # ── Paginate ───────────────────────────────────────────────────────
    total_pages = max(1, (total_after_filter + page_size - 1) // page_size)
    # Clamp page so out-of-range requests return last page rather than empty
    page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    page_alerts = filtered_alerts[start:end]

    elapsed_ms = int((time.time() - start_time) * 1000)

    return JSONResponse(content={
        "alerts": page_alerts,
        "metadata": {
            # Post-filter total (the badge in the UI). Pre-filter total is
            # available as `raw_total` for "X of Y" display.
            "total_alerts":        total_after_filter,
            "raw_total":           raw_total,
            "wazuh_count":         wazuh_count,
            "suricata_count":      suricata_count,
            "agelia_muted_count":  agelia_muted_count,
            "agelia_blocked_count": agelia_blocked_count,
            "page":                page,
            "page_size":           page_size,
            "total_pages":         total_pages,
            "time_range":          time_range,
            "min_level":           min_level,
            "scan_duration_ms":    elapsed_ms,
            "errors":              errors if errors else None,
        }
    })


# ─── Narrative cache ──────────────────────────────────────────────────────────
# Pallas LLM calls are 30-120s and the same alert often gets clicked multiple
# times during investigation. Cache by alert id (LLM output is deterministic
# enough at temperature=0.3 to make this safe within an investigation window).
_narrative_cache: dict = {}    # alert_id -> (narrative_text, fetched_at_monotonic)
_NARRATIVE_TTL: float = 3600.0  # 1 hour


# ─── LLMClient singleton ──────────────────────────────────────────────────────
# Avoid creating a fresh httpx client + handshake on every narrative request.
# Reuses connection pooling to the LLM provider, saves ~100-300ms per call.
_llm_client_singleton = None


async def _get_llm_client():
    global _llm_client_singleton
    if _llm_client_singleton is None:
        client = LLMClient(
            base_url=os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus"),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        await client.initialize()
        _llm_client_singleton = client
    return _llm_client_singleton


@router.post("/api/narrative/enhance")
async def enhance_narrative(request: Request):
    """Generate AI narrative for a single alert. Long-running: 30-120s."""
    import time
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    alert_id = body.get("id", "")

    # Cache hit: return the cached narrative immediately (avoids the 30-120s
    # LLM call). Particularly valuable when an analyst re-clicks an alert
    # during investigation to re-read the narrative.
    if alert_id and alert_id in _narrative_cache:
        cached_text, fetched_at = _narrative_cache[alert_id]
        if time.monotonic() - fetched_at < _NARRATIVE_TTL:
            return JSONResponse(content={
                "alert_id": alert_id,
                "narrative": cached_text,
                "generation_time_ms": 0,
                "cached": True,
            })
    alert_source = body.get("source", "wazuh")
    timestamp = body.get("timestamp", "")
    severity = body.get("severity", "")
    severity_num = body.get("severity_num", 0)
    rule_id = body.get("rule_id", "")
    rule_description = body.get("rule_description", "")
    agent_name = body.get("agent_name", "")
    agent_id = body.get("agent_id", "")
    source_ip = body.get("source_ip", "")
    dest_ip = body.get("dest_ip", "")
    mitre_tactics = body.get("mitre_tactics", [])
    mitre_techniques = body.get("mitre_techniques", [])
    category = body.get("category", "")
    data_snippet = body.get("data_snippet", "")

    # Suricata HTTP enrichment fields
    http_xff = body.get("http_xff", "")
    http_hostname = body.get("http_hostname", "")
    http_url = body.get("http_url", "")
    http_method = body.get("http_method", "")
    http_user_agent = body.get("http_user_agent", "")
    http_status = body.get("http_status", "")

    system_prompt = (
        "You are a senior SOC analyst writing a detailed forensic incident narrative. "
        "Your audience is a Tier 2/3 SOC analyst who needs actionable context.\n\n"
        "Structure your narrative with these sections (use markdown headers):\n\n"
        "## Incident Summary\n"
        "A 2-3 sentence executive summary: what happened, when, and the severity assessment.\n\n"
        "## Technical Analysis\n"
        "Detailed breakdown of the alert:\n"
        "- What triggered the detection (signature/rule specifics)\n"
        "- Network flow analysis (source, destination, direction of traffic)\n"
        "- If X-Forwarded-For or proxy headers are present, identify the REAL attacker IP vs proxy IP\n"
        "- Protocol and application-layer context (HTTP method, URL, user-agent if available)\n"
        "- MITRE ATT&CK mapping and kill chain stage (if applicable)\n\n"
        "## Threat Assessment\n"
        "- What does this activity indicate? (scanning, exploitation, C2, data exfiltration, etc.)\n"
        "- Known attack patterns or TTPs this matches\n"
        "- Potential impact if this activity is malicious\n"
        "- Confidence level (HIGH/MEDIUM/LOW) with reasoning\n\n"
        "## Recommended Actions\n"
        "Numbered list of specific, actionable response steps for the SOC team.\n\n"
        "RULES:\n"
        "- Be specific: reference exact IPs, timestamps, SIDs, rule IDs — never generalize\n"
        "- Never invent data not provided in the alert\n"
        "- Use professional SOC terminology\n"
        "- If XFF (X-Forwarded-For) IP is present, explicitly call it out as the true origin IP\n"
    )

    if alert_source == "suricata":
        user_prompt = (
            f"Generate a detailed SOC forensic narrative for this Suricata IDS/IPS alert:\n\n"
            f"**Detection Details:**\n"
            f"- Timestamp: {timestamp}\n"
            f"- Signature: SID {rule_id} — {rule_description}\n"
            f"- Severity: {severity.upper()} (Level {severity_num}/4, where 1=Critical)\n"
            f"- Alert Category: {category}\n\n"
            f"**Network Flow:**\n"
            f"- Source IP: {source_ip}\n"
            f"- Destination IP: {dest_ip}\n"
        )
        if http_xff:
            user_prompt += (
                f"- X-Forwarded-For (Real Attacker IP): {http_xff}\n"
                f"  ⚠ The XFF header reveals the true origin IP behind the proxy/load balancer. "
                f"This is the ACTUAL attacker IP that should be investigated and blocked.\n"
            )
        if http_hostname:
            user_prompt += f"- Target Hostname: {http_hostname}\n"
        if http_method or http_url:
            user_prompt += f"\n**HTTP Context:**\n"
            if http_method:
                user_prompt += f"- Method: {http_method}\n"
            if http_url:
                user_prompt += f"- URL: {http_url}\n"
            if http_user_agent:
                user_prompt += f"- User-Agent: {http_user_agent}\n"
            if http_status:
                user_prompt += f"- Response Status: {http_status}\n"
    else:
        # Extract Wazuh enrichment fields
        wz_src_port = body.get("src_port", "")
        wz_dest_port = body.get("dest_port", "")
        wz_protocol = body.get("protocol", "")
        wz_action = body.get("action", "")
        wz_decoder_name = body.get("decoder_name", "")
        wz_location = body.get("location", "")
        wz_rule_groups = body.get("rule_groups", [])
        wz_firedtimes = body.get("firedtimes", 0)

        mitre_str = ""
        if mitre_tactics:
            mitre_str += f"- MITRE ATT&CK Tactics: {', '.join(mitre_tactics)}\n"
        if mitre_techniques:
            mitre_str += f"- MITRE ATT&CK Techniques: {', '.join(mitre_techniques)}\n"

        user_prompt = (
            f"Generate a detailed SOC forensic narrative for this Wazuh SIEM alert:\n\n"
            f"**Detection Details:**\n"
            f"- Timestamp: {timestamp}\n"
            f"- Rule: {rule_id} — {rule_description}\n"
            f"- Severity: {severity.upper()} (Level {severity_num}/15)\n"
            f"- Categories: {category}\n"
        )
        if wz_rule_groups:
            user_prompt += f"- Rule Groups: {', '.join(wz_rule_groups)}\n"
        if wz_decoder_name:
            user_prompt += f"- Decoder: {wz_decoder_name}\n"
        if wz_firedtimes and wz_firedtimes > 1:
            user_prompt += (
                f"- Rule Fire Count: {wz_firedtimes} times — "
                f"{'HIGH frequency, possible active attack or recurring misconfiguration' if wz_firedtimes > 100 else 'moderate frequency, investigate pattern'}\n"
            )
        if wz_location:
            user_prompt += f"- Log Source: {wz_location}\n"

        # Network flow section
        has_network = source_ip or dest_ip or wz_src_port or wz_dest_port
        if has_network:
            user_prompt += f"\n**Network Flow:**\n"
            src_str = source_ip or "N/A"
            if wz_src_port:
                src_str += f":{wz_src_port}"
            user_prompt += f"- Source: {src_str}\n"
            if dest_ip:
                dst_str = dest_ip
                if wz_dest_port:
                    dst_str += f":{wz_dest_port}"
                user_prompt += f"- Destination: {dst_str}\n"
            if wz_protocol:
                user_prompt += f"- Protocol: {wz_protocol}\n"
            if wz_action:
                user_prompt += f"- Action Taken: {wz_action}\n"

        # Agent context
        user_prompt += (
            f"\n**Source Context:**\n"
            f"- Agent: {agent_name} (ID: {agent_id})\n"
        )

        if mitre_str:
            user_prompt += f"\n**MITRE ATT&CK Mapping:**\n{mitre_str}"
        if data_snippet:
            user_prompt += f"\n**Raw Log (for deeper analysis):**\n```\n{data_snippet}\n```\n"

    try:
        # Cloud-only per directive (local LLM removed 2026-04-15).
        # WARNING: forensic narrative contains real IPs, agent names, MITRE data — now sent to cloud provider.
        # Reuses a process-singleton client so connection pool / TLS handshake
        # isn't paid per request (saves ~100-300ms).
        llm = await _get_llm_client()

        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
            temperature=0.3
        )

        narrative_text = response.strip()
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Cache the generated narrative for repeat clicks within the TTL window.
        if alert_id:
            _narrative_cache[alert_id] = (narrative_text, time.monotonic())

        return JSONResponse(content={
            "alert_id": alert_id,
            "narrative": narrative_text,
            "generation_time_ms": elapsed_ms,
            "cached": False,
        })

    except Exception as e:
        logger.error(f"Narrative generation failed: {e}", exc_info=True)
        elapsed_ms = int((time.time() - start_time) * 1000)
        return JSONResponse(content={
            "alert_id": alert_id,
            "narrative": "",
            "generation_time_ms": elapsed_ms,
            "error": str(e),
        }, status_code=500)


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private/RFC1918/loopback/link-local."""
    import ipaddress
    try:
        return ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False


@router.post("/api/narrative/enrich-observable")
async def enrich_observable(request: Request):
    """Enrich an observable (IP/domain/hash) via AbuseIPDB + Wazuh alert history for the narrative view."""
    import time
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    # Accept new {indicator, indicator_type} params with backward compat for {ip}
    indicator = body.get("indicator", "").strip()
    indicator_type = body.get("indicator_type", "ip").strip().lower()
    if not indicator:
        # Backward compat: fall back to "ip" field
        indicator = body.get("ip", "").strip()
        indicator_type = "ip"
    if not indicator:
        return JSONResponse(content={"error": "Missing 'indicator' (or 'ip') parameter"}, status_code=400)

    # Validate indicator_type
    if indicator_type not in ("ip", "domain", "hash"):
        return JSONResponse(content={"error": f"Invalid indicator_type '{indicator_type}'. Must be 'ip', 'domain', or 'hash'."}, status_code=400)

    # Reject private/RFC1918 IPs — enrichment is not applicable
    if indicator_type == "ip" and _is_private_ip(indicator):
        return JSONResponse(content={
            "error": "Private/RFC1918 IP — enrichment not applicable",
            "indicator": indicator,
            "is_private": True,
        }, status_code=400)

    try:
        # IP enrichment: route through Agelia (Redis-cached 24h, shared with All Events page)
        if indicator_type == "ip" and AGELIA_BASE_URL:
            result, status = await _call_agelia(
                "GET", f"/api/threat-intel/enrich-ip/{indicator}", request
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            if status in (200, 201):
                data = result.get("data", result)
                data["enrichment_time_ms"] = elapsed_ms
                data["indicator_type"] = indicator_type
                return JSONResponse(content=data)
            # Fall through to local enrichment on Agelia error
            logger.warning("Agelia enrichment returned %s for %s — falling back", status, indicator)

        # Domain / hash enrichment (or IP fallback if Agelia unavailable)
        result = await wazuh_client.check_ioc_reputation(indicator, indicator_type)
        elapsed_ms = int((time.time() - start_time) * 1000)
        result["enrichment_time_ms"] = elapsed_ms
        result["indicator_type"] = indicator_type
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Observable enrichment failed for {indicator} ({indicator_type}): {e}", exc_info=True)
        elapsed_ms = int((time.time() - start_time) * 1000)
        return JSONResponse(content={
            "indicator": indicator,
            "indicator_type": indicator_type,
            "error": str(e),
            "enrichment_time_ms": elapsed_ms,
        }, status_code=500)


# ============================================================================
# Pallas 4.3 - SOC Workflow Automation Endpoints
# ============================================================================


@router.get("/api/narrative/incident/{alert_id}")
async def get_incident(alert_id: str):
    """Get incident workflow state for an alert."""
    try:
        incident = await incident_store.get_incident(alert_id)
        if incident:
            incident.pop("_doc_id", None)
            return JSONResponse(content=incident)
        return JSONResponse(content=None)
    except Exception as e:
        logger.error(f"Get incident failed: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/narrative/incidents/batch")
async def get_incidents_batch(request: Request):
    """Get incident states for multiple alerts in one call."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_ids = body.get("alert_ids", [])
    if not alert_ids:
        return JSONResponse(content={})

    try:
        incidents = await incident_store.get_incidents_batch(alert_ids)
        # Remove internal _doc_id from results
        for k in incidents:
            incidents[k].pop("_doc_id", None)
        return JSONResponse(content=incidents)
    except Exception as e:
        logger.error(f"Batch get incidents failed: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/narrative/incident/acknowledge")
async def acknowledge_incident(request: Request):
    """Acknowledge an alert - creates incident record in OpenSearch."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id = body.get("alert_id", "").strip()
    alert = body.get("alert", {})
    analyst_name = body.get("analyst_name", "SOC Analyst")

    if not alert_id:
        return JSONResponse(content={"error": "Missing alert_id"}, status_code=400)

    try:
        incident = await incident_store.create_incident(alert_id, alert, analyst_name)
        incident.pop("_doc_id", None)

        return JSONResponse(content=incident)
    except Exception as e:
        logger.error(f"Acknowledge failed: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/narrative/incident/create-ticket")
async def create_jira_ticket(request: Request):
    """Create a Jira incident ticket for an acknowledged alert."""
    if not jira_client_instance.is_configured:
        return JSONResponse(
            content={"error": "Jira not configured. Set JIRA_BASE_URL + JIRA_SERVICE_TOKEN (or JIRA_USER_EMAIL + JIRA_API_TOKEN)."},
            status_code=503,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id = body.get("alert_id", "").strip()
    incident_id = body.get("incident_id", "")
    narrative = body.get("narrative", "")
    alert = body.get("alert", {})
    analyst_name = body.get("analyst_name", "SOC Analyst")

    if not alert_id:
        return JSONResponse(content={"error": "Missing alert_id"}, status_code=400)

    # Verify incident exists and is acknowledged
    existing = await incident_store.get_incident(alert_id)
    if not existing:
        return JSONResponse(content={"error": "Incident not found. Acknowledge the alert first."}, status_code=400)

    # Create Jira ticket
    result = await jira_client_instance.create_ticket(alert, narrative, incident_id)
    if "error" in result:
        return JSONResponse(content=result, status_code=502)

    # Update incident with Jira details
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    await incident_store.update_incident(
        alert_id,
        {
            "status": "in_progress",
            "jira_ticket_id": result.get("jira_ticket_id"),
            "jira_ticket_url": result.get("jira_ticket_url"),
            "jira_created_at": now,
        },
        action_log={
            "action": "jira_created",
            "timestamp": now,
            "actor": analyst_name,
            "details": f"Jira ticket created: {result.get('jira_ticket_id')} by {analyst_name}",
        },
    )

    return JSONResponse(content=result)


@router.post("/api/narrative/incident/waf-block")
async def waf_block_ip(request: Request):
    """Block an attacker IP via Agelia dual-layer blocking (WAF + Network Firewall).

    Formerly called AWS WAF directly via boto3. Now proxied through Agelia so that
    all block actions are audited in one place under the analyst's Keycloak identity.
    The 'waf-block' name is kept for frontend compatibility; Agelia applies both
    layers (Network Firewall + WAF) in a single call.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id     = body.get("alert_id", "").strip()
    ip           = body.get("ip", "").strip()
    analyst_name = body.get("analyst_name", "SOC Analyst")
    alert_source = body.get("alert_source", "pallas")
    user_reason  = (body.get("reason") or "").strip()
    threat_type  = (body.get("threat_type") or "other").strip()
    expires_hours = body.get("expires_in_hours")  # int or None — Permanent if missing

    if not alert_id or not ip:
        return JSONResponse(content={"error": "Missing alert_id or ip"}, status_code=400)

    # Forward the analyst-supplied reason verbatim; fall back to a default only
    # if the frontend sends an empty string. Agelia overrides `analyst` from the
    # validated Keycloak JWT, so analyst_name here is informational only.
    agelia_payload = {
        "ip":           ip,
        "alert_id":     alert_id,
        "alert_source": alert_source,
        "analyst":      analyst_name,
        "reason":       user_reason or f"Blocked from Pallas Attack Narrative by {analyst_name}",
        "threat_type":  threat_type,
    }
    if expires_hours:
        try:
            agelia_payload["expires_in_hours"] = int(expires_hours)
        except (TypeError, ValueError):
            pass  # ignore malformed value, leave as Permanent

    result, status = await _call_agelia("POST", "/api/firewall/block-from-alert", request, agelia_payload)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    if status not in (200, 201) or not result.get("success"):
        await incident_store.update_incident(
            alert_id,
            {"waf_error": result.get("error", "Agelia block failed")},
            action_log={
                "action": "waf_block_failed",
                "timestamp": now,
                "actor": analyst_name,
                "details": f"Block failed for {ip}: {result.get('error')} (by {analyst_name})",
            },
        )
        return JSONResponse(content=result, status_code=status or 502)

    # Invalidate the blocked-IPs cache so the next /alerts fetch reflects this
    # new block immediately instead of waiting for the 60-second TTL.
    _invalidate_blocked_ips_cache()

    data = result.get("data", result)
    existing = await incident_store.get_incident(alert_id)
    await incident_store.update_incident(
        alert_id,
        {
            "waf_blocked": True,
            "waf_blocked_at": now,
            "waf_blocked_ip": ip,
            "waf_error": data.get("waf_error"),
            "firewall_blocked": True,
            "firewall_blocked_at": now,
            "firewall_blocked_ip": ip,
        },
        action_log={
            "action": "block_ip",
            "timestamp": now,
            "actor": analyst_name,
            "details": f"Blocked {ip} via Agelia (layers: {data.get('layers_blocked', [])}) by {analyst_name}",
        },
    )

    if existing and existing.get("jira_ticket_id") and jira_client_instance.is_configured:
        layers = ", ".join(data.get("layers_blocked", ["network_firewall", "waf"]))
        await jira_client_instance.add_comment(
            existing["jira_ticket_id"],
            f"IP block executed by {analyst_name}: {ip} blocked via Agelia ({layers})",
        )

    return JSONResponse(content=data)


@router.post("/api/narrative/incident/firewall-block")
async def firewall_block_ip(request: Request):
    """Block an attacker IP via Agelia dual-layer blocking (Network Firewall + WAF).

    Kept as a separate endpoint for frontend compatibility. Internally calls the
    same Agelia block-from-alert endpoint as waf-block — both layers are applied
    in one call. Direct boto3/AWS SDK calls have been removed.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    # Delegate entirely to waf_block_ip — same Agelia call, same result
    return await waf_block_ip(request)


@router.post("/api/narrative/incident/mute-agelia")
async def mute_in_agelia(request: Request):
    """Mute an alert in Agelia with specific conditions and classification.

    Calls Agelia's /api/mute/rules directly with the analyst's Keycloak JWT,
    replacing the former Integration Bridge (X-Integration-Key) call.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    conditions     = body.get('conditions', {})
    # 'suppressed' is not a valid Agelia classification — use 'known_benign' as default
    classification = body.get('classification', 'known_benign')
    valid_cls      = {'false_positive', 'known_benign', 'tuning'}
    if classification not in valid_cls:
        classification = 'known_benign'
    analyst_name   = body.get('analyst_name', 'SOC Analyst')
    alert_id       = body.get('alert_id', '')
    reason         = body.get('reason', '') or f'Muted from Pallas by {analyst_name}'

    if not conditions:
        return JSONResponse(content={"error": "conditions dict is required"}, status_code=400)

    # Detect source_type from conditions keys
    if 'signature_id' in conditions:
        source_type = 'suricata'
    elif 'rule_id' in conditions or 'agent_name' in conditions:
        source_type = 'wazuh'
    else:
        source_type = 'both'

    result, status = await _call_agelia("POST", "/api/mute/rules", request, {
        "conditions":        conditions,
        "classification":    classification,
        "reason":            reason,
        "source_type":       source_type,
        "original_alert_id": alert_id,
    })

    if status not in (200, 201):
        return JSONResponse(content=result, status_code=status or 500)

    # Invalidate the mute summary cache so the next /alerts fetch reflects this
    # new mute immediately instead of waiting for the 60-second TTL.
    _invalidate_mute_summary_cache()

    # Side-effect: when the analyst's verdict is False Positive, also flip the
    # incident's lifecycle status. Replaces the legacy `dismiss-fp` shortcut —
    # the Mute modal defaults to false_positive so this is the path the
    # frontend uses for FP dismissals now. Other classifications (known_benign,
    # tuning) leave incident status alone — the analyst is suppressing noise,
    # not declaring the incident resolved.
    existing = await incident_store.get_incident(alert_id) if alert_id else None
    if classification == 'false_positive' and existing:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        try:
            await incident_store.update_incident(
                alert_id,
                {
                    "status": "false_positive",
                    "fp_reason": reason,
                    "fp_dismissed_by": analyst_name,
                    "fp_dismissed_at": now,
                    "agelia_muted": True,
                    "agelia_muted_conditions": conditions,
                },
                action_log={
                    "action": "false_positive",
                    "timestamp": now,
                    "actor": analyst_name,
                    "details": f"Dismissed as false positive by {analyst_name} via Mute modal: {reason}",
                },
            )
        except Exception as exc:
            logger.warning("Incident status update for FP mute failed (non-fatal): %s", exc)

    # Mirror the action onto the Jira ticket if one exists for this alert,
    # for parity with the Block flow. Failures here are non-fatal — they don't
    # roll back the mute, just leave the ticket without a comment.
    try:
        if existing and existing.get("jira_ticket_id") and jira_client_instance.is_configured:
            cond_summary = ", ".join(f"{k}={v}" for k, v in conditions.items())
            await jira_client_instance.add_comment(
                existing["jira_ticket_id"],
                f"Alert muted by {analyst_name}: classification={classification}, conditions=[{cond_summary}]",
            )
    except Exception as exc:
        logger.warning("Jira comment for mute action failed (non-fatal): %s", exc)

    return JSONResponse(content={
        "success": True,
        "conditions": conditions,
        "classification": classification,
        "mute_rule_id": result.get("data", {}).get("id"),
        "message": "Alert muted in Agelia",
    })


@router.post("/api/narrative/incident/close")
async def close_incident(request: Request):
    """Close an incident - update Jira, calculate MTTR."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id = body.get("alert_id", "").strip()
    analyst_name = body.get("analyst_name", "SOC Analyst")
    resolution_notes = body.get("resolution_notes", "Incident resolved.")

    if not alert_id:
        return JSONResponse(content={"error": "Missing alert_id"}, status_code=400)

    existing = await incident_store.get_incident(alert_id)
    if not existing:
        return JSONResponse(content={"error": "Incident not found."}, status_code=400)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Calculate MTTR
    mttr_seconds = None
    alert_ts = existing.get("alert_timestamp", "")
    if alert_ts:
        try:
            if alert_ts.endswith("Z"):
                alert_dt = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
            else:
                alert_dt = datetime.fromisoformat(alert_ts)
            if alert_dt.tzinfo is None:
                alert_dt = alert_dt.replace(tzinfo=timezone.utc)
            mttr_seconds = (now - alert_dt).total_seconds()
        except Exception:
            pass

    # Build resolution summary for Jira
    actions_taken = []
    if existing.get("waf_blocked"):
        actions_taken.append(f"WAF: Blocked IP {existing.get('waf_blocked_ip', 'N/A')}")
    if existing.get("firewall_blocked"):
        actions_taken.append(f"Network Firewall: Blocked IP {existing.get('firewall_blocked_ip', 'N/A')}")
    if existing.get("observability_verdict"):
        actions_taken.append(f"Observability: Verdict = {existing.get('observability_verdict')}")

    resolution_summary = (
        f"h2. Incident Closed\n\n"
        f"*Closed by:* {analyst_name}\n"
        f"*Resolution:* {resolution_notes}\n\n"
    )
    if mttr_seconds is not None:
        mttr_min = int(mttr_seconds / 60)
        resolution_summary += f"*MTTR:* {mttr_min} minutes\n\n"
    if actions_taken:
        resolution_summary += "*Actions Taken:*\n"
        for a in actions_taken:
            resolution_summary += f"* {a}\n"
    resolution_summary += "\n----\n_Closed via Pallas AI Security Platform_"

    # Close Jira ticket if exists
    jira_closed = False
    jira_ticket_id = existing.get("jira_ticket_id")
    if jira_ticket_id and jira_client_instance.is_configured:
        close_result = await jira_client_instance.close_ticket(jira_ticket_id, resolution_summary)
        jira_closed = close_result.get("closed", False)

    # Update incident
    await incident_store.update_incident(
        alert_id,
        {
            "status": "closed",
            "closed_at": now_iso,
            "closed_by": analyst_name,
            "resolution_notes": resolution_notes,
            "mttr_seconds": mttr_seconds,
        },
        action_log={
            "action": "closed",
            "timestamp": now_iso,
            "actor": analyst_name,
            "details": resolution_notes,
        },
    )

    return JSONResponse(content={
        "incident_id": existing.get("incident_id"),
        "status": "closed",
        "closed_at": now_iso,
        "mttr_seconds": mttr_seconds,
        "mttr_minutes": int(mttr_seconds / 60) if mttr_seconds else None,
        "jira_closed": jira_closed,
    })


@router.post("/api/narrative/incident/integration-status")
async def get_integration_status(request: Request):
    """Return which integrations are configured (for frontend button visibility)."""
    return JSONResponse(content={
        "jira": jira_client_instance.is_configured,
        "agelia_blocking": agelia_blocking_configured,
        "incident_store": True,  # Always available (uses existing OpenSearch)
    })


@router.post("/api/narrative/incident/dismiss-fp")
async def dismiss_false_positive(request: Request):
    """DEPRECATED: Use /api/narrative/incident/mute-agelia with classification='false_positive'.

    This endpoint is kept as a thin compatibility wrapper for any external
    callers (older Pallas frontends, scripts, integrations) that still post
    here. It auto-derives the mute conditions from the incident's stored
    alert metadata and delegates to the same code path as the Mute modal,
    so behavior is identical: mute rule created in Agelia + incident status
    flipped to false_positive + Jira comment added.

    Plan: remove in next minor release once logs confirm zero callers.
    """
    logger.warning(
        "DEPRECATED endpoint called: /api/narrative/incident/dismiss-fp. "
        "Callers should switch to /api/narrative/incident/mute-agelia with "
        "classification='false_positive'."
    )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id = (body.get("alert_id") or "").strip()
    reason = (body.get("reason") or "").strip()
    analyst_name = body.get("analyst_name", "SOC Analyst")

    if not alert_id:
        return JSONResponse(content={"error": "Missing alert_id"}, status_code=400)

    existing = await incident_store.get_incident(alert_id)
    if not existing:
        return JSONResponse(content={"error": "Incident not found. Acknowledge the alert first."}, status_code=400)

    # Auto-derive conditions from the alert metadata stored on the incident,
    # the way the original endpoint did. Mute modal callers pass conditions
    # explicitly; this endpoint is for callers that don't.
    rule_id = existing.get("rule_id", "")
    sig_id  = existing.get("signature_id", "")
    if rule_id:
        conditions  = {"rule_id": rule_id}
        source_type = "wazuh"
    elif sig_id:
        conditions  = {"signature_id": sig_id}
        source_type = "suricata"
    else:
        conditions  = {}
        source_type = "both"

    if not conditions:
        return JSONResponse(content={
            "error": "Cannot derive mute conditions from incident — please use the Mute modal directly.",
        }, status_code=400)

    # Delegate to Agelia (same call mute-agelia makes), then run the same
    # FP-status side-effect + Jira comment as the consolidated mute handler.
    mute_result, mute_status = await _call_agelia("POST", "/api/mute/rules", request, {
        "conditions":        conditions,
        "classification":    "false_positive",
        "reason":            reason or f"False positive dismissed from Pallas by {analyst_name}",
        "source_type":       source_type,
        "original_alert_id": alert_id,
    })

    if mute_status not in (200, 201):
        logger.warning("Agelia mute rule creation failed for FP dismiss: %s", mute_result)
        return JSONResponse(content=mute_result, status_code=mute_status or 500)

    _invalidate_mute_summary_cache()

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    await incident_store.update_incident(
        alert_id,
        {
            "status": "false_positive",
            "fp_reason": reason or "No reason provided",
            "fp_dismissed_by": analyst_name,
            "fp_dismissed_at": now,
            "agelia_muted": True,
            "agelia_muted_conditions": conditions,
        },
        action_log={
            "action": "false_positive",
            "timestamp": now,
            "actor": analyst_name,
            "details": f"Dismissed as false positive by {analyst_name} (deprecated endpoint): {reason}" if reason else f"Dismissed as false positive by {analyst_name} (deprecated endpoint)",
        },
    )

    try:
        if existing.get("jira_ticket_id") and jira_client_instance.is_configured:
            reason_text = reason or "No reason provided"
            await jira_client_instance.add_comment(
                existing["jira_ticket_id"],
                f"Alert dismissed as false positive by {analyst_name}: {reason_text}",
            )
    except Exception as exc:
        logger.warning("Jira comment for FP dismiss failed (non-fatal): %s", exc)

    return JSONResponse(content={"dismissed": True, "status": "false_positive", "reason": reason})


@router.post("/api/narrative/incident/assignable-users")
async def get_assignable_users(request: Request):
    """Fetch Jira users assignable to tickets in the SECOPS project."""
    if not jira_client_instance.is_configured:
        return JSONResponse(
            content={"error": "Jira not configured."},
            status_code=503,
        )
    users = await jira_client_instance.get_assignable_users()
    return JSONResponse(content={"users": users})


@router.post("/api/narrative/incident/assign-ticket")
async def assign_ticket(request: Request):
    """Assign a Jira ticket to a team member."""
    if not jira_client_instance.is_configured:
        return JSONResponse(content={"error": "Jira not configured."}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id     = body.get("alert_id", "").strip()
    account_id   = (body.get("account_id") or body.get("assignee_id") or "").strip()
    display_name = body.get("display_name") or body.get("assignee_name") or ""
    analyst_name = body.get("analyst_name", "SOC Analyst")

    if not alert_id or not account_id:
        return JSONResponse(content={"error": "Missing alert_id or account_id"}, status_code=400)

    # Get incident to find ticket ID
    existing = await incident_store.get_incident(alert_id)
    if not existing or not existing.get("jira_ticket_id"):
        return JSONResponse(content={"error": "No Jira ticket found. Create a ticket first."}, status_code=400)

    # Assign via Jira API
    result = await jira_client_instance.assign_ticket(existing["jira_ticket_id"], account_id)
    if "error" in result:
        return JSONResponse(content=result, status_code=502)

    # Update incident store
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    await incident_store.update_incident(
        alert_id,
        {
            "assigned_to": display_name,
            "assigned_at": now,
            "assigned_by": analyst_name,
        },
        action_log={
            "action": "assigned",
            "timestamp": now,
            "actor": analyst_name,
            "details": f"Ticket {existing['jira_ticket_id']} assigned to {display_name} by {analyst_name}",
        },
    )

    # Add Jira comment for audit trail
    await jira_client_instance.add_comment(
        existing["jira_ticket_id"],
        f"Ticket assigned to {display_name} by {analyst_name}",
    )

    return JSONResponse(content={"assigned": True, "assigned_to": display_name})


@router.post("/api/narrative/incident/update-observability")
async def update_observability(request: Request):
    """Update incident observability state after IP enrichment."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    alert_id = body.get("alert_id", "")
    verdict = body.get("verdict", "")
    if not alert_id or not verdict:
        return JSONResponse(content={"error": "alert_id and verdict required"}, status_code=400)

    try:
        updates = {
            "observability_completed": True,
            "observability_verdict": verdict,
            "observability_completed_at": datetime.now(timezone.utc).isoformat(),
        }
        action_log = {
            "action": "observability_completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": "system",
            "details": f"Verdict: {verdict}",
        }
        await incident_store.update_incident(alert_id, updates, action_log)
        incident = await incident_store.get_incident(alert_id)
        return JSONResponse(content=incident or {"success": True})
    except Exception as e:
        logger.error(f"Update observability failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/api/decoder/explain")
async def explain_decoder(request: Request):
    """Generate AI explanation and deployment guide for a Wazuh decoder XML."""
    import time
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    decoder_xml = body.get("decoder_xml", "")
    if not decoder_xml.strip():
        return JSONResponse(content={"error": "decoder_xml is required"}, status_code=400)

    system_prompt = (
        "You are a senior Wazuh engineer and SOC architect. "
        "Explain the provided Wazuh decoder in clear, actionable, comprehensive detail. "
        "Use markdown formatting with headers, tables, and code blocks."
    )

    user_prompt = (
        f"Analyze and explain this Wazuh decoder in full detail:\n\n"
        f"```xml\n{decoder_xml}\n```\n\n"
        f"Structure your response with these sections:\n\n"
        f"## 1. Decoder Overview\n"
        f"- **What this decoder is** — its name, purpose, and what log source it targets\n"
        f"- **Decoder type** — whether it is a parent decoder, child decoder, or standalone; "
        f"explain the decoder hierarchy (parent/child chain)\n"
        f"- **Log format** — what type of logs this decoder parses (syslog, JSON, CEF, custom, etc.)\n\n"
        f"## 2. Fields & Components\n"
        f"- List every field extracted by the `<order>` tag in a table: "
        f"Field Name | Description | Example Value\n"
        f"- Explain each XML element used: `<parent>`, `<prematch>`, `<regex>`, `<order>`, `<program_name>`, etc.\n"
        f"- Break down the regex patterns — explain what each capture group `(\\S+)`, `(\\d+)`, etc. matches\n\n"
        f"## 3. How to Add This Decoder to Wazuh\n"
        f"Provide step-by-step deployment instructions:\n"
        f"- Exact file path where to save it (e.g., `/var/ossec/etc/decoders/local_decoder.xml`)\n"
        f"- How to add it without breaking existing decoders\n"
        f"- How to test it with `wazuh-logtest` (show the exact command)\n"
        f"- How to restart the Wazuh Manager to apply changes\n"
        f"- How to verify it is working after deployment\n\n"
        f"## 4. Recommendations\n"
        f"- Suggestions for improving the decoder (additional fields, better regex, sibling decoders)\n"
        f"- Related rules that could be written to alert on events parsed by this decoder\n"
    )

    try:
        llm = LLMClient(
            base_url=os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus"),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        await llm.initialize()

        try:
            response = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.3
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return JSONResponse(content={
                "explanation": response.strip(),
                "generation_time_ms": elapsed_ms,
            })
        finally:
            await llm.close()

    except Exception as e:
        logger.error(f"Decoder explain failed: {e}", exc_info=True)
        elapsed_ms = int((time.time() - start_time) * 1000)
        return JSONResponse(content={
            "explanation": "",
            "generation_time_ms": elapsed_ms,
            "error": str(e),
        }, status_code=500)


@router.post("/api/ip/investigate")
async def investigate_ip(request: Request):
    """Search an IP across Wazuh alerts, Wazuh index (vulnerabilities), and Suricata — return counts."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    ip = body.get("ip", "").strip()
    time_range = body.get("time_range", "24h")

    if not ip:
        return JSONResponse(content={"error": "ip is required"}, status_code=400)

    import time as _time
    start = _time.time()

    results = {
        "ip": ip,
        "time_range": time_range,
        "wazuh_alerts": {"count": 0, "error": None},
        "wazuh_vulnerabilities": {"count": 0, "error": None},
        "suricata_alerts": {"count": 0, "error": None},
        "total_count": 0,
    }

    indexer = wazuh_client._indexer_client if wazuh_client else None

    # --- Wazuh alerts (wazuh-alerts-*) ---
    if indexer:
        try:
            wazuh_alert_query = {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
                        {"multi_match": {
                            "query": ip,
                            "fields": ["data.srcip", "data.dstip", "agent.ip", "data.src_ip", "data.dst_ip", "data.win.eventdata.sourceAddress", "data.win.eventdata.destinationAddress", "data.http_xff", "data.xff"],
                            "type": "phrase"
                        }}
                    ]
                }
            }
            url = f"{indexer.base_url}/wazuh-alerts-*/_search"
            resp = await indexer.client.post(url, json={"query": wazuh_alert_query, "size": 0, "track_total_hits": True}, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            results["wazuh_alerts"]["count"] = data.get("hits", {}).get("total", {}).get("value", 0)
        except Exception as e:
            logger.warning(f"IP investigate - Wazuh alerts error: {e}")
            results["wazuh_alerts"]["error"] = str(e)

        # --- Wazuh vulnerabilities (wazuh-states-vulnerabilities-*) ---
        try:
            vuln_query = {
                "bool": {
                    "must": [
                        {"multi_match": {
                            "query": ip,
                            "fields": ["agent.ip", "host.ip"],
                            "type": "phrase"
                        }}
                    ]
                }
            }
            url = f"{indexer.base_url}/wazuh-states-vulnerabilities-*/_search"
            resp = await indexer.client.post(url, json={"query": vuln_query, "size": 0, "track_total_hits": True}, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            results["wazuh_vulnerabilities"]["count"] = data.get("hits", {}).get("total", {}).get("value", 0)
        except Exception as e:
            logger.warning(f"IP investigate - Wazuh vulns error: {e}")
            results["wazuh_vulnerabilities"]["error"] = str(e)

    # --- Suricata alerts (suricata-1.1.0-*) ---
    if suricata_client:
        try:
            suricata_query = {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
                        {"bool": {"should": [
                            {"term": {"src_ip": ip}},
                            {"term": {"dest_ip": ip}},
                            {"term": {"http.xff": ip}},
                            {"term": {"http.xff.keyword": ip}},
                        ], "minimum_should_match": 1}}
                    ]
                }
            }
            resp = await suricata_client._request(
                "POST",
                f"/{suricata_client.index_pattern}/_search",
                json={"query": suricata_query, "size": 0, "track_total_hits": True}
            )
            results["suricata_alerts"]["count"] = resp.get("hits", {}).get("total", {}).get("value", 0)
        except Exception as e:
            logger.warning(f"IP investigate - Suricata error: {e}")
            results["suricata_alerts"]["error"] = str(e)

    results["total_count"] = (
        results["wazuh_alerts"]["count"] +
        results["wazuh_vulnerabilities"]["count"] +
        results["suricata_alerts"]["count"]
    )
    results["duration_ms"] = int((_time.time() - start) * 1000)

    return JSONResponse(content=results)


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.post("/auth/token")
async def get_auth_token(request: Request):
    """Get JWT token using API key."""
    try:
        body = await request.json()
        api_key = body.get("api_key")

        if not api_key:
            raise HTTPException(status_code=400, detail="API key required")

        if not api_key.startswith("wazuh_"):
            raise HTTPException(status_code=401, detail="Invalid API key")

        token = create_access_token(
            data={
                "sub": "wazuh_mcp_user",
                "iat": datetime.now(timezone.utc).timestamp(),
                "scope": "wazuh:read wazuh:write"
            },
            secret_key=config.AUTH_SECRET_KEY
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 86400
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Include router with /ai-analyst prefix in the main app
app.include_router(router)


# ============================================================================
# STARTUP EVENT - Initialize Hybrid Query Orchestrator
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup with hybrid query orchestrator."""
    global _oauth_manager, _query_orchestrator

    logger.info(" Wazuh MCP Server v4.0.3 starting up...")
    logger.info(f"MCP Protocol: {MCP_PROTOCOL_VERSION}")
    logger.info(f"Wazuh Host: {config.WAZUH_HOST}")
    logger.info(f" CORS Origins: {config.ALLOWED_ORIGINS}")
    logger.info(f"Auth Mode: {config.AUTH_MODE}")
    logger.info(f" API Prefix: {API_PREFIX}")

    # Initialize OAuth if enabled
    if config.is_oauth:
        try:
            from wazuh_mcp_server.oauth import init_oauth_manager, create_oauth_router
            _oauth_manager = init_oauth_manager(config)
            oauth_router = create_oauth_router(_oauth_manager)
            router.include_router(oauth_router)
            logger.info(" OAuth 2.0 with DCR initialized")
        except Exception as e:
            logger.error(f" OAuth initialization failed: {e}")

    if config.is_authless:
        logger.warning(" Running in AUTHLESS mode - no authentication required!")
    elif config.is_bearer:
        logger.info("Bearer token authentication enabled")

    # Initialize Wazuh client
    try:
        await wazuh_client.initialize()
        logger.info(" Wazuh client initialized successfully")

        async def cleanup_wazuh():
            if hasattr(wazuh_client, 'close'):
                await wazuh_client.close()
                logger.info("Wazuh client connections closed")

        shutdown_manager.add_cleanup_task(cleanup_wazuh)

    except Exception as e:
        logger.warning(f" Wazuh client initialization failed: {e}")

    # Initialize Suricata client
    if suricata_client:
        try:
            await suricata_client.initialize()
            logger.info(" Suricata client initialized successfully")

            async def cleanup_suricata():
                if hasattr(suricata_client, 'close'):
                    await suricata_client.close()
                    logger.info("Suricata client connections closed")

            shutdown_manager.add_cleanup_task(cleanup_suricata)

            health = await suricata_client.health_check()
            if health.get("status") in ["green", "yellow"]:
                logger.info(f" Suricata Elasticsearch health: {health.get('status')}")
            else:
                logger.warning(f" Suricata Elasticsearch health: {health.get('status')}")

        except Exception as e:
            logger.warning(f" Suricata client initialization failed: {e}")

    # Test Wazuh connectivity
    try:
        await wazuh_client.get_manager_info()
        logger.info(" Wazuh connectivity test passed")
    except Exception as e:
        logger.warning(f" Wazuh connectivity test failed: {e}")

    # ========================================================================
    # Initialize Hybrid Query Orchestrator
    # ========================================================================
    try:
        _query_orchestrator = await create_orchestrator(
            execute_mcp_tool,
            suricata_enabled=(suricata_client is not None)
        )
        logger.info(" Hybrid Query Orchestrator initialized")
        logger.info(" Strategy: Rules first (fast), LLM fallback (flexible)")
        logger.info("WebSocket chat available at: ws://localhost:3000/ai-analyst/ws/chat")
    except Exception as e:
        logger.error(f" Query Orchestrator initialization failed: {e}", exc_info=True)
        logger.error("   Chat functionality will not be available")

        logger.info(" Server startup complete with hybrid intelligence enabled")


# ============================================================================
# SHUTDOWN EVENT
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown with graceful resource management."""
    global _query_orchestrator

    logger.info("Wazuh MCP Server initiating graceful shutdown...")

    try:
        await shutdown_manager.initiate_shutdown()

        # Cleanup query orchestrator
        if _query_orchestrator:
            # Close LLM client
            if hasattr(_query_orchestrator, 'llm'):
                try:
                    await _query_orchestrator.llm.close()
                    logger.info("LLM client closed")
                except Exception as e:
                    logger.error(f"Error closing LLM client: {e}")

            _query_orchestrator = None
            logger.info("Query orchestrator closed")

        # Clear and cleanup auth manager
        from wazuh_mcp_server.auth import auth_manager
        auth_manager.cleanup_expired()
        auth_manager.tokens.clear()
        logger.info("Authentication tokens cleared")

        # Clear sessions
        await sessions.clear()
        logger.info("Sessions cleared")

        # Cleanup Suricata client
        if suricata_client:
            try:
                await suricata_client.close()
                logger.info("Suricata client closed")
            except Exception as e:
                logger.error(f"Error closing Suricata client: {e}")

        # Cleanup rate limiter
        if hasattr(rate_limiter, 'cleanup'):
            rate_limiter.cleanup()

        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Garbage collection completed")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        logger.info(" Graceful shutdown completed")


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    uvicorn.run(
        app,
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )