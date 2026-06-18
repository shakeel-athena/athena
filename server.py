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
import hmac
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
from wazuh_mcp_server.api.aws_waf_client import AWSWAFClient
from wazuh_mcp_server.api.aws_firewall_client import AWSFirewallClient
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

# ============================================================================
# E3.0 (2026-06-12): shared route state foundation.
# Routes that will move into ./routes/<family>.py reach the singletons
# below via accessor functions in routes/_shared.py instead of importing
# from this module. Setters are called whenever the singleton is created
# or upgraded so the route-side getters always see the live instance.
# ============================================================================
from . import routes as _routes_pkg  # ensures routes package is loaded
from .routes import _shared as route_state
route_state.set_memory(_conversation_memory)  # initial set; upgraded in startup_event


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
route_state.set_session_store(_session_store)  # E3.0
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

# === PALLAS_DOCS_DISABLE === DO NOT REMOVE
# In production, hide auto-generated docs (they leak the full API surface)
if os.getenv("ENVIRONMENT", "production").lower() == "production":
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = "/ai-analyst/openapi.json"  # keep allowlisted, but undiscoverable from /docs
# === END PALLAS_DOCS_DISABLE ===

# === PALLAS_AUTH_MIDDLEWARE === DO NOT REMOVE
# Added 20260511T224912Z â€” enforces MCP_API_KEY for every HTTP request.
# Safe fallback: if MCP_API_KEY env is unset, the check is a no-op.
import hmac as _pallas_hmac
from fastapi import Request as _PallasRequest
from fastapi.responses import JSONResponse as _PallasJSONResponse

_PALLAS_PUBLIC_PATHS = {
    "/health",
    "/ai-analyst/health",
    "/ai-analyst/openapi.json",
}

@app.middleware("http")
async def pallas_enforce_api_key(request: _PallasRequest, call_next):
    # Skip CORS preflight
    if request.method == "OPTIONS":
        return await call_next(request)
    # Allowlist public endpoints
    if request.url.path in _PALLAS_PUBLIC_PATHS:
        return await call_next(request)
    # PALLAS_P3_MCP_AUTH_REQUIRED: bypass removed. /mcp/* now goes through
    # the same MCP_API_KEY check as the rest of the app. External MCP clients
    # (Claude Desktop, Cursor) must pass Authorization: Bearer <key>, x-api-key,
    # or ?api_key=<key>.
    # Allowlist docs in non-production
    if os.getenv("ENVIRONMENT", "production").lower() != "production":
        if request.url.path in ("/ai-analyst/docs", "/docs", "/redoc", "/docs/oauth2-redirect"):
            return await call_next(request)

    expected = (os.getenv("MCP_API_KEY") or "").strip()
    if not expected:
        # No key configured â€” middleware disabled (safe fallback during rollout)
        return await call_next(request)

    # Accept the key via Authorization: Bearer, x-api-key header, or ?api_key= query
    provided = ""
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        provided = auth_header[7:].strip()
    if not provided:
        provided = (request.headers.get("x-api-key") or "").strip()
    if not provided:
        provided = (request.query_params.get("api_key") or "").strip()

    if not provided or not _pallas_hmac.compare_digest(provided, expected):
        return _PallasJSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)
# === END PALLAS_AUTH_MIDDLEWARE ===

# === PALLAS_MCP_MOUNT === Pallas 2.3 Enhancement Week 1 - additive
# Mounts the FastMCP catalogue at /mcp/sse. If anything fails at import or
# mount time, log and continue so the existing app keeps working.
try:
    from wazuh_mcp_server.mcp_adapter import get_mcp_asgi_app, REGISTERED_COUNT as _PALLAS_MCP_COUNT
    app.mount("/mcp", get_mcp_asgi_app())
    logging.getLogger(__name__).info(
        f"[STARTUP] Mounted MCP transport at /mcp/sse - "
        f"{_PALLAS_MCP_COUNT} tools exposed via FastMCP"
    )
except Exception as _pallas_mcp_mount_exc:
    logging.getLogger(__name__).error(
        f"[STARTUP] Failed to mount MCP transport: {_pallas_mcp_mount_exc}"
    )
# === END PALLAS_MCP_MOUNT ===


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
# Blocking has two paths:
#   1) Agelia (single control plane, JWT-audited) — used when AGELEIA_BASE_URL is set
#   2) Direct AWS WAF + Network Firewall — fallback when Agelia isn't deployed
#      (this release). The direct clients each self-detect their env-var
#      configuration; if neither path is configured the Block button stays
#      hidden on the frontend.
agelia_blocking_configured: bool = bool(os.getenv("AGELEIA_BASE_URL", "").rstrip("/"))
aws_waf_client = AWSWAFClient()
aws_firewall_client = AWSFirewallClient()
direct_block_configured: bool = aws_waf_client.is_configured or aws_firewall_client.is_configured
logger.info(
    f"  SOC Workflow: Jira={'ON' if jira_client_instance.is_configured else 'OFF'}, "
    f"AgeliaBlocking={'ON' if agelia_blocking_configured else 'OFF'}, "
    f"DirectWAF={'ON' if aws_waf_client.is_configured else 'OFF'}, "
    f"DirectFirewall={'ON' if aws_firewall_client.is_configured else 'OFF'}"
)

# ─── Agelia Integration ───────────────────────────────────────────────────────
# AGELEIA_BASE_URL is the primary integration point for JWT-forwarding proxy calls
# (analyst response actions: block, mute, enrich).
# AGELEIA_API_URL + INTEGRATION_API_KEY are used only for the read-only
# machine-to-machine mute summary endpoint consumed by alert display filtering.
AGELEIA_BASE_URL       = os.getenv('AGELEIA_BASE_URL', '').rstrip('/')
AGELEIA_API_URL        = os.getenv('AGELEIA_API_URL', '').rstrip('/')
AGELEIA_API_KEY        = os.getenv('INTEGRATION_API_KEY', '')
AGELEIA_NOTIFY_ENABLED = bool(AGELEIA_API_URL and AGELEIA_API_KEY)
if AGELEIA_BASE_URL:
    logger.info(f"  Agelia JWT proxy: ON ({AGELEIA_BASE_URL})")
else:
    logger.info("  Agelia JWT proxy: OFF (set AGELEIA_BASE_URL to enable)")
if AGELEIA_NOTIFY_ENABLED:
    logger.info(f"  Agelia mute summary (M2M read-only): ON ({AGELEIA_API_URL})")


def _block_path_configured() -> bool:
    """Returns True if Pallas can issue an IP block at the perimeter.

    Available via either:
      - Direct AWS: WAF_IP_SET_ID + NETWORK_FW_RULE_GROUP_ARN env vars set
        (optionally with AWS_ASSUME_ROLE_ARN for cross-account actions)
      - Agelia proxy: AGELEIA_BASE_URL set (handles WAF + Network Firewall + audit)

    Exposed by /api/meta so the frontend can hide the Block button when no
    blocking path is available.
    """
    has_aws = bool(os.getenv("WAF_IP_SET_ID") and os.getenv("NETWORK_FW_RULE_GROUP_ARN"))
    return has_aws or bool(AGELEIA_BASE_URL)


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

    Two auth modes:
      1. JWT-forward (default): caller has Authorization: Bearer header from
         a real Keycloak SSO session. Forward as-is to Agelia which validates
         the JWT and uses preferred_username as the audit actor.
      2. Iframe-bridge: caller is the Wazuh-dashboard-embedded Pallas iframe.
         No raw Keycloak JWT is available to the iframe (OSD security plugin
         doesn't expose it), but the parent posted the username via
         postMessage and the frontend sends it as X-Pallas-Iframe-User.
         Fall back to the shared integration key + forward the iframe-user
         header so Agelia can record the claimed actor in its audit row.
         (Agelia still verifies trust of the integration key, and the iframe
         user header is recorded as a *claim* — not a verified identity —
         which is the correct semantic for a perimeter-trusted intermediary.)

    Returns (response_dict, status_code). Never raises — caller decides how to
    surface errors.
    """
    if not AGELEIA_BASE_URL:
        return {"error": "Agelia integration not configured (AGELEIA_BASE_URL missing)"}, 503
    analyst_token   = request.headers.get("Authorization", "")
    iframe_user     = request.headers.get("X-Pallas-Iframe-User", "").strip()
    headers = {
        "Content-Type": "application/json",
        "X-Initiated-From": "pallas",
    }
    if analyst_token:
        headers["Authorization"] = analyst_token
    elif iframe_user and AGELEIA_API_KEY:
        # Iframe-bridge mode: no per-user JWT available. Use the shared
        # machine-to-machine integration key and forward the claimed actor
        # so Agelia can log it (Agelia is responsible for deciding whether
        # to honor the claim or treat it as a service-account action).
        headers["X-Integration-Key"]      = AGELEIA_API_KEY
        headers["X-Pallas-Iframe-User"]   = iframe_user
        headers["X-Initiated-From"]       = "pallas-iframe"
        logger.info(
            "Agelia iframe-bridge call %s %s claimed_user=%s",
            method, path, iframe_user,
        )
    try:
        client = _get_agelia_client()
        resp = await client.request(
            method,
            f"{AGELEIA_BASE_URL}{path}",
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

    # RE-ENABLE NEXT RELEASE — Agelia mute summary returns alongside Agelia.
    # When AGELEIA_BASE_URL is unset (standalone release), short-circuit so the
    # alert-annotation loop in /api/narrative/generate sees an empty mute set
    # and renders all alerts as not muted.
    if not AGELEIA_BASE_URL:
        return []

    if not AGELEIA_NOTIFY_ENABLED:
        return []

    now = time.monotonic()
    if now - _mute_summary_fetched_at < _MUTE_SUMMARY_TTL:
        return _mute_summary_cache

    try:
        client = _get_agelia_client()
        resp = await client.get(
            f"{AGELEIA_API_URL}/api/integrations/mute/active-summary",
            headers={"X-Integration-Key": AGELEIA_API_KEY},
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

    # RE-ENABLE NEXT RELEASE — Agelia blocked-IP summary returns alongside Agelia.
    # When AGELEIA_BASE_URL is unset (standalone release), return empty set so
    # alerts render without the "IP Blocked" badge derived from Agelia's view.
    # (Direct AWS block actions still work — this just suppresses the global
    # blocked-IPs annotation that comes from Agelia.)
    if not AGELEIA_BASE_URL:
        return set()

    if not AGELEIA_NOTIFY_ENABLED:
        return set()

    now = time.monotonic()
    if now - _blocked_ips_fetched_at < _BLOCKED_IPS_TTL:
        return _blocked_ips_cache

    try:
        client = _get_agelia_client()
        resp = await client.get(
            f"{AGELEIA_API_URL}/api/integrations/firewall/active-summary",
            headers={"X-Integration-Key": AGELEIA_API_KEY},
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
                "name": "get_suricata_medium_alerts",
                "description": "Get medium severity (3) Suricata IDS alerts",
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
                "name": "get_suricata_low_alerts",
                "description": "Get low severity (>= 4) Suricata IDS alerts",
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
                "name": "get_suricata_top_destinations",
                "description": "Top destination IPs targeted in Suricata IDS alerts (mirror of top_attackers but for destination IPs).",
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


# ============================================================
# Agent-not-found helpers (used by every agent-specific tool)
# ============================================================
def _is_agent_not_found(e: Exception) -> bool:
    """Return True if the Wazuh API exception indicates the agent does not exist (HTTP 404)."""
    err = str(e).lower()
    return "404" in err or "not found" in err or "not_found" in err


def _agent_not_found_response(agent_id: str) -> Dict[str, Any]:
    """Standard 'agent not found' response for any agent-specific tool that hits a Wazuh 404."""
    msg = (
        f"## Agent {agent_id} Not Found\n\n"
        f"The Wazuh manager has no record of agent `{agent_id}`.\n\n"
        f"**Possible causes:**\n"
        f"- The agent ID is incorrect (typo or wrong number)\n"
        f"- The agent has been removed or decommissioned\n"
        f"- The agent has not registered with the manager yet\n\n"
        f"**Next steps:**\n"
        f"- Run `show active agents` to see registered agent IDs\n"
        f"- Run `show disconnected agents` to see agents that registered but are offline\n"
        f"- Verify the agent ID in your asset inventory"
    )
    return {"content": [{"type": "text", "text": msg}]}



# === PALLAS_CLOUD_FORMATTERS === DO NOT REMOVE
# Markdown formatters for cloud-module / Defender / Intune tool responses.
# Replaces the previous raw json.dumps output that violated the NO-RAW-JSON rule.

def _pallas_md_safe(v, max_len=80):
    """Stringify and escape for markdown table cells."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        s = str(v)
    else:
        s = str(v)
    s = s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if max_len and len(s) > max_len:
        s = s[:max_len - 1] + "..."
    return s


def _pallas_get_path(obj, path):
    """Resolve dotted-path through nested dicts. Returns '' if missing."""
    cur = obj
    for k in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return ""
        if cur is None:
            return ""
    return cur


def _format_cloud_list(title, result, columns, sub_title=""):
    """Render a list-of-records response as a markdown section.
    columns: [(header, dotted_key, max_col_width), ...]
    """
    if not isinstance(result, dict):
        return f"## {title}\n\n_Unexpected response shape._"
    if result.get("error"):
        return f"## {title}\n\n**Tool error:** `{result['error']}`\n\n_Showing 0 records._"
    data = result.get("data") or {}
    items = data.get("affected_items") or []
    total = data.get("total_affected_items", len(items))

    lines = [f"## {title}"]
    if sub_title:
        lines.append(f"_{sub_title}_")
    lines.append(f"**Total found:** {total}  |  **Showing:** {len(items)}")
    lines.append("")
    if not items:
        lines.append("_No records found._")
        return "\n".join(lines)

    # Header row
    headers = [c[0] for c in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for item in items:
        row = []
        for header, key, max_w in columns:
            val = _pallas_get_path(item, key)
            row.append(_pallas_md_safe(val, max_len=max_w))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_cloud_summary(title, result, breakdowns, sub_title=""):
    """Render a summary response (with breakdown lists) as markdown.
    breakdowns: [(section_title, key_in_summary, item_label, count_label), ...]
    """
    if not isinstance(result, dict):
        return f"## {title}\n\n_Unexpected response shape._"
    if result.get("error"):
        return f"## {title}\n\n**Tool error:** `{result['error']}`"
    data = result.get("data") or {}
    summary = data.get("summary") or {}
    total = summary.get("total_events") or summary.get("total_vulnerabilities") or \
            summary.get("total_audit_events") or data.get("total_affected_items", 0)

    lines = [f"## {title}"]
    if sub_title:
        lines.append(f"_{sub_title}_")
    lines.append(f"**Total:** {total}")
    lines.append("")

    for section_title, key, item_label, count_label in breakdowns:
        bucket_list = summary.get(key) or []
        if not bucket_list:
            continue
        lines.append(f"### {section_title}")
        lines.append(f"| {item_label} | {count_label} |")
        lines.append("|---|---:|")
        for b in bucket_list[:20]:
            if not isinstance(b, dict):
                continue
            # Pick whatever key isn't 'count' to be the label
            label_v = None
            for k, v in b.items():
                if k != "count":
                    label_v = v
                    break
            count_v = b.get("count", "")
            lines.append(f"| {_pallas_md_safe(label_v, max_len=60)} | {count_v} |")
        lines.append("")

    # Optional summary extras (CVSS stats etc.)
    if "cvss_stats" in summary and isinstance(summary["cvss_stats"], dict):
        cv = summary["cvss_stats"]
        lines.append("### CVSS Statistics")
        lines.append(f"- **count:** {cv.get('count', '?')}  ")
        lines.append(f"- **min:** {cv.get('min', '?')}  ")
        lines.append(f"- **max:** {cv.get('max', '?')}  ")
        if cv.get("avg") is not None:
            lines.append(f"- **avg:** {round(cv['avg'], 2)}")
        lines.append("")

    for extra_key in ("with_public_exploit", "with_exploit_verified", "with_exploit_in_kit"):
        if extra_key in summary:
            label = extra_key.replace("with_", "").replace("_", " ").title()
            lines.append(f"- **{label}:** {summary[extra_key]}")

    return "\n".join(lines)


# Column definitions reused below
_PALLAS_COLS_OFFICE365 = [
    ("Time", "@timestamp", 22),
    ("Operation", "data.office365.Operation", 30),
    ("User", "data.office365.UserId", 40),
    ("Client IP", "data.office365.ClientIPAddress", 24),
    ("Workload", "data.office365.Workload", 18),
]
_PALLAS_COLS_GITHUB = [
    ("Time", "@timestamp", 22),
    ("Action", "data.github.action", 30),
    ("Actor", "data.github.actor", 32),
    ("Repo", "data.github.repo", 40),
    ("Org", "data.github.org", 24),
]
_PALLAS_COLS_AWS = [
    ("Time", "@timestamp", 22),
    ("Rule", "rule.id", 8),
    ("Severity", "data.aws.severity", 9),
    ("Source", "data.aws.source", 14),
    ("Type", "data.aws.type", 40),
    ("Description", "data.aws.description", 60),
]
_PALLAS_COLS_DEFENDER_AUDIT = [
    ("Time", "timestamp", 22),
    ("Action", "action", 18),
    ("Machine", "computer_dns_name", 26),
    ("Performed By", "performed_by", 22),
    ("Role", "role", 18),
    ("Status", "status", 10),
]
_PALLAS_COLS_DEFENDER_MACHINES = [
    ("Machine ID", "machine_id", 42),
    ("Tenant", "tenant_id", 36),
    ("Fetched", "fetched_at", 22),
    ("Primary User", "primary_user.accountName", 30),
]
_PALLAS_COLS_DEFENDER_VULNS = [
    ("CVE", "id", 18),
    ("Severity", "severity", 9),
    ("CVSS", "cvssV3", 6),
    ("Exposed", "exposedMachines", 8),
    ("Public Exploit", "publicExploit", 7),
    ("Status", "status", 22),
    ("Published", "publishedOn", 22),
    ("Description", "description", 80),
]
_PALLAS_COLS_INTUNE_AUDIT = [
    ("Time", "activity_date_time", 26),
    ("Activity Type", "activity_type", 38),
    ("Operation", "activity_operation_type", 12),
    ("Result", "activity_result", 10),
    ("Actor", "actor_user_principal_name", 36),
    ("Category", "category", 22),
]
# === END PALLAS_CLOUD_FORMATTERS ===

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
            # 2026-05-15: forward min_level/max_level so severity->level
            # translation in _build_arguments actually reaches the indexer
            # (Critical=15+, High=12-14, Medium=7-11, Low=0-6 per dashboard).
            min_level = arguments.get("min_level")
            max_level = arguments.get("max_level")
            agent_id = arguments.get("agent_id")
            timestamp_start = arguments.get("timestamp_start")
            timestamp_end = arguments.get("timestamp_end")
            # 2026-05-17: also forward time_range so "last 30m / 4h / 7d"
            # actually filters the alert search (previously dropped).
            time_range = arguments.get("time_range") or "24h"
            result = await wazuh_client.get_alerts(
                limit=limit, rule_id=rule_id, level=level,
                min_level=min_level, max_level=max_level,
                agent_id=agent_id, timestamp_start=timestamp_start,
                timestamp_end=timestamp_end, time_range=time_range
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

            try:
                result = await wazuh_client.get_agents(agent_id=agent_id, **kwargs)
            except Exception as e:
                if agent_id and _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise
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
            try:
                result = await wazuh_client.check_agent_health(agent_id)
            except Exception as e:
                if _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise

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
            try:
                result = await wazuh_client.get_agent_processes(agent_id, limit)
            except Exception as e:
                if _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise

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
            try:
                result = await wazuh_client.get_agent_ports(agent_id, limit=100)
            except Exception as e:
                if _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise

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
            try:
                result = await wazuh_client.get_agent_configuration(agent_id)
            except Exception as e:
                if _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise
            return {"content": [{"type": "text", "text": f"Agent Configuration:\n{json.dumps(result, indent=2)}"}]}

        # Vulnerability Management Tools
        elif tool_name == "get_wazuh_vulnerabilities":
            agent_id = arguments.get("agent_id")
            severity = arguments.get("severity")
            os_filter = arguments.get("os")
            cve_id = arguments.get("cve_id")  # 2026-05-18: was dropped — caused CVE search to return all vulns
            limit = arguments.get("limit", 10000)
            logger.error(f"TOOL CALL: get_wazuh_vulnerabilities")
            logger.error(f"   Arguments received: {arguments}")
            logger.error(f"   agent_id={agent_id}, severity={severity}, os={os_filter}, cve_id={cve_id}, limit={limit}")
            try:
                result = await wazuh_client.get_vulnerabilities(
                    agent_id=agent_id, severity=severity, os=os_filter, cve_id=cve_id, limit=limit
                )
            except Exception as e:
                if agent_id and _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise
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
                # Short-circuit on Wazuh 404 (agent doesn't exist) — give the user a clean message
                if agent_id and _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
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
                _agent = threat.get('agent', {}) if isinstance(threat.get('agent'), dict) else {}
                output += f"### {i}. {rule.get('description', 'Unknown Threat')}\n"
                output += f"- **Level:** {rule.get('level')} "
                output += f"- **Rule ID:** {rule.get('id')}\n"
                output += f"- **Agent:** {_agent.get('name', 'N/A')}\n"
                # Additive: emit the numeric agent ID on its own line so the
                # frontend can deep-link the agent name to its Athena detail
                # page. The **Agent:** line above is unchanged, so the existing
                # name parse is unaffected.
                if _agent.get('id'):
                    output += f"- **Agent ID:** {_agent.get('id')}\n"
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
            try:
                result = await wazuh_client.run_compliance_check(framework, agent_id)
            except Exception as e:
                if agent_id and _is_agent_not_found(e):
                    return _agent_not_found_response(agent_id)
                raise
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

        # === PALLAS_CLOUD_MODULES === DO NOT REMOVE
        elif tool_name == "get_office365_events":
            result = await wazuh_client._indexer_client.get_office365_events(**arguments)
            text = _format_cloud_list("Office 365 Events", result, _PALLAS_COLS_OFFICE365, sub_title=f"time_range={arguments.get('time_range', '24h')}")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_office365_summary":
            result = await wazuh_client._indexer_client.get_office365_summary(**arguments)
            text = _format_cloud_summary("Office 365 Summary", result, [
                ("By Operation", "breakdown", "Operation", "Count"),
                ("Top Rule IDs", "top_rule_ids", "Rule ID", "Count"),
                ("Top Agents", "top_agents", "Agent", "Count"),
            ], sub_title=f"time_range={arguments.get('time_range', '24h')}")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_github_events":
            result = await wazuh_client._indexer_client.get_github_events(**arguments)
            text = _format_cloud_list("GitHub Events", result, _PALLAS_COLS_GITHUB, sub_title=f"time_range={arguments.get('time_range', '24h')}")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_github_summary":
            result = await wazuh_client._indexer_client.get_github_summary(**arguments)
            text = _format_cloud_summary("GitHub Summary", result, [
                ("By Action", "breakdown", "Action", "Count"),
                ("Top Rule IDs", "top_rule_ids", "Rule ID", "Count"),
                ("Top Agents", "top_agents", "Agent", "Count"),
            ], sub_title=f"time_range={arguments.get('time_range', '24h')}")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_aws_events":
            result = await wazuh_client._indexer_client.get_aws_events(**arguments)
            text = _format_cloud_list("AWS Events", result, _PALLAS_COLS_AWS, sub_title=f"time_range={arguments.get('time_range', '30d')}")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_aws_summary":
            result = await wazuh_client._indexer_client.get_aws_summary(**arguments)
            text = _format_cloud_summary("AWS Summary", result, [
                ("Breakdown", "breakdown", "Rule Description", "Count"),
                ("Top Rule IDs", "top_rule_ids", "Rule ID", "Count"),
                ("Top Agents", "top_agents", "Agent", "Count"),
            ], sub_title=f"time_range={arguments.get('time_range', '30d')}")
            return {"content": [{"type": "text", "text": text}]}
        # === END PALLAS_CLOUD_MODULES ===
                # === PALLAS_DEFENDER_INTUNE === DO NOT REMOVE
        elif tool_name == "get_defender_audit_events":
            result = await wazuh_client._indexer_client.get_defender_audit_events(**arguments)
            text = _format_cloud_list("Defender Audit Events", result, _PALLAS_COLS_DEFENDER_AUDIT)
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_defender_audit_summary":
            result = await wazuh_client._indexer_client.get_defender_audit_summary(**arguments)
            text = _format_cloud_summary("Defender Audit Summary", result, [
                ("By Action", "by_action", "Action", "Count"),
                ("By Status", "by_status", "Status", "Count"),
                ("By User", "by_user", "User", "Count"),
                ("By Role", "by_role", "Role", "Count"),
            ])
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_defender_machines":
            result = await wazuh_client._indexer_client.get_defender_machines(**arguments)
            text = _format_cloud_list("Defender Machines", result, _PALLAS_COLS_DEFENDER_MACHINES)
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_defender_vulnerabilities":
            result = await wazuh_client._indexer_client.get_defender_vulnerabilities(**arguments)
            text = _format_cloud_list("Defender Vulnerabilities", result, _PALLAS_COLS_DEFENDER_VULNS, sub_title="sorted by CVSS desc")
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_defender_vulnerability_summary":
            result = await wazuh_client._indexer_client.get_defender_vulnerability_summary(**arguments)
            text = _format_cloud_summary("Defender Vulnerability Summary", result, [
                ("By Severity", "by_severity", "Severity", "Count"),
                ("By Status", "by_status", "Status", "Count"),
            ])
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_intune_audit_events":
            result = await wazuh_client._indexer_client.get_intune_audit_events(**arguments)
            text = _format_cloud_list("Intune Audit Events", result, _PALLAS_COLS_INTUNE_AUDIT)
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_intune_audit_summary":
            result = await wazuh_client._indexer_client.get_intune_audit_summary(**arguments)
            text = _format_cloud_summary("Intune Audit Summary", result, [
                ("By Activity Type", "by_activity_type", "Activity Type", "Count"),
                ("By Operation", "by_operation", "Operation", "Count"),
                ("By Result", "by_result", "Result", "Count"),
                ("By Category", "by_category", "Category", "Count"),
                ("By Actor", "by_actor", "Actor", "Count"),
                ("By Component", "by_component", "Component", "Count"),
            ])
            return {"content": [{"type": "text", "text": text}]}
        elif tool_name == "get_intune_inventory":
            result = await wazuh_client._indexer_client.get_intune_inventory(**arguments)
            rt = arguments.get("resource_type", "devices")
            # Generic columns â€” Intune inventory schemas vary; show top-level keys.
            items = (result.get("data") or {}).get("affected_items", [])
            cols = []
            if items and isinstance(items[0], dict):
                # Pick up to 6 most-useful-looking keys, skipping deeply nested objects
                for k in list(items[0].keys())[:7]:
                    cols.append((k.replace("_", " ").title(), k, 36))
            else:
                cols = [("Key", "_id", 40)]
            text = _format_cloud_list(f"Intune Inventory ({rt})", result, cols)
            return {"content": [{"type": "text", "text": text}]}
        # === END PALLAS_DEFENDER_INTUNE ===
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

        # ----------------------------------------------------------------
        # CLOUDFLARE TOOLS (Phase 1 MVP)
        # ----------------------------------------------------------------
        elif tool_name == "get_cloudflare_http_summary":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client._indexer_client.get_cloudflare_http_summary(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_http_errors":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 50)
            result = await wazuh_client._indexer_client.get_cloudflare_http_errors(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_firewall_events":
            time_range = arguments.get("time_range", "24h")
            action = arguments.get("action")
            limit = arguments.get("limit", 50)
            result = await wazuh_client._indexer_client.get_cloudflare_firewall_events(time_range, action, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_firewall_top_attackers":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            result = await wazuh_client._indexer_client.get_cloudflare_firewall_top_attackers(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_firewall_top_rules":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 20)
            result = await wazuh_client._indexer_client.get_cloudflare_firewall_top_rules(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_security_summary":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client._indexer_client.get_cloudflare_security_summary(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        # Cloudflare Phase 2
        elif tool_name == "get_cloudflare_http_top_paths":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_http_top_paths(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_http_top_clients":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_http_top_clients(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_http_search":
            query = arguments.get("query")
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 50)
            result = await wazuh_client._indexer_client.get_cloudflare_http_search(query, time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_cache_performance":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client._indexer_client.get_cloudflare_cache_performance(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_cache_top_misses":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_cache_top_misses(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_dns_summary":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client._indexer_client.get_cloudflare_dns_summary(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_dns_top_queries":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_dns_top_queries(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_dns_errors":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_dns_errors(time_range, limit)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_workers_summary":
            time_range = arguments.get("time_range", "24h")
            result = await wazuh_client._indexer_client.get_cloudflare_workers_summary(time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_cloudflare_workers_errors":
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 25)
            result = await wazuh_client._indexer_client.get_cloudflare_workers_errors(time_range, limit)
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

        elif tool_name == "get_suricata_medium_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            limit = arguments.get("limit", 50)
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_medium_alerts(limit=limit, time_range=time_range)
            output = json.dumps(result, indent=2, default=str)
            return {"content": [{"type": "text", "text": output}]}

        elif tool_name == "get_suricata_low_alerts":
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            limit = arguments.get("limit", 50)
            time_range = arguments.get("time_range", "24h")
            result = await suricata_client.get_low_alerts(limit=limit, time_range=time_range)
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

        elif tool_name == "get_suricata_top_destinations":
            # M6c (2026-06-10) - mirror of top_attackers but extracts the
            # destination-IP aggregation. Same underlying network_analysis call;
            # the formatter focuses only on destinations so the analyst gets a
            # destination-IP ranking rather than the dual src/dst view of the
            # full network-analysis tool.
            if suricata_client is None:
                return {"content": [{"type": "text", "text": "Suricata integration is not enabled. Set SURICATA_ENABLED=true and configure WAZUH_INDEXER_HOST."}]}
            time_range = arguments.get("time_range", "24h")
            limit = arguments.get("limit", 10)
            result = await suricata_client.get_network_analysis(time_range=time_range, limit=limit)
            destinations_data = {
                "top_destinations": result.get("top_dest_ips", []),
                "total_alerts": result.get("total_alerts", 0),
                "time_range": time_range,
            }
            output = json.dumps(destinations_data, indent=2, default=str)
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
    # PALLAS_P3_ARG_STRIP: strip kwargs not in the registered input_schema
    # so a hallucinated LLM arg (e.g. time_range on a tool that does not
    # accept one) cannot crash the dispatcher.
    try:
        from wazuh_mcp_server.tool_registry import ToolRegistry
        _meta = ToolRegistry.TOOLS.get(tool_name)
        if _meta and isinstance(arguments, dict) and isinstance(_meta.input_schema, dict):
            _allowed = set(_meta.input_schema.keys())
            _dropped = [k for k in list(arguments.keys()) if k not in _allowed]
            if _dropped:
                logger.warning(
                    f"[ARG-STRIP] {tool_name}: dropping unknown args {_dropped}; "
                    f"allowed={sorted(_allowed)}"
                )
                arguments = {k: v for k, v in arguments.items() if k in _allowed}
    except Exception as _strip_exc:
        logger.warning(f"[ARG-STRIP] guard failed: {_strip_exc}; passing args through")

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

    # === PALLAS_WS_AUTH === DO NOT REMOVE

    _pallas_expected = (os.getenv('MCP_API_KEY') or '').strip()

    if _pallas_expected:

        _pallas_provided = (websocket.query_params.get('api_key') or '').strip()

        if not _pallas_provided:

            _pallas_auth_hdr = websocket.headers.get('authorization', '')

            if _pallas_auth_hdr.lower().startswith('bearer '):

                _pallas_provided = _pallas_auth_hdr[7:].strip()

        if not _pallas_provided:

            _pallas_provided = (websocket.headers.get('x-api-key') or '').strip()

        import hmac as _pallas_hmac

        if not _pallas_provided or not _pallas_hmac.compare_digest(_pallas_provided, _pallas_expected):

            await websocket.close(code=1008, reason='Invalid API key')

            return

    # === END PALLAS_WS_AUTH ===

    await websocket.accept()

    # PALLAS_P4_IDENTITY_KEY (Track A2) - build a durable session id so the
    # same analyst's tabs/devices share memory, while different analysts stay
    # isolated. Key format: {tenant}:{api_key_fingerprint}:{conversation_id}
    # A client can pass ?conversation_id=<uuid> to resume an existing thread;
    # absent, a fresh UUID is generated for this WebSocket.
    from wazuh_mcp_server.conversation_memory import derive_conversation_key
    _tenant_name = os.getenv("TENANT_NAME", "default").strip().lower()
    _client_conv_id = (websocket.query_params.get("conversation_id") or "").strip()
    _conversation_id = _client_conv_id or str(uuid.uuid4())
    # _pallas_provided is only set when auth is enabled (see PALLAS_WS_AUTH block
    # above). Fall back to query-string / header without enforcement so the
    # fingerprint is still stable for auth-less local dev.
    _api_key_for_identity = locals().get("_pallas_provided")
    if not _api_key_for_identity:
        _api_key_for_identity = (
            websocket.query_params.get("api_key")
            or (websocket.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            or websocket.headers.get("x-api-key")
            or None
        )
    session_id = derive_conversation_key(
        tenant=_tenant_name,
        api_key=_api_key_for_identity,
        conversation_id=_conversation_id,
    )
    _analyst_id = session_id.split(":")[1] if ":" in session_id else "anon"
    logger.info(
        f"WebSocket chat session started: tenant={_tenant_name} "
        f"analyst={_analyst_id} conv={_conversation_id[:8]} from {websocket.client}"
    )

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
                # NLP Sprint 1: wire ConversationMemory + ReferenceResolver for multi-turn.
                # When a query is a follow-up, carry entities forward from the last turn
                # so anaphora ("show its ports", "now that agent", "focus on those") works.
                _resolved_query = data
                _is_followup = False
                _followup_pattern = None  # M2a (2026-06-10): which regex matched
                _carried = {}
                _conv_summary = ""
                _prev_tool = None
                _prev_result_type = None
                try:
                    # M10 (2026-06-10): chat-session-based follow-up detection.
                    # Treat ANY query after the first in the same chat as a
                    # follow-up. "New Chat" gives the analyst the explicit reset
                    # signal (handled by the frontend regenerating
                    # conversation_id, which yields a new session_id). The regex
                    # matcher is kept only so M3 / M6d can still distinguish a
                    # *narrow* modifier (filter/narrow/now) from a *pivot*
                    # modifier (drill/show their) for source-pin and
                    # dimension-swap decisions.
                    _followup_pattern = ReferenceResolver.match_pattern(data)
                    _last_turn = _conversation_memory.get_last_turn(session_id)
                    _is_followup = _last_turn is not None
                    if _is_followup:
                        _resolved_query, _carried = ReferenceResolver.resolve_references(data, _last_turn)
                        _prev_tool = getattr(_last_turn, "tool_used", None)
                        _prev_result_type = getattr(_last_turn, "result_type", None)
                    # PALLAS_P3_CTX_GATE: only carry verbose conversation summary forward
                    # when the new query LOOKS like a follow-up. Otherwise stale prior-turn
                    # context (e.g. agents + vulns) biases Qwen on new topics (e.g. PCI).
                    try:
                        if _is_followup:
                            _conv_summary = _conversation_memory.get_context_summary(session_id) or ""
                        else:
                            _conv_summary = ""
                    except Exception:
                        _conv_summary = ""
                except Exception as _conv_exc:
                    logger.warning(f"Chat [{session_id[:8]}]: conversation memory wiring failed: {_conv_exc}")

                # === PALLAS_P3_AUTO_ESCALATE === Phase 3 smart hybrid
                # Decide if this query should run through the agent loop.
                # Triggers (any one):
                #   - JSON frame with mode=="agent" (Week-2 contract preserved)
                #   - JSON frame with mode=="fast" -> opt OUT of agent (escape hatch)
                #   - Plain text matches escalation keywords (investigate/end-to-end/...)
                #   - Plain text mentions multiple agents joined by "and"
                _agent_mode = False
                _agent_query_text = data
                _explicit_fast = False
                try:
                    _agent_msg = json.loads(data) if isinstance(data, str) else {}
                    if isinstance(_agent_msg, dict):
                        _m = _agent_msg.get("mode")
                        if _m == "agent":
                            _agent_mode = True
                            _agent_query_text = _agent_msg.get("query") or _agent_msg.get("content") or data
                        elif _m == "fast":
                            _explicit_fast = True
                            _agent_query_text = _agent_msg.get("query") or _agent_msg.get("content") or data
                except Exception:
                    pass
                if not _agent_mode and not _explicit_fast:
                    # Pattern-based auto-escalation on plain text
                    _q_low = (data or "").lower()
                    _ESC_KW = (
                        "investigate", "end-to-end", "end to end", "compare ",
                        "why is", "why are", "why does", "explain why", "correlate",
                        "hunt for", "find systems", "across", " versus ", " vs ",
                        "what changed", "deep dive",
                    )
                    if any(k in _q_low for k in _ESC_KW):
                        _agent_mode = True
                        _agent_query_text = data
                    else:
                        # multi-entity heuristic: two known agent names connected by "and"
                        # cheap check - if query has " and " plus at least 2 dash-separated tokens
                        if " and " in _q_low:
                            _dashed = sum(1 for tok in _q_low.split() if "-" in tok and len(tok) > 3)
                            if _dashed >= 2:
                                _agent_mode = True
                                _agent_query_text = data
                # === END PALLAS_P3_AUTO_ESCALATE ===

                if _agent_mode:
                    try:
                        from wazuh_mcp_server.agent_loop import run_agent
                        from wazuh_mcp_server.agent_llm_client import AgentLLMClient
                        from wazuh_mcp_server.query_orchestrator import ToolExecutor
                        _agent_llm = AgentLLMClient()
                        _agent_exec = ToolExecutor(execute_mcp_tool)
                        active_task = asyncio.get_running_loop().create_task(
                            run_agent(_agent_query_text, session_id, _agent_llm, _agent_exec)
                        )
                        try:
                            _agent_result = await active_task
                        except asyncio.CancelledError:
                            logger.info(f"Chat [{session_id[:8]}]: agent run cancelled by client")
                            continue
                        finally:
                            active_task = None
                        # === PALLAS_P3_STRIP_METADATA ===
                        # UI gets only the markdown; trace + stats go to container logs.
                        try:
                            logger.info(
                                f"[AGENT-AUDIT] session={session_id[:8]} "
                                f"tier=2-agent iters={_agent_result.get('iterations', 0)} "
                                f"tokens={_agent_result.get('tokens_used', 0)} "
                                f"wall={_agent_result.get('wall_time_s', 0)}s "
                                f"stopped={_agent_result.get('stopped_reason', '')}"
                            )
                            for _i_t, _tr in enumerate(_agent_result.get("trace", []) or [], 1):
                                logger.info(
                                    f"[AGENT-TRACE] session={session_id[:8]} "
                                    f"step={_i_t} tool={_tr.get('tool')} "
                                    f"ok={_tr.get('success')} latency={_tr.get('latency_ms')}ms"
                                )
                        except Exception:
                            pass
                        # PALLAS_P3_NEVER_BLANK_V2: ensure UI gets a useful answer.
                        # Treat as incomplete if final is empty OR missing the standard
                        # markdown shape (## Response section).
                        _final = _agent_result.get("final_answer", "") if isinstance(_agent_result, dict) else ""
                        _trace = (_agent_result or {}).get("trace", []) if isinstance(_agent_result, dict) else []
                        _stopped = (_agent_result or {}).get("stopped_reason", "") if isinstance(_agent_result, dict) else ""
                        _final_stripped = (_final or "").strip()
                        _has_shape = ("## Response" in _final_stripped) or ("## SOC Analysis" in _final_stripped)
                        if not _final_stripped or not _has_shape:
                            # ============================================================
                            # PALLAS_P4_RICH_GUIDANCE_V2 (Layer 6, v2.4 2026-06-07)
                            # When the agent loop doesn't produce a markdown-shaped
                            # answer, synthesize from the trace: per-tool data preview
                            # + entity-aware suggested follow-ups. Replaces the prior
                            # PALLAS_P3_NEVER_BLANK_V2 "Unable to assess" template
                            # which discarded successful tool data.
                            # ============================================================
                            _rg2_on = os.getenv("PALLAS_P4_RICH_GUIDANCE_V2", "true").lower() != "false"
                            _ok_trace = [_t for _t in _trace if _t.get("success")]
                            _fail_trace = [_t for _t in _trace if not _t.get("success")]
                            _ok = len(_ok_trace)
                            _fail = len(_fail_trace)
                            _tools_seen = sorted({_t.get("tool") or "?" for _t in _trace})
                            _ok_tools = sorted({_t.get("tool") or "?" for _t in _ok_trace})
                            _fail_tools = sorted({_t.get("tool") or "?" for _t in _fail_trace})

                            if _rg2_on:
                                import re as _re_rg2
                                _qtxt = (_agent_query_text or "").strip()
                                _qlow = _qtxt.lower()
                                _agent_m = _re_rg2.search(r"agent\s*(\d{1,5})", _qlow)
                                _agent_id = _agent_m.group(1) if _agent_m else None
                                _mitre_m = _re_rg2.search(r"\bT\d{4}(?:\.\d{3})?\b", _qtxt, _re_rg2.IGNORECASE)
                                _mitre_code = _mitre_m.group(0).upper() if _mitre_m else None
                                _has_time = bool(_re_rg2.search(
                                    r"\b(last|past|since|today|yesterday|hours?|days?|weeks?)\b", _qlow))

                                # Per-tool preview (top 6 successful)
                                _what_lines = []
                                for _t in _ok_trace[:6]:
                                    _tn = _t.get("tool") or "?"
                                    _sm = (_t.get("summary") or "").strip()
                                    if _sm:
                                        _prev = "\n".join(_sm.split("\n")[:6])[:480]
                                        _what_lines.append(f"### `{_tn}`\n{_prev}")
                                _what_md = (
                                    "\n\n".join(_what_lines) if _what_lines
                                    else "_No tools returned usable data before the loop stopped._"
                                )

                                # Errors (compact)
                                _err_md = ""
                                if _fail_trace:
                                    _err_md = "**Errors observed:**\n" + "\n".join(
                                        f"- `{_t.get('tool')}`: {(_t.get('summary') or '')[:140]}"
                                        for _t in _fail_trace[:5]
                                    ) + "\n\n"

                                # Suggested follow-ups (3-5)
                                _sugg = []
                                if _agent_id:
                                    _sugg.append(f"show recent alerts for agent {_agent_id} last 24h")
                                    _sugg.append(f"show critical vulnerabilities for agent {_agent_id}")
                                if _mitre_code:
                                    _sugg.append(f"show MITRE {_mitre_code} alerts last 24h")
                                if "get_wazuh_alerts" in _ok_tools and not _agent_id:
                                    _sugg.append("show critical alerts last 24h")
                                if any(t.startswith("get_suricata") for t in _ok_tools):
                                    _sugg.append("show top suricata signatures last 24h")
                                if "get_wazuh_vulnerabilities" in _ok_tools and not _agent_id:
                                    _sugg.append("show critical vulnerabilities for the fleet")
                                if not _mitre_code:
                                    _sugg.append("show MITRE T1059 PowerShell execution last 24h")
                                if not _has_time:
                                    _sugg.append("show active agents and their critical vulnerabilities")
                                _seen = set(); _sugg_dedup = []
                                for _s in _sugg:
                                    if _s not in _seen:
                                        _seen.add(_s); _sugg_dedup.append(_s)
                                _sugg_final = _sugg_dedup[:5]
                                for _pad in [
                                    "show active agents",
                                    "show critical vulnerabilities for dev-zeek",
                                    "show recent alerts",
                                ]:
                                    if len(_sugg_final) >= 5:
                                        break
                                    if _pad not in _seen:
                                        _seen.add(_pad); _sugg_final.append(_pad)
                                _sugg_md = "\n".join(f"- `{_s}`" for _s in _sugg_final)

                                # Risk summary speaks to what we have
                                if _ok > 0:
                                    _risk_sum = (
                                        f"Based on partial data from {_ok} tool(s) before "
                                        f"stop (`{_stopped or 'unknown'}`). Pick one of the suggested "
                                        "queries below for focused depth."
                                    )
                                    _priority = "MEDIUM (partial signal worth reviewing)"
                                    _threats_line = (
                                        f"Indicators present across {_ok} successful tool(s); "
                                        "see partial data above."
                                    )
                                else:
                                    _risk_sum = (
                                        "Insufficient data — no tool returned a usable result. "
                                        "Try one of the suggested queries below."
                                    )
                                    _priority = "LOW (inconclusive)"
                                    _threats_line = "None identified (no data gathered)."

                                _fb_lines = [
                                    "## Operator Request",
                                    f"{_agent_query_text}",
                                    "",
                                    "## Response",
                                    f"I gathered data from **{_ok} of {len(_trace)} tool(s)** before "
                                    f"stopping (`{_stopped or 'unknown'}`). Partial findings are below, "
                                    "followed by suggested follow-up queries that will give you focused depth.",
                                    "",
                                    "### What I found",
                                    _what_md,
                                    "",
                                ]
                                if _err_md:
                                    _fb_lines.append(_err_md.rstrip())
                                    _fb_lines.append("")
                                _fb_lines.extend([
                                    "### Suggested next queries",
                                    _sugg_md,
                                    "",
                                    "## SOC Analysis",
                                    f"**Risk Summary:** {_risk_sum}",
                                    "**Observations:**",
                                    f"- Successful tools: {', '.join(_ok_tools) or 'none'}",
                                    f"- Failed tools: {', '.join(_fail_tools) or 'none'}",
                                    f"- Stopped reason: `{_stopped or 'unknown'}`",
                                    f"- Iterations used: {len(_trace)}",
                                    f"**Potential Threats:** {_threats_line}",
                                    "**Recommended Actions:**",
                                    "1. Click or paste one of the suggested queries above to deepen the investigation.",
                                    "2. If a tool failed, the error message above shows the specific arg that needs correcting.",
                                    f"**Priority Level:** {_priority}",
                                ])
                                _final = "\n".join(_fb_lines)
                                # PALLAS_P4_RICH_GUIDANCE_V2 marker
                            else:
                                # Legacy template (rollback path)
                                _err_lines = []
                                for _t in _fail_trace:
                                    _err_lines.append(f"- `{_t.get('tool')}`: {(_t.get('summary') or '')[:140]}")
                                    if len(_err_lines) >= 3: break
                                _fb_lines = [
                                    "## Operator Request",
                                    f"{_agent_query_text}",
                                    "",
                                    "## Response",
                                    f"The agent investigated this query across **{len(_trace)} tool call(s)** "
                                    f"({_ok} succeeded, {_fail} failed) but did not converge on a complete answer "
                                    f"within its budget (stopped: `{_stopped or 'unknown'}`).",
                                    "",
                                    f"**Tools tried:** {', '.join(_tools_seen) if _tools_seen else 'none'}",
                                    "",
                                ]
                                if _err_lines:
                                    _fb_lines.append("**Errors observed:**")
                                    _fb_lines.extend(_err_lines)
                                    _fb_lines.append("")
                                _fb_lines.extend([
                                    "**Suggested next step:** try a tighter, single-source question — "
                                    "e.g. `show recent PowerShell alerts on agent 073` "
                                    "or `MITRE T1059.001 last 24h` — and chain follow-ups from there.",
                                    "",
                                    "## SOC Analysis",
                                    "**Risk Summary:** Unable to assess — investigation incomplete.",
                                    "**Observations:**",
                                    f"- Stopped reason: `{_stopped or 'unknown'}`",
                                    f"- Iterations used: {len(_trace)}",
                                    "**Potential Threats:** None identified (no data gathered).",
                                    "**Recommended Actions:**",
                                    "1. Re-run with a narrower question.",
                                    "2. Ops: check container audit log for `Tool execution error` lines.",
                                    "**Priority Level:** LOW (inconclusive)",
                                ])
                                _final = "\n".join(_fb_lines)
                                # PALLAS_P3_NEVER_BLANK_V2 marker (legacy)
                        await websocket.send_json({
                            "role": "bot",
                            "message": _final,
                            "warn": False,
                        })
                        # === END PALLAS_P3_STRIP_METADATA ===
                        continue
                    except Exception as _agent_exc:
                        logger.error(f"Chat [{session_id[:8]}]: agent mode failed, falling back to fast: {_agent_exc}")
                # === END PALLAS_AGENT_MODE_DISPATCH ===

                context = QueryContext(
                    original_query=_resolved_query,
                    timestamp=datetime.utcnow(),
                    session_id=session_id,
                    conversation_summary=_conv_summary,
                    carried_entities=(_carried or {}),
                    is_follow_up=_is_followup,
                    previous_tool=_prev_tool,
                    previous_result_type=_prev_result_type,
                    follow_up_pattern=_followup_pattern,  # M2a (2026-06-10)
                )

                # Live progress streaming: forward orchestrator phase frames to
                # the client so the loader shows REAL telemetry (sources/events/
                # signals) instead of a simulation. Best-effort — a send failure
                # never interrupts the query.
                async def _send_progress(_payload):
                    try:
                        await websocket.send_json({"type": "progress", **_payload})
                    except Exception:
                        pass

                # Process query in a cancellable task
                active_task = asyncio.get_running_loop().create_task(
                    _query_orchestrator.process_query(data, context, progress_cb=_send_progress)
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

                # NLP Sprint 1: record this turn into ConversationMemory so future
                # follow-up queries can carry entities forward.
                try:
                    _md = ws_response.get("metadata") or {}
                    _ent = (_md.get("entities") if isinstance(_md, dict) else None) or {}
                    if not isinstance(_ent, dict):
                        _ent = {}
                    _key_ids = []
                    for _k in ("agent_id", "agent_ids", "cve_id", "ip"):
                        _v = _ent.get(_k)
                        if isinstance(_v, list):
                            _key_ids.extend([str(x) for x in _v if x])
                        elif _v not in (None, ""):
                            _key_ids.append(str(_v))
                    _tool_used = _md.get("primary_tool") if isinstance(_md, dict) else None
                    _result_type = _md.get("correlation_type") if isinstance(_md, dict) else None
                    if not _result_type and isinstance(_tool_used, str):
                        if "vulnerab" in _tool_used:
                            _result_type = "vulnerabilities"
                        elif "alert" in _tool_used:
                            _result_type = "alerts"
                        elif "agent" in _tool_used:
                            _result_type = "agents"
                        else:
                            _result_type = "results"
                    _rec_count = 0
                    if isinstance(_md, dict):
                        for _ck in ("record_count", "total_count", "result_count"):
                            _cv = _md.get(_ck)
                            if isinstance(_cv, int):
                                _rec_count = _cv
                                break
                    # PALLAS_P4_STRUCTURED_MEM (Track A3) - persist the top 20
                    # affected_items so a follow-up like "filter that to high
                    # severity" can operate on real data without re-running the
                    # tool. Best-effort: ws_response.metadata.result_items is
                    # populated by the orchestrator's _normalize_tool_payload
                    # path; if absent (e.g. agent loop), we just store an
                    # empty list and follow-ups fall back to re-fetch.
                    _result_items: list = []
                    try:
                        if isinstance(_md, dict):
                            _ri = _md.get("result_items") or []
                            if isinstance(_ri, list):
                                _result_items = _ri[:20]
                    except Exception:
                        pass
                    _conversation_memory.add_turn(
                        session_id,
                        ConversationTurn(
                            query=data,
                            timestamp=datetime.utcnow(),
                            tool_used=(_tool_used or "unknown"),
                            entities=_ent,
                            result_summary=(response[:240] if isinstance(response, str) else ""),
                            result_type=(_result_type or "results"),
                            key_ids=_key_ids,
                            record_count=_rec_count,
                            result_items=_result_items,
                        ),
                        tenant=_tenant_name,
                        analyst_id=_analyst_id,
                    )
                except Exception as _tr_exc:
                    logger.warning(f"Chat [{session_id[:8]}]: turn recording failed: {_tr_exc}")

                # Send structured response back to client
                # M12 (2026-06-10) - guard against the client closing the WS
                # mid-query (e.g. analyst clicked "New Chat" while a long-running
                # SOC analysis was still being computed). The send_json would
                # raise WebSocketDisconnect(code=1006) and bubble up as a fake
                # "error" on the analyst's screen even though the result was
                # already abandoned client-side.
                try:
                    await websocket.send_json(ws_response)
                except (WebSocketDisconnect, RuntimeError, Exception) as _send_exc:
                    _exc_name = type(_send_exc).__name__
                    if "Disconnect" in _exc_name or "close" in str(_send_exc).lower() or "ClosedOK" in str(_send_exc) or "1005" in str(_send_exc) or "1006" in str(_send_exc):
                        logger.info(
                            f"Chat [{session_id[:8]}]: client disconnected before response could "
                            f"be delivered ({_exc_name}); dropping silently"
                        )
                        break  # client gone - exit the receive loop
                    raise

            except Exception as e:
                logger.error(f"Error processing chat message: {e}", exc_info=True)
                # M12 - if the client is already gone, don't try to send an
                # error frame; just exit gracefully.
                try:
                    await websocket.send_json({
                        "role": "bot",
                        "message": f" An error occurred while processing your query:\n\n{str(e)}\n\nPlease try rephrasing your question or contact support.",
                        "warn": True
                    })
                except Exception:
                    logger.info(f"Chat [{session_id[:8]}]: client gone, skipping error-frame send")
                    break

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
            "capabilities": {
                "agelia_enabled": bool(AGELEIA_BASE_URL),
                "mute_enabled": False,  # RE-ENABLE NEXT RELEASE — flip to bool(AGELEIA_BASE_URL)
                "block_enabled": _block_path_configured(),
                "auth_mode": "iframe-bridge",
                "jira_enabled": bool(jira_client_instance.is_configured),
            },
        }

        return JSONResponse(content=meta_response)

    except Exception as e:
        logger.error(f"Meta endpoint error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


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
    # Default fetch ceiling scales with the time window because the alert
    # truncation at line ~4290 (unified_alerts[:max_alerts]) runs BEFORE the
    # status_filter further down. With a flat 200 cap, busy SOCs on the 7-day
    # view were systematically under-reporting In Progress / Closed counts
    # and lists — only the most recent 200 critical alerts made it past the
    # truncation, and those were almost all "open", leaving the other tabs
    # empty even when older in-progress / closed work existed in the window.
    # Bumping the ceiling per range gives those tabs honest data on 7d while
    # keeping shorter windows fast. Caller can still override with an explicit
    # body.max_alerts when needed.
    _TR_MAX_ALERTS = {"12h": 200, "24h": 200, "48h": 400, "7d": 1500}
    max_alerts = body.get("max_alerts") or _TR_MAX_ALERTS.get(time_range, 200)

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
    # When the analyst's "Show muted" toggle is OFF (default), we exclude
    # muted alerts BEFORE pagination so each page is full of actionable
    # alerts. Without this, pages can render empty when many recent alerts
    # happen to match a mute rule. Frontend sends include_muted=true only
    # when the toggle is on. (Mute counts in the KPI tile remain accurate
    # because we still annotate every alert with _agelia_muted before
    # filtering.)
    include_muted   = bool(body.get("include_muted", False))
    # disable_mute is the frontend's "Mute is off this release" signal. When set,
    # the backend skips the entire Agelia mute annotation path: no alert is
    # flagged _agelia_muted=True, so the mute filter never excludes anything
    # from the list and the tab_counts never bucket to 'muted'. This keeps the
    # Open/In Progress/Closed tabs honest while the Mute UI is hidden — without
    # this, pre-existing Agelia mute rules silently absorb alerts and analysts
    # see Open=0 even when raw alert counts (wazuh_count + suricata_count) are
    # non-zero. Remove this opt-out when the Mute UI is re-enabled.
    disable_mute    = bool(body.get("disable_mute", False))
    # Optional grouping. When `group_by` is provided as a non-empty list of
    # field names, we fold filtered alerts into groups using a composite key
    # built from those fields. Pagination operates on GROUPS, not alerts.
    # Each returned group includes its full alert list so the frontend can
    # render per-alert detail without a second fetch when the analyst expands
    # a group. Today we accept ["rule_id","source","src_ip"] as the canonical
    # signature for "duplicate alerts of the same pattern from the same IP".
    group_by = body.get("group_by") or []
    if not isinstance(group_by, list):
        group_by = []
    group_by = [str(f).strip() for f in group_by if f]
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
    # Skip the Agelia mute rules fetch entirely when the client signals
    # disable_mute. Saves a network round-trip AND guarantees no alert can
    # be marked _agelia_muted (since `mute_rules` will be empty).
    mute_rules = [] if disable_mute else await _get_mute_summary()
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
        # Hide muted unless the analyst's "Show muted" toggle is on. This
        # runs before pagination so pages always contain actionable alerts.
        # KPI tile counts (muted/blocked) are computed BEFORE this filter,
        # so they still reflect the full picture.
        if not include_muted and alert.get("_agelia_muted"):
            return False
        # Severity (normalized field — both Wazuh and Suricata go through normalize)
        if severity_filter != "all":
            if (alert.get("severity") or "").lower() != severity_filter:
                return False
        # Source
        if source_filter != "all":
            if (alert.get("source") or "").lower() != source_filter:
                return False
        # Incident lifecycle status. Tab buckets cover multiple raw statuses
        # so the list filter aliases match the same buckets the counter uses:
        #   - 'in_progress' tab matches statuses 'in_progress' OR 'acknowledged'
        #   - 'closed' tab matches statuses 'closed' OR 'false_positive'
        # Without this aliasing the counter and list disagree (counter buckets
        # both, list strict-equals only one).
        if status_filter != "all":
            inc = alert.get("_incident") or {}
            current_status = (inc.get("status") or "open").lower()
            if status_filter == "in_progress":
                if current_status not in ("in_progress", "acknowledged"):
                    return False
            elif status_filter == "closed":
                if current_status not in ("closed", "false_positive"):
                    return False
            elif current_status != status_filter:
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

    # ── Tab counters ──────────────────────────────────────────────────
    # Compute per-lifecycle-tab counts. We run the non-status, non-mute
    # predicates once (severity, source, assigned, search) over unified_alerts
    # and bucket by status + agelia_muted. This gives the frontend accurate
    # badge counts for ALL tabs in one round-trip — analyst sees "Open: 12,
    # In Progress: 5, Closed: 200, Muted: 73" at a glance regardless of which
    # tab they're currently viewing.
    def _alert_matches_non_status_filters(alert):
        if severity_filter != "all":
            if (alert.get("severity") or "").lower() != severity_filter:
                return False
        if source_filter != "all":
            if (alert.get("source") or "").lower() != source_filter:
                return False
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
            matches_actor = any((v or "").lower() == af_lower for v in actor_fields)
            matches_assigned = (
                assigned_to_value
                and (af_lower in assigned_to_value.lower() or assigned_to_value.lower() == af_lower)
            )
            if not (matches_actor or matches_assigned):
                return False
        if search_query:
            haystack = " ".join([
                str(alert.get("rule_id") or ""),
                str(alert.get("signature_id") or ""),
                str(alert.get("source_ip") or ""),
                str(alert.get("src_ip") or ""),
                str(alert.get("dest_ip") or ""),
                str(alert.get("agent_name") or ""),
                str(alert.get("rule_description") or ""),
            ]).lower()
            if search_query not in haystack:
                return False
        return True

    tab_counts = {"open": 0, "in_progress": 0, "closed": 0, "muted": 0, "total": 0}
    for a in unified_alerts:
        if not _alert_matches_non_status_filters(a):
            continue
        if a.get("_agelia_muted"):
            tab_counts["muted"] += 1
        else:
            inc = a.get("_incident") or {}
            st = (inc.get("status") or "open").lower()
            if st == "open":
                tab_counts["open"] += 1
            elif st == "in_progress" or st == "acknowledged":
                tab_counts["in_progress"] += 1
            elif st in ("closed", "false_positive"):
                tab_counts["closed"] += 1
        tab_counts["total"] += 1

    # ── Optional grouping ─────────────────────────────────────────────
    # When group_by is requested, fold filtered alerts into groups using
    # composite key. Pagination then operates on groups. Each group includes
    # its full alert list, sorted newest-first within the group.
    groups = None
    if group_by:
        from collections import OrderedDict
        groups_map = OrderedDict()
        for a in filtered_alerts:
            key_parts = []
            for f in group_by:
                v = a.get(f)
                # http_xff first hop is the meaningful "true source IP" for
                # Suricata alerts behind XFF — match the existing src_ip
                # convention used elsewhere in the renderer.
                if f == "src_ip" and not v:
                    v = a.get("source_ip") or (a.get("http_xff") or "").split(",")[0].strip()
                key_parts.append(str(v or "").strip())
            sig_key = "|".join(key_parts)
            if sig_key not in groups_map:
                groups_map[sig_key] = {
                    "signature_key": sig_key,
                    "count":         0,
                    "first_seen":    a.get("timestamp"),
                    "last_seen":     a.get("timestamp"),
                    "rule_id":       a.get("rule_id"),
                    "src_ip":        a.get("src_ip") or a.get("source_ip"),
                    "source":        a.get("source"),
                    "severity":      a.get("severity"),
                    "alerts":        [],
                }
            g = groups_map[sig_key]
            g["count"] += 1
            g["alerts"].append(a)
            # Track first/last seen timestamps. unified_alerts is sorted
            # newest-first upstream, so the first alert we encounter for a
            # signature is the newest (last_seen) and the last is the oldest
            # (first_seen). Update both regardless to be defensive.
            ts = a.get("timestamp") or ""
            if ts and (not g["first_seen"] or ts < g["first_seen"]):
                g["first_seen"] = ts
            if ts and (not g["last_seen"] or ts > g["last_seen"]):
                g["last_seen"] = ts
        groups = list(groups_map.values())

    # ── Paginate ───────────────────────────────────────────────────────
    if groups is not None:
        # Group-based pagination — page over groups, return full alert lists
        # inside each group on the current page so the frontend can expand
        # them without a second fetch.
        total_units = len(groups)
        total_pages = max(1, (total_units + page_size - 1) // page_size)
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_groups = groups[start:end]
        # Also flatten to alert list for backwards-compat callers and so the
        # incident-store batch fetch + KPI tile counts still work unchanged.
        page_alerts = [a for g in page_groups for a in g["alerts"]]
    else:
        total_pages = max(1, (total_after_filter + page_size - 1) // page_size)
        # Clamp page so out-of-range requests return last page rather than empty
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_alerts = filtered_alerts[start:end]
        page_groups = None

    elapsed_ms = int((time.time() - start_time) * 1000)

    response_payload = {
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
            "tab_counts":          tab_counts,
            "page":                page,
            "page_size":           page_size,
            "total_pages":         total_pages,
            "time_range":          time_range,
            "min_level":           min_level,
            "scan_duration_ms":    elapsed_ms,
            "errors":              errors if errors else None,
        }
    }
    if page_groups is not None:
        # Only include groups when requested; old clients that don't ask for
        # group_by continue to get a flat list and ignore this field.
        response_payload["metadata"]["groups"] = page_groups
    return JSONResponse(content=response_payload)


# ─── Narrative cache ──────────────────────────────────────────────────────────
# Pallas LLM calls are 30-120s and the same alert often gets clicked multiple
# times during investigation. Two-layer cache:
#   • _narrative_cache (by alert_id) — re-click on the same alert
#   • _narrative_signature_cache (by signature hash) — different alert ids
#     that match the same rule/agent/source combo. In a noisy environment
#     50 alerts can all be "ET HUNTING signature 2034567 from 1.2.3.4 on
#     dev-host" and they should share a narrative instead of triggering 50
#     separate LLM calls.
# Signature hash includes rule_id, agent name, signature_id, source IP,
# severity — enough specificity to keep narratives accurate, broad enough
# to dedupe noisy repeats. NOT included: timestamp, description, observables
# (which vary between instances of the same pattern but don't change the
# fundamental "what's happening" narrative meaningfully).
_narrative_cache: dict = {}              # alert_id -> (text, fetched_at_monotonic)
_narrative_signature_cache: dict = {}    # sig_hash -> (text, fetched_at_monotonic)
_NARRATIVE_TTL: float = 3600.0  # 1 hour


def _alert_signature_key(body: dict) -> str:
    """Build a stable signature key for narrative caching across alert instances.

    Returns an empty string if we can't form a meaningful signature — caller
    should treat that as a cache miss and skip the signature cache (still
    benefits from the per-alert-id cache).
    """
    rule_id   = str(body.get("rule_id") or body.get("rule", {}).get("id") or "")
    sig_id    = str(body.get("signature_id") or body.get("alert", {}).get("signature_id") or "")
    agent     = str(body.get("agent", {}).get("name") if isinstance(body.get("agent"), dict) else body.get("agent") or "")
    src_ip    = str(body.get("src_ip") or body.get("source_ip") or "")
    severity  = str(body.get("severity") or "")
    source    = str(body.get("source") or "")
    # Need at least one identifying signal — otherwise signature is meaningless
    if not (rule_id or sig_id):
        return ""
    return f"{source}|{rule_id}|{sig_id}|{agent}|{src_ip}|{severity}"


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
    sig_key  = _alert_signature_key(body)

    # Cache lookup, in priority order:
    #   1. Per-alert-id (exact match — analyst re-clicks the same alert)
    #   2. Per-signature  (different alert id, same rule+agent+source — common
    #      in noisy environments where dozens of alerts share a pattern)
    if alert_id and alert_id in _narrative_cache:
        cached_text, fetched_at = _narrative_cache[alert_id]
        if time.monotonic() - fetched_at < _NARRATIVE_TTL:
            return JSONResponse(content={
                "alert_id": alert_id,
                "narrative": cached_text,
                "generation_time_ms": 0,
                "cached": "alert_id",
            })
    if sig_key and sig_key in _narrative_signature_cache:
        cached_text, fetched_at = _narrative_signature_cache[sig_key]
        if time.monotonic() - fetched_at < _NARRATIVE_TTL:
            # Promote into the alert-id cache so subsequent re-clicks of THIS
            # alert hit the faster path without recomputing the signature.
            if alert_id:
                _narrative_cache[alert_id] = (cached_text, fetched_at)
            return JSONResponse(content={
                "alert_id": alert_id,
                "narrative": cached_text,
                "generation_time_ms": 0,
                "cached": "signature",
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
            max_tokens=16000,
            temperature=0.3
        )

        narrative_text = response.strip()
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Persist to both caches — alert-id for repeat clicks on the same alert,
        # signature key so the next noisy alert from the same rule+source on
        # the same agent reuses the narrative without firing another LLM call.
        now_mono = time.monotonic()
        if alert_id:
            _narrative_cache[alert_id] = (narrative_text, now_mono)
        if sig_key:
            _narrative_signature_cache[sig_key] = (narrative_text, now_mono)

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
        if indicator_type == "ip" and AGELEIA_BASE_URL:
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


# ─── Bulk-action idempotency layer ────────────────────────────────────────────
# When the analyst clicks a bulk action button, the frontend generates a UUID
# and sends it as `idempotency_key`. If the same key arrives twice (network
# retry, double-click, refresh-while-in-flight), we return the cached result
# instead of re-running the action — matches firewall_audit's existing pattern
# in Agelia. Cache lives in-process for v1; promote to OpenSearch index when
# Pallas is multi-instance.
_bulk_action_log: dict = {}    # idempotency_key -> (result_dict, fetched_at_monotonic)
_BULK_ACTION_TTL: float = 86400.0  # 24h — long enough that retries always hit


def _bulk_idempotency_check(key: str):
    """Return cached result for `key` if present and not expired, else None."""
    import time as _t
    if not key:
        return None
    if key not in _bulk_action_log:
        return None
    cached, fetched_at = _bulk_action_log[key]
    if _t.monotonic() - fetched_at > _BULK_ACTION_TTL:
        del _bulk_action_log[key]
        return None
    return cached


def _bulk_idempotency_store(key: str, result: dict) -> None:
    import time as _t
    if key:
        _bulk_action_log[key] = (result, _t.monotonic())


async def _transition_jira_for_disposition(
    ticket_id: str,
    disposition: str,
    analyst_name: str,
    notes: str = "",
) -> bool:
    """Move a Jira ticket to a terminal status that matches a Pallas closure.

    Centralizes the disposition→transition mapping shared by FP dismissal,
    close-with-disposition, and bulk-close so every Pallas-side closure stops
    the Jira SLA (Time to Resolution) clock consistently — previously only the
    generic /close endpoint transitioned the ticket, so FP/bulk closures left
    the ticket "Open" forever and the SLA never completed.

    Gated on a Jira ticket existing + Jira being configured, so the no-Jira
    "Mode B" closure paths are unaffected. Non-fatal: the incident-store
    closure is already recorded by the caller, so any Jira failure here is
    logged and reported (False) rather than raised.
    """
    if not (ticket_id and jira_client_instance.is_configured):
        return False

    # False positive / benign are noise dispositions — prefer a Cancelled /
    # Won't-Do transition if the SECOPS workflow defines one, so SLA reporting
    # can tell them apart from genuine remediations. Everything else resolves
    # normally. The first transition the live workflow exposes wins.
    if disposition in ("false_positive", "benign"):
        targets = (
            "cancelled", "canceled", "cancel", "won't do", "wont do", "declined",
            "false positive", "resolved", "done", "closed", "complete",
        )
        resolution_name = jira_client_instance.resolution_cancelled
    else:
        targets = ("resolve", "resolved", "done", "closed", "complete", "cancelled", "canceled", "cancel")
        resolution_name = jira_client_instance.resolution_done

    label = disposition.replace("_", " ").title()
    comment = (
        f"h2. Incident closed via Pallas — {label}\n\n"
        f"*Analyst:* {analyst_name}\n"
        + (f"*Notes:* {notes}\n" if notes else "")
        + "\n----\n_Closed via Pallas AI Security Platform_"
    )
    try:
        result = await jira_client_instance.transition_ticket(
            ticket_id,
            targets,
            comment=comment,
            resolution_name=resolution_name,
        )
        if not result.get("transitioned"):
            logger.warning(
                "Jira transition for disposition=%s did not complete for %s: %s",
                disposition, ticket_id, result,
            )
        return bool(result.get("transitioned"))
    except Exception as exc:
        logger.warning(
            "Jira transition for disposition=%s failed (non-fatal) for %s: %s",
            disposition, ticket_id, exc,
        )
        return False


@router.post("/api/narrative/incident/bulk-acknowledge")
async def bulk_acknowledge_incidents(request: Request):
    """Acknowledge a batch of alerts in one call. Best-effort batch — partial
    success is reported per-alert and the idempotency key dedupes retries.

    Body:
      alert_ids:        [str]   — list of alert ids to acknowledge
      alerts:           {id:obj}— optional map of full alert objects keyed by id
                                  (used by incident_store.create_incident; if
                                  missing, only minimal incident metadata is
                                  recorded)
      analyst_name:     str
      idempotency_key:  str (UUID) — required for retry safety
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_ids       = body.get("alert_ids") or []
    alerts_map      = body.get("alerts") or {}
    analyst_name    = body.get("analyst_name", "SOC Analyst")
    idempotency_key = (body.get("idempotency_key") or "").strip()

    if not isinstance(alert_ids, list) or not alert_ids:
        return JSONResponse(content={"error": "alert_ids must be a non-empty list"}, status_code=400)
    if not idempotency_key:
        return JSONResponse(content={"error": "idempotency_key required"}, status_code=400)

    cached = _bulk_idempotency_check(idempotency_key)
    if cached is not None:
        return JSONResponse(content=cached)

    results = {}
    failed = []
    for aid in alert_ids:
        aid = str(aid).strip()
        if not aid:
            failed.append({"alert_id": aid, "error": "empty alert_id"})
            continue
        try:
            alert_obj = alerts_map.get(aid) or {}
            incident = await incident_store.create_incident(aid, alert_obj, analyst_name)
            incident.pop("_doc_id", None)
            results[aid] = incident
        except Exception as e:
            logger.error("Bulk ack failed for %s: %s", aid, e, exc_info=True)
            failed.append({"alert_id": aid, "error": str(e)})

    response = {
        "results":         results,
        "failed":          failed,
        "succeeded_count": len(results),
        "failed_count":    len(failed),
        "idempotency_key": idempotency_key,
    }
    _bulk_idempotency_store(idempotency_key, response)
    return JSONResponse(content=response)


@router.post("/api/narrative/incident/bulk-mute")
async def bulk_mute_incidents(request: Request):
    """Create ONE pattern-based mute rule covering all alerts in a group, then
    annotate each alert's incident record as muted. The single Agelia mute
    rule covers all matching FUTURE alerts because mute is pattern-based;
    this endpoint just records the analyst's decision against the current
    instances and returns the new mute rule + per-alert incident states.

    Body:
      alert_ids:        [str]
      conditions:       {key: value} — mute rule conditions (e.g. rule_id, src_ip)
      classification:   str ('false_positive' | 'true_positive' | 'benign')
      analyst_name:     str
      idempotency_key:  str (UUID)
      reason:           str (optional)
    """
    # RE-ENABLE NEXT RELEASE — mute returns alongside Agelia
    if not AGELEIA_BASE_URL:
        return JSONResponse(
            content={"error": "Mute is not available in this release",
                     "reason": "deferred_to_next_release"},
            status_code=501,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_ids       = body.get("alert_ids") or []
    conditions      = body.get("conditions") or {}
    classification  = (body.get("classification") or "false_positive").strip().lower()
    analyst_name    = body.get("analyst_name", "SOC Analyst")
    idempotency_key = (body.get("idempotency_key") or "").strip()
    reason          = (body.get("reason") or f"Bulk muted from Pallas: {classification}").strip()

    if not isinstance(alert_ids, list) or not alert_ids:
        return JSONResponse(content={"error": "alert_ids must be a non-empty list"}, status_code=400)
    if not isinstance(conditions, dict) or not conditions:
        return JSONResponse(content={"error": "conditions must be a non-empty object"}, status_code=400)
    if not idempotency_key:
        return JSONResponse(content={"error": "idempotency_key required"}, status_code=400)

    cached = _bulk_idempotency_check(idempotency_key)
    if cached is not None:
        return JSONResponse(content=cached)

    # 1. Create the single mute rule via Agelia. Reuses the existing
    #    /api/mute/rules endpoint that the per-alert mute already calls.
    mute_result, mute_status = await _call_agelia("POST", "/api/mute/rules", request, {
        "conditions":     conditions,
        "classification": classification,
        "reason":         reason,
        "analyst":        analyst_name,
    })
    if mute_status not in (200, 201) or not mute_result.get("success"):
        # Don't store the failed result in idempotency cache — let the
        # analyst retry the action with the same key once Agelia is back.
        return JSONResponse(content={"error": mute_result.get("error", "Mute rule create failed")},
                            status_code=mute_status or 502)
    _invalidate_mute_summary_cache()

    # 2. Annotate each alert's incident record as muted by this analyst.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    results = {}
    failed = []
    for aid in alert_ids:
        aid = str(aid).strip()
        if not aid:
            continue
        try:
            await incident_store.update_incident(
                aid,
                {
                    "agelia_muted":              True,
                    "agelia_muted_at":           now,
                    "agelia_muted_by":           analyst_name,
                    "agelia_muted_classification": classification,
                    "agelia_muted_conditions":   conditions,
                },
                action_log={
                    "action":    "bulk_muted",
                    "timestamp": now,
                    "actor":     analyst_name,
                    "details":   f"Bulk-muted with conditions {conditions} (classification={classification})",
                },
            )
            updated = await incident_store.get_incident(aid)
            if updated:
                updated.pop("_doc_id", None)
                results[aid] = updated
            else:
                results[aid] = {"agelia_muted": True}
        except Exception as e:
            logger.error("Bulk mute annotate failed for %s: %s", aid, e, exc_info=True)
            failed.append({"alert_id": aid, "error": str(e)})

    response = {
        "mute_rule":       mute_result.get("data") or mute_result,
        "results":         results,
        "failed":          failed,
        "succeeded_count": len(results),
        "failed_count":    len(failed),
        "idempotency_key": idempotency_key,
    }
    _bulk_idempotency_store(idempotency_key, response)
    return JSONResponse(content=response)


@router.post("/api/narrative/incident/bulk-close")
async def bulk_close_incidents(request: Request):
    """Close a batch of alerts with one disposition. Loops the existing
    close-with-disposition logic per alert.

    Body:
      alert_ids:        [str]
      disposition:      str ('resolved' | 'false_positive' | 'benign' | 'other')
      notes:            str (optional; required when disposition='other')
      analyst_name:     str
      idempotency_key:  str (UUID)
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_ids       = body.get("alert_ids") or []
    disposition     = (body.get("disposition") or "").strip().lower()
    notes           = (body.get("notes") or "").strip()
    analyst_name    = body.get("analyst_name", "SOC Analyst")
    idempotency_key = (body.get("idempotency_key") or "").strip()

    valid_dispositions = {"resolved", "false_positive", "benign", "other"}
    if not isinstance(alert_ids, list) or not alert_ids:
        return JSONResponse(content={"error": "alert_ids must be a non-empty list"}, status_code=400)
    if disposition not in valid_dispositions:
        return JSONResponse(
            content={"error": f"Invalid disposition. Must be one of: {sorted(valid_dispositions)}"},
            status_code=400,
        )
    if disposition == "other" and not notes:
        return JSONResponse(content={"error": "Notes required when disposition is 'other'"}, status_code=400)
    if not idempotency_key:
        return JSONResponse(content={"error": "idempotency_key required"}, status_code=400)

    cached = _bulk_idempotency_check(idempotency_key)
    if cached is not None:
        return JSONResponse(content=cached)

    from datetime import datetime, timezone
    label = disposition.replace("_", " ").title()
    resolution_notes = f"{label}" + (f" — {notes}" if notes else "")
    results = {}
    failed = []
    for aid in alert_ids:
        aid = str(aid).strip()
        if not aid:
            continue
        try:
            existing = await incident_store.get_incident(aid)
            if not existing:
                failed.append({"alert_id": aid, "error": "Incident not found"})
                continue
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
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
            await incident_store.update_incident(
                aid,
                {
                    "status":           "closed",
                    "closed_at":        now_iso,
                    "closed_by":        analyst_name,
                    "disposition":      disposition,
                    "resolution_notes": resolution_notes,
                    "mttr_seconds":     mttr_seconds,
                },
                action_log={
                    "action":    "bulk_closed_with_disposition",
                    "timestamp": now_iso,
                    "actor":     analyst_name,
                    "details":   f"Bulk-closed disposition={label}" + (f" notes={notes}" if notes else ""),
                },
            )
            jira_transitioned = await _transition_jira_for_disposition(
                existing.get("jira_ticket_id"), disposition, analyst_name, notes,
            )
            results[aid] = {
                "incident_id":      existing.get("incident_id"),
                "status":           "closed",
                "disposition":      disposition,
                "closed_at":        now_iso,
                "mttr_seconds":     mttr_seconds,
                "jira_transitioned": jira_transitioned,
            }
        except Exception as e:
            logger.error("Bulk close failed for %s: %s", aid, e, exc_info=True)
            failed.append({"alert_id": aid, "error": str(e)})

    response = {
        "results":         results,
        "failed":          failed,
        "succeeded_count": len(results),
        "failed_count":    len(failed),
        "disposition":     disposition,
        "idempotency_key": idempotency_key,
    }
    _bulk_idempotency_store(idempotency_key, response)
    return JSONResponse(content=response)


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
    """Block an attacker IP via either Agelia (preferred) or direct AWS WAF.

    Two paths, picked at request time:
      1) Agelia (AGELEIA_BASE_URL set) — JWT-audited single control plane,
         blocks both WAF + Network Firewall in one call. Preferred when
         available because the analyst's Keycloak identity is enforced
         server-side and all block actions roll up to one audit log.
      2) Direct AWS WAF + Network Firewall (this release) — when Agelia isn't
         deployed, fall back to direct boto3 calls against the env-configured
         WAF IP set and Network Firewall rule group. Analyst identity comes
         from the request payload; less centralized audit but functional.

    When Agelia is later restored, AGELEIA_BASE_URL flips on and the Agelia
    path takes precedence automatically.
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

    # Branch on Agelia availability.
    if not agelia_blocking_configured and direct_block_configured:
        # ── Direct path: hit AWS WAF + Network Firewall via boto3 ──
        # Each call returns {"success": bool, "error": str|None, "ip": str, ...}.
        # We attempt both layers and report which actually succeeded so the
        # incident audit captures the truth (e.g., WAF blocked but Firewall
        # IAM permissions are missing). Block is considered successful if at
        # least ONE layer accepted the IP.
        layers_blocked = []
        waf_error = None
        fw_error = None
        if aws_waf_client.is_configured:
            waf_result = aws_waf_client.block_ip(ip)
            if waf_result.get("error"):
                waf_error = waf_result["error"]
            else:
                layers_blocked.append("waf")
        if aws_firewall_client.is_configured:
            fw_result = aws_firewall_client.block_ip(ip)
            if fw_result.get("error"):
                fw_error = fw_result["error"]
            else:
                layers_blocked.append("network_firewall")
        result = {
            "success": bool(layers_blocked),
            "data": {
                "ip": ip,
                "layers_blocked": layers_blocked,
                "waf_error": waf_error,
                "firewall_error": fw_error,
            },
            "error": None if layers_blocked else (waf_error or fw_error or "No block layer configured"),
        }
        status = 200 if layers_blocked else 502
    else:
        # ── Agelia path (preferred when available) ──
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
        jira_comment = f"IP block executed by {analyst_name}: {ip} blocked via Agelia ({layers})"
        if user_reason:
            jira_comment = f"{jira_comment}\n\nReason: {user_reason}"
        await jira_client_instance.add_comment(
            existing["jira_ticket_id"],
            jira_comment,
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
    # RE-ENABLE NEXT RELEASE — mute returns alongside Agelia
    if not AGELEIA_BASE_URL:
        return JSONResponse(
            content={"error": "Mute is not available in this release",
                     "reason": "deferred_to_next_release"},
            status_code=501,
        )
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
    close_result = None
    jira_ticket_id = existing.get("jira_ticket_id")
    if jira_ticket_id and jira_client_instance.is_configured:
        close_result = await jira_client_instance.close_ticket(jira_ticket_id, resolution_summary)
        jira_closed = close_result.get("closed", False)
        if not jira_closed:
            error = close_result.get("error", "Unknown Jira close failure")
            logger.warning(
                "Refusing to close local incident %s because Jira ticket %s did not transition: %s",
                alert_id,
                jira_ticket_id,
                error,
            )
            return JSONResponse(
                content={
                    "error": f"Jira close failed for {jira_ticket_id}: {error}",
                    "incident_id": existing.get("incident_id"),
                    "status": existing.get("status"),
                    "jira_ticket_id": jira_ticket_id,
                    "jira_closed": False,
                    "jira_close_result": close_result,
                },
                status_code=502,
            )

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
        "jira_close_result": close_result,
    })


@router.post("/api/narrative/incident/integration-status")
async def get_integration_status(request: Request):
    """Return which integrations are configured (for frontend button visibility).

    Reports both block paths so the frontend can show the Block button when
    EITHER Agelia is configured OR the direct AWS WAF/Firewall env vars are
    set. The frontend's compatibility shim reads
        blockingOk = data.agelia_blocking ?? (data.waf || data.firewall)
    so returning waf/firewall here keeps Block usable in this release while
    Agelia is being added. When Agelia ships, agelia_blocking=true takes
    precedence and the same code path continues to work.
    """
    return JSONResponse(content={
        "jira": jira_client_instance.is_configured,
        "agelia_blocking": agelia_blocking_configured,
        # Direct AWS clients (fallback path while Agelia is offline).
        "waf":      aws_waf_client.is_configured,
        "firewall": aws_firewall_client.is_configured,
        "incident_store": True,  # Always available (uses existing OpenSearch)
    })


@router.post("/api/narrative/incident/close-with-disposition")
async def close_with_disposition(request: Request):
    """Close an incident with an explicit categorical disposition + analyst notes.

    Used by Pallas Mode B (no-Jira deployments) where the analyst is the only
    party who closes incidents. The disposition becomes part of the audit
    record so SOC reporting can split closures by category (Resolved /
    False Positive / Benign / Other).

    For tenants that have Jira, the standard /close endpoint is preferred —
    closure is normally engineer-driven via Jira webhook (Phase 2).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    alert_id    = body.get("alert_id", "").strip()
    disposition = (body.get("disposition") or "").strip().lower()
    notes       = (body.get("notes") or "").strip()
    analyst     = body.get("analyst_name", "SOC Analyst")

    valid_dispositions = {"resolved", "false_positive", "benign", "other"}
    if not alert_id:
        return JSONResponse(content={"error": "Missing alert_id"}, status_code=400)
    if disposition not in valid_dispositions:
        return JSONResponse(
            content={"error": f"Invalid disposition. Must be one of: {sorted(valid_dispositions)}"},
            status_code=400,
        )
    if disposition == "other" and not notes:
        return JSONResponse(content={"error": "Notes required when disposition is 'other'"}, status_code=400)

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

    label = disposition.replace("_", " ").title()  # "false_positive" -> "False Positive"
    resolution_notes = f"{label}" + (f" — {notes}" if notes else "")

    await incident_store.update_incident(
        alert_id,
        {
            "status": "closed",
            "closed_at": now_iso,
            "closed_by": analyst,
            "disposition": disposition,
            "resolution_notes": resolution_notes,
            "mttr_seconds": mttr_seconds,
        },
        action_log={
            "action": "closed_with_disposition",
            "timestamp": now_iso,
            "actor": analyst,
            "details": f"Disposition: {label}" + (f" | Notes: {notes}" if notes else ""),
        },
    )
    jira_transitioned = await _transition_jira_for_disposition(
        existing.get("jira_ticket_id"), disposition, analyst, notes,
    )
    return JSONResponse(content={
        "incident_id": existing.get("incident_id"),
        "status": "closed",
        "disposition": disposition,
        "closed_at": now_iso,
        "mttr_seconds": mttr_seconds,
        "mttr_minutes": int(mttr_seconds / 60) if mttr_seconds else None,
        "jira_transitioned": jira_transitioned,
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

    # FP dismissal is a STATUS-ONLY operation in this release.
    #
    # Prior behavior delegated to Agelia (created a mute rule that suppressed
    # future identical signatures, then flipped status). With Agelia disabled
    # this release, that path 503'd and the analyst couldn't dismiss FPs at
    # all. FP is conceptually about THIS alert ("not a real threat") — the
    # broader "stop similar alerts from firing again" workflow is Mute, which
    # is intentionally deferred until Agelia ships.
    #
    # Future: when Agelia is restored, the Mute UI returns and the FP dropdown
    # can re-add a "Mute this signature" sibling option (see plugin's
    # attack-narrative.tsx:2015-2028 for the unified pattern). Until then,
    # FP just records the disposition + closes the incident.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # Compute MTTR from acknowledgment time to now (incident is terminating).
    mttr_seconds = None
    ack_at = existing.get("acknowledged_at")
    if ack_at:
        try:
            ack_dt = datetime.fromisoformat(ack_at.replace("Z", "+00:00") if ack_at.endswith("Z") else ack_at)
            if ack_dt.tzinfo is None:
                ack_dt = ack_dt.replace(tzinfo=timezone.utc)
            mttr_seconds = (datetime.now(timezone.utc) - ack_dt).total_seconds()
        except Exception:
            pass

    updates = {
        "status": "false_positive",
        "fp_reason": reason or "No reason provided",
        "fp_dismissed_by": analyst_name,
        "fp_dismissed_at": now,
    }
    if mttr_seconds is not None:
        updates["mttr_seconds"] = mttr_seconds

    await incident_store.update_incident(
        alert_id,
        updates,
        action_log={
            "action": "false_positive",
            "timestamp": now,
            "actor": analyst_name,
            "details": f"Dismissed as false positive by {analyst_name}: {reason}" if reason else f"Dismissed as false positive by {analyst_name}",
        },
    )

    # Transition the Jira ticket to a terminal state so its SLA (Time to
    # Resolution) stops and the ticket doesn't linger as "Open" — previously
    # this only posted a comment, leaving the ticket open forever and breaking
    # the MTTR/SLA reporting.
    jira_transitioned = await _transition_jira_for_disposition(
        existing.get("jira_ticket_id"), "false_positive", analyst_name,
        reason or "No reason provided",
    )

    return JSONResponse(content={
        "dismissed": True,
        "status": "false_positive",
        "reason": reason,
        "jira_transitioned": jira_transitioned,
    })


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


# E3.3b hotfix (2026-06-12): the from-imports for health_meta /
# mcp_transport / api_data got swept up by the E3.3b cluster cut.
# Restoring here so the include_router calls below have their
# module references in scope.
from .routes import health_meta as _route_health_meta
from .routes import mcp_transport as _route_mcp_transport
from .routes import api_data as _route_api_data

app.include_router(router)
# E3.1: mount the extracted health_meta routes under the SAME prefix
# (/ai-analyst) so /ai-analyst/metrics and /ai-analyst/auth/token
# continue to resolve at their legacy URLs.
app.include_router(_route_health_meta.router, prefix=API_PREFIX)
app.include_router(_route_mcp_transport.mcp_router, prefix=API_PREFIX)
app.include_router(_route_api_data.api_data_router, prefix=API_PREFIX)


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
            route_state.set_oauth_manager(_oauth_manager)  # E3.0
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

    # ========================================================================
    # PALLAS_P4_OS_MEMORY (Track A1) - upgrade ConversationMemory to use
    # OpenSearch persistence once the indexer client is ready. If the indexer
    # is not configured, the module-level fallback in _MemoryBackend stays.
    # ========================================================================
    try:
        global _conversation_memory
        _indexer = getattr(wazuh_client, "_indexer_client", None)
        if _indexer is not None:
            _conversation_memory = ConversationMemory(
                max_turns=10, ttl_minutes=30, indexer_client=_indexer
            )
            route_state.set_memory(_conversation_memory)  # E3.0 - upgrade visible to routes
            logger.info(
                f"[MEMORY] ConversationMemory upgraded to backend="
                f"{_conversation_memory.backend.name()}"
            )
        else:
            logger.info(
                "[MEMORY] ConversationMemory remaining on in-memory backend "
                "(WAZUH_INDEXER not configured)"
            )
    except Exception as e:
        logger.warning(f"[MEMORY] Could not upgrade ConversationMemory backend: {e}")

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
            # Phase 5.3: gate by config intent, not by client init success.
            # Routing patterns register based on SURICATA_ENABLED env var;
            # transient indexer outages no longer permanently disable NIDS
            # routing for the container lifetime.
            suricata_enabled=getattr(wazuh_config, "suricata_enabled", False),
        )
        route_state.set_orchestrator(_query_orchestrator)  # E3.0
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
