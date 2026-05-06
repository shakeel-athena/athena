"""
Pallas Jira Client - Create, update, and close incident tickets.
Uses httpx for async REST API calls. Supports Jira Cloud and Server.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("pallas.jira")

# Atlassian OAuth 2.0 endpoints
ATLASSIAN_AUTH_URL = "https://auth.atlassian.com/oauth/token"
ATLASSIAN_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"
ATLASSIAN_API_BASE = "https://api.atlassian.com/ex/jira"


class JiraClient:
    """Async Jira REST API client for incident ticket management.

    Supports three auth modes (auto-detected from env vars):
    - OAuth 2.0 (recommended): Set JIRA_CLIENT_ID + JIRA_CLIENT_SECRET
      Uses client_credentials grant for machine-to-machine auth with Jira Cloud.
    - Bearer auth: Set JIRA_BASE_URL + JIRA_SERVICE_TOKEN
      For Jira Cloud scoped service tokens or Server/DC Personal Access Tokens (PATs).
    - Basic auth (fallback): Set JIRA_BASE_URL + JIRA_USER_EMAIL + JIRA_API_TOKEN
      For Jira Cloud classic (non-scoped) tokens or personal API tokens.
    """

    def __init__(self):
        self.base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
        self.service_token = os.getenv("JIRA_SERVICE_TOKEN", "")
        self.user_email = os.getenv("JIRA_USER_EMAIL", "")
        self.api_token = os.getenv("JIRA_API_TOKEN", "")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "SECOPS")
        self.issue_type = os.getenv("JIRA_ISSUE_TYPE", "Security Alert")
        # JSM Request Type ID — binds the ticket to a JSM request-type queue
        # so the sidebar shows it correctly. Get this ID from JSM:
        # Project settings → Request types → click the type → ID in URL/page.
        # If unset, tickets create successfully but show "Request Type: None".
        self.request_type_id = os.getenv("JIRA_REQUEST_TYPE_ID", "").strip()
        # Custom field name varies per Jira instance. Default 'customfield_10010'
        # is the most common (JSM Cloud default). Override via env if your
        # instance uses a different field id.
        self.request_type_field = os.getenv("JIRA_REQUEST_TYPE_FIELD", "customfield_10010").strip()

        # OAuth 2.0 credentials
        self.client_id = os.getenv("JIRA_CLIENT_ID", "")
        self.client_secret = os.getenv("JIRA_CLIENT_SECRET", "")
        self.cloud_id = os.getenv("JIRA_CLOUD_ID", "")  # Optional — auto-discovered if not set

        # OAuth token state
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        # Determine auth mode: OAuth > Bearer > Basic
        if self.client_id and self.client_secret:
            self._auth_mode = "oauth"
            self._configured = True
        elif self.base_url and self.service_token:
            self._auth_mode = "bearer"
            self._configured = True
        elif self.base_url and self.user_email and self.api_token:
            self._auth_mode = "basic"
            self._configured = True
        else:
            self._auth_mode = None
            self._configured = False

        if self._configured:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self._auth_mode == "oauth":
                # OAuth client — base_url set after cloud_id discovery, auth header set per-request
                self.client = httpx.AsyncClient(
                    headers=headers,
                    timeout=20.0,
                )
                logger.info(f"Jira client configured (OAuth 2.0): cloud_id={'set' if self.cloud_id else 'auto-discover'} (project: {self.project_key})")
            elif self._auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.service_token}"
                self.client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=20.0,
                )
                logger.info(f"Jira client configured (Bearer/service token): {self.base_url} (project: {self.project_key})")
            else:
                self.client = httpx.AsyncClient(
                    base_url=self.base_url,
                    auth=(self.user_email, self.api_token),
                    headers=headers,
                    timeout=20.0,
                )
                logger.info(f"Jira client configured (Basic auth): {self.base_url} (project: {self.project_key})")
        else:
            self.client = None
            logger.info("Jira not configured - set JIRA_CLIENT_ID + JIRA_CLIENT_SECRET (OAuth), or JIRA_BASE_URL + JIRA_SERVICE_TOKEN (Bearer), or JIRA_BASE_URL + JIRA_USER_EMAIL + JIRA_API_TOKEN (Basic)")

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def _ensure_oauth_token(self) -> bool:
        """Ensure we have a valid OAuth access token. Returns True if ready."""
        if self._auth_mode != "oauth":
            return True

        # Check if token is still valid (refresh 60s before expiry)
        if self._access_token and time.time() < (self._token_expires_at - 60):
            return True

        # Request new access token via client_credentials grant
        try:
            async with httpx.AsyncClient(timeout=15.0) as auth_client:
                resp = await auth_client.post(
                    ATLASSIAN_AUTH_URL,
                    json={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                )
                if resp.status_code != 200:
                    logger.error(f"OAuth token request failed: {resp.status_code} {resp.text[:300]}")
                    return False

                data = resp.json()
                self._access_token = data["access_token"]
                self._token_expires_at = time.time() + data.get("expires_in", 3600)
                logger.info(f"OAuth access token obtained (expires in {data.get('expires_in', 3600)}s)")

                # Discover cloud_id if not set
                if not self.cloud_id:
                    resources_resp = await auth_client.get(
                        ATLASSIAN_RESOURCES_URL,
                        headers={"Authorization": f"Bearer {self._access_token}"},
                    )
                    if resources_resp.status_code == 200:
                        sites = resources_resp.json()
                        if sites:
                            self.cloud_id = sites[0]["id"]
                            logger.info(f"Auto-discovered Jira cloud_id: {self.cloud_id} ({sites[0].get('name', 'unknown')})")
                        else:
                            logger.error("No accessible Jira sites found for this OAuth app")
                            return False
                    else:
                        logger.error(f"Failed to discover cloud_id: {resources_resp.status_code}")
                        return False

                return True

        except Exception as e:
            logger.error(f"OAuth token exchange failed: {e}")
            return False

    def _api_url(self, path: str) -> str:
        """Build the full API URL based on auth mode."""
        if self._auth_mode == "oauth":
            return f"{ATLASSIAN_API_BASE}/{self.cloud_id}{path}"
        return path  # For Bearer/Basic, httpx base_url handles it

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an authenticated API request with OAuth token refresh."""
        if self._auth_mode == "oauth":
            if not await self._ensure_oauth_token():
                raise Exception("Failed to obtain Jira OAuth token")
            url = self._api_url(path)
            kwargs.setdefault("headers", {})
            kwargs["headers"]["Authorization"] = f"Bearer {self._access_token}"
            kwargs["headers"]["Content-Type"] = "application/json"
            kwargs["headers"]["Accept"] = "application/json"
            if method == "GET":
                return await self.client.get(url, **kwargs)
            elif method == "POST":
                return await self.client.post(url, **kwargs)
            elif method == "PUT":
                return await self.client.put(url, **kwargs)
        else:
            # Bearer/Basic — use httpx client with pre-configured auth
            if method == "GET":
                return await self.client.get(path, **kwargs)
            elif method == "POST":
                return await self.client.post(path, **kwargs)
            elif method == "PUT":
                return await self.client.put(path, **kwargs)

    def _format_ticket_body(self, alert: Dict[str, Any], narrative: Optional[str] = None) -> str:
        """Format incident details as Jira ticket description (ADF or wiki markup)."""
        severity = alert.get("severity", "unknown").upper()
        rule_id = alert.get("rule_id", "N/A")
        rule_desc = alert.get("rule_description", "N/A")
        source = alert.get("source", "unknown")
        src_ip = alert.get("source_ip", "N/A")
        dst_ip = alert.get("dest_ip", "N/A")
        agent_name = alert.get("agent_name", "N/A")
        agent_id = alert.get("agent_id", "N/A")
        timestamp = alert.get("timestamp", "N/A")
        category = alert.get("category", "N/A")

        mitre_tactics = ", ".join(alert.get("mitre_tactics", [])) or "None"
        mitre_techniques = ", ".join(alert.get("mitre_techniques", [])) or "None"

        # Suricata-specific
        http_xff = alert.get("http_xff", "")
        http_url = alert.get("http_url", "")

        # Wazuh-specific
        decoder = alert.get("decoder_name", "")
        location = alert.get("location", "")
        protocol = alert.get("protocol", "")
        action = alert.get("action", "")

        body = f"""h2. Incident Details

||Field||Value||
|Severity|{severity}|
|Source|{source.upper()}|
|Rule|{rule_id} -- {rule_desc}|
|Timestamp|{timestamp}|
|Category|{category}|

h2. Network Flow

||Field||Value||
|Source IP|{src_ip}|
|Destination IP|{dst_ip}|"""

        if protocol:
            body += f"\n|Protocol|{protocol}|"
        if action:
            body += f"\n|Action|{action}|"
        if http_xff:
            body += f"\n|X-Forwarded-For (Real Attacker IP)|{http_xff}|"
        if http_url:
            body += f"\n|Target URL|{http_url}|"

        body += f"""

h2. Affected Host

||Field||Value||
|Agent|{agent_name} (ID: {agent_id})|"""

        if decoder:
            body += f"\n|Decoder|{decoder}|"
        if location:
            body += f"\n|Log Source|{location}|"

        body += f"""

h2. MITRE ATT&CK

||Field||Value||
|Tactics|{mitre_tactics}|
|Techniques|{mitre_techniques}|"""

        if narrative:
            body += f"""

h2. AI Forensic Narrative

{narrative}"""

        body += """

h2. Response Actions

This section will be updated as actions are taken:
* _Pending: Observable enrichment (AbuseIPDB + Wazuh history)_
* _Pending: WAF IP block (if public IP detected)_
* _Pending: Network Firewall block (if public IP detected)_

h2. Workflow

OPEN → WORK IN PROGRESS → RESOLVED → CLOSED

----
_Generated by Pallas AI Security Platform_"""

        return body

    async def create_ticket(self, alert: Dict[str, Any], narrative: Optional[str] = None, incident_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a Jira incident ticket."""
        if not self._configured:
            return {"error": "Jira not configured. Set JIRA_CLIENT_ID + JIRA_CLIENT_SECRET (OAuth), or JIRA_BASE_URL + JIRA_SERVICE_TOKEN (Bearer), or JIRA_BASE_URL + JIRA_USER_EMAIL + JIRA_API_TOKEN (Basic)."}

        severity = alert.get("severity", "medium").upper()
        rule_id = alert.get("rule_id", "N/A")
        rule_desc = alert.get("rule_description", "Security Alert")
        source = alert.get("source", "unknown").upper()
        src_ip = alert.get("source_ip", "")

        summary = f"[{severity}] [{source}] {rule_desc}"
        if src_ip:
            summary += f" (from {src_ip})"
        # Jira summary max 255 chars
        summary = summary[:255]

        description = self._format_ticket_body(alert, narrative)

        # Priority mapping
        priority_map = {
            "CRITICAL": "Highest",
            "HIGH": "High",
            "MEDIUM": "Medium",
            "LOW": "Low",
        }
        priority_name = priority_map.get(severity, "Medium")

        payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": self.issue_type},
                "priority": {"name": priority_name},
            }
        }

        # JSM Request Type binding — without this the sidebar shows
        # "Request Type: None" and the ticket isn't queued under any JSM type.
        # Skip silently if not configured (ticket still creates).
        if self.request_type_id:
            payload["fields"][self.request_type_field] = self.request_type_id

        # Add labels
        payload["fields"]["labels"] = [
            "pallas-ai",
            f"severity-{severity.lower()}",
            f"source-{alert.get('source', 'unknown')}",
        ]
        if incident_id:
            payload["fields"]["labels"].append(f"incident-{incident_id[:8]}")

        # ── JSM required custom fields for "Security Alert" Request Type ──
        # When the ticket binds to a Request Type (via customfield_10010), the
        # JSM form enforces these fields as required. Each field is populated
        # from alert data with sensible defaults; option-typed fields are
        # env-overridable so the operator can adjust without code changes if
        # Jira's option labels differ.
        alert_source = (alert.get("source") or "unknown").lower()

        # 10593 — Affected Asset(s) (string)
        if alert_source == "wazuh":
            asset_value = alert.get("agent_name") or alert.get("agent_id") or "unknown"
        elif alert_source == "suricata":
            asset_value = (
                alert.get("dest_ip")
                or alert.get("destination_ip")
                or alert.get("src_ip")
                or alert.get("source_ip")
                or "unknown"
            )
        else:
            asset_value = alert.get("agent_name") or alert.get("dest_ip") or "unknown"
        payload["fields"]["customfield_10593"] = str(asset_value)[:255]

        # 10594 — Asset Type (option) — env-overridable per source
        asset_type = (
            os.getenv("JIRA_ASSET_TYPE_WAZUH", "Endpoint")
            if alert_source == "wazuh"
            else os.getenv("JIRA_ASSET_TYPE_SURICATA", "Network")
        )
        payload["fields"]["customfield_10594"] = {"value": asset_type}

        # 10405 — Severity (option) — map UPPERCASE severity to title case
        severity_label_map = {
            "CRITICAL": "Critical",
            "HIGH": "High",
            "MEDIUM": "Medium",
            "LOW": "Low",
        }
        payload["fields"]["customfield_10405"] = {
            "value": severity_label_map.get(severity, "Medium")
        }

        # 10418 — Incident Time (Eastern Time - ET) (datetime). Pass alert
        # timestamp through; Jira interprets and renders as ET.
        incident_time = (
            alert.get("timestamp")
            or alert.get("@timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        payload["fields"]["customfield_10418"] = incident_time

        # 10419 — Type of Incident (string) — derive from rule context
        rule_groups = alert.get("rule_groups") or []
        incident_type = (
            alert.get("category")
            or (rule_groups[0] if rule_groups else None)
            or rule_desc
            or "Security Alert"
        )
        payload["fields"]["customfield_10419"] = str(incident_type)[:255]

        # 10491 — Affected Environment(s) (array of options) — env-overridable
        affected_env = os.getenv("JIRA_AFFECTED_ENV", "Production")
        payload["fields"]["customfield_10491"] = [{"value": affected_env}]

        try:
            resp = await self._request("POST", "/rest/api/2/issue", json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                ticket_key = data.get("key", "")
                # Build browse URL
                if self._auth_mode == "oauth":
                    browse_url = f"{self.base_url}/browse/{ticket_key}" if self.base_url else f"https://athenasoftwaregrp.atlassian.net/browse/{ticket_key}"
                else:
                    browse_url = f"{self.base_url}/browse/{ticket_key}"
                logger.info(f"JSM {self.issue_type} created: {ticket_key} (project: {self.project_key})")
                return {
                    "jira_ticket_id": ticket_key,
                    "jira_ticket_url": browse_url,
                    "jira_self": data.get("self", ""),
                }
            else:
                error_msg = resp.text[:500]
                logger.error(f"Jira create failed: {resp.status_code} {error_msg}")
                return {"error": f"Jira API error ({resp.status_code}): {error_msg}"}
        except Exception as e:
            logger.error(f"Jira create exception: {e}")
            return {"error": f"Jira connection failed: {str(e)}"}

    async def close_ticket(self, ticket_id: str, resolution_summary: str) -> Dict[str, Any]:
        """Close a Jira ticket with resolution comment and transition to Resolved/Closed."""
        if not self._configured:
            return {"error": "Jira not configured."}

        # Add resolution comment first
        comment_result = await self.add_comment(ticket_id, resolution_summary)
        if "error" in comment_result:
            return comment_result

        # Get available transitions
        try:
            resp = await self._request("GET", f"/rest/api/2/issue/{ticket_id}/transitions")
            if resp.status_code != 200:
                return {"error": f"Failed to get transitions: {resp.status_code}"}

            transitions = resp.json().get("transitions", [])

            # JSM workflow priority: Resolved first, then Closed, then others
            # Matches SECOPS workflow: OPEN → WORK IN PROGRESS → RESOLVED → CLOSED
            close_priority = ("resolved", "closed", "done", "close", "complete", "monitoring")
            close_transition = None
            for target_name in close_priority:
                for t in transitions:
                    if t.get("name", "").lower() == target_name:
                        close_transition = t
                        break
                if close_transition:
                    break

            if not close_transition:
                # Use the last transition as fallback
                available = [t.get("name", "?") for t in transitions]
                if transitions:
                    close_transition = transitions[-1]
                    logger.warning(f"No standard close transition found for {ticket_id}. Available: {available}. Using fallback: {close_transition['name']}")
                else:
                    return {"error": "No transitions available for this ticket."}

            # Execute transition
            resp = await self._request(
                "POST",
                f"/rest/api/2/issue/{ticket_id}/transitions",
                json={"transition": {"id": close_transition["id"]}},
            )
            if resp.status_code in (200, 204):
                logger.info(f"JSM ticket resolved: {ticket_id} (transition: {close_transition['name']})")
                return {"closed": True, "transition": close_transition["name"]}
            else:
                return {"error": f"Transition failed: {resp.status_code} {resp.text[:200]}"}

        except Exception as e:
            logger.error(f"Jira close exception: {e}")
            return {"error": f"Jira close failed: {str(e)}"}

    async def add_comment(self, ticket_id: str, comment_body: str) -> Dict[str, Any]:
        """Add a comment to a Jira ticket."""
        if not self._configured:
            return {"error": "Jira not configured."}

        try:
            resp = await self._request(
                "POST",
                f"/rest/api/2/issue/{ticket_id}/comment",
                json={"body": comment_body},
            )
            if resp.status_code in (200, 201):
                return {"success": True}
            else:
                return {"error": f"Comment failed: {resp.status_code}"}
        except Exception as e:
            return {"error": f"Comment failed: {str(e)}"}

    async def get_assignable_users(self, project_key: str = None) -> list:
        """Fetch users assignable to tickets in the project."""
        if not self._configured:
            return []

        project = project_key or self.project_key
        try:
            resp = await self._request(
                "GET",
                f"/rest/api/2/user/assignable/search?project={project}&maxResults=100",
            )
            if resp.status_code == 200:
                users = resp.json()
                return [
                    {
                        "accountId": u.get("accountId", ""),
                        "displayName": u.get("displayName", ""),
                        "emailAddress": u.get("emailAddress", ""),
                        "avatarUrl": u.get("avatarUrls", {}).get("24x24", ""),
                    }
                    for u in users if u.get("active", True)
                ]
            else:
                logger.error(f"Jira user search failed: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Jira user search exception: {e}")
            return []

    async def assign_ticket(self, ticket_id: str, account_id: str) -> Dict[str, Any]:
        """Assign a Jira ticket to a user by accountId."""
        if not self._configured:
            return {"error": "Jira not configured."}

        try:
            resp = await self._request(
                "PUT",
                f"/rest/api/2/issue/{ticket_id}/assignee",
                json={"accountId": account_id},
            )
            if resp.status_code == 204:
                logger.info(f"Jira ticket assigned: {ticket_id} → {account_id}")
                return {"assigned": True}
            else:
                error_msg = resp.text[:200]
                logger.error(f"Jira assign failed: {resp.status_code} {error_msg}")
                return {"error": f"Assignment failed: {resp.status_code} {error_msg}"}
        except Exception as e:
            logger.error(f"Jira assign exception: {e}")
            return {"error": f"Assignment failed: {str(e)}"}

