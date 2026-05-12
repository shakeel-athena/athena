"""
Pallas Incident Store - OpenSearch CRUD for pallas-incidents index.
Manages incident lifecycle: acknowledge, ticket, block, close.
"""

import os
import ssl
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger("pallas.incident_store")


class IncidentStore:
    """OpenSearch-backed incident state store for SOC workflow."""

    INDEX_NAME = "pallas-incidents"

    INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "incident_id": {"type": "keyword"},
                "alert_id": {"type": "keyword"},
                "alert_source": {"type": "keyword"},
                "rule_id": {"type": "keyword"},
                "rule_description": {"type": "text"},
                "severity": {"type": "keyword"},
                "severity_num": {"type": "integer"},
                "source_ip": {"type": "ip", "ignore_malformed": True},
                "dest_ip": {"type": "ip", "ignore_malformed": True},
                "agent_id": {"type": "keyword"},
                "agent_name": {"type": "keyword"},
                "alert_timestamp": {"type": "date"},
                "status": {"type": "keyword"},
                "acknowledged_at": {"type": "date"},
                "acknowledged_by": {"type": "keyword"},
                "jira_ticket_id": {"type": "keyword"},
                "jira_ticket_url": {"type": "text"},
                "jira_created_at": {"type": "date"},
                "observability_completed": {"type": "boolean"},
                "observability_verdict": {"type": "keyword"},
                "observability_completed_at": {"type": "date"},
                "waf_blocked": {"type": "boolean"},
                "waf_blocked_at": {"type": "date"},
                "waf_blocked_ip": {"type": "ip", "ignore_malformed": True},
                "waf_error": {"type": "text"},
                "firewall_blocked": {"type": "boolean"},
                "firewall_blocked_at": {"type": "date"},
                "firewall_blocked_ip": {"type": "ip", "ignore_malformed": True},
                "firewall_error": {"type": "text"},
                "assigned_to": {"type": "keyword"},
                "assigned_at": {"type": "date"},
                "assigned_by": {"type": "keyword"},
                "fp_reason": {"type": "text"},
                "fp_dismissed_by": {"type": "keyword"},
                "fp_dismissed_at": {"type": "date"},
                "closed_at": {"type": "date"},
                "closed_by": {"type": "keyword"},
                "resolution_notes": {"type": "text"},
                "mttd_seconds": {"type": "float"},
                "mttr_seconds": {"type": "float"},
                "actions_log": {
                    "type": "nested",
                    "properties": {
                        "action": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "actor": {"type": "keyword"},
                        "details": {"type": "text"},
                    }
                },
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }
    }

    def __init__(self):
        host = os.getenv("WAZUH_INDEXER_HOST", "localhost")
        port = int(os.getenv("WAZUH_INDEXER_PORT", "9200"))
        user = os.getenv("WAZUH_INDEXER_USER", "admin")
        password = os.getenv("WAZUH_INDEXER_PASS", "admin")
        verify_ssl = os.getenv("VERIFY_SSL", "false").lower() == "true"

        self.base_url = f"https://{host}:{port}"
        self.auth = (user, password)

        # Build an SSL context that handles Wazuh's internal CA — same pattern as
        # api/wazuh_indexer.py and api/suricata_client.py. The default Wazuh
        # internal CA cert is missing the keyUsage extension, which strict
        # X.509 verification rejects with
        #   "CA cert does not include key usage extension"
        # Without this, every IncidentStore call to OpenSearch fails on SSL
        # handshake and the analyst sees HTTP 500 on Acknowledge / Create Ticket.
        if not verify_ssl:
            # Permissive — match wazuh_indexer.initialize() VERIFY_SSL=false path.
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            try:
                ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
            except Exception:
                pass
            try:
                ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            except Exception:
                pass
        else:
            # Strict but tolerant of internal CAs without keyUsage. Hostname
            # check stays ON so SAN must still match.
            ssl_context = ssl.create_default_context()
            ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
            cert_file = os.getenv("SSL_CERT_FILE")
            if cert_file and os.path.isfile(cert_file):
                try:
                    ssl_context.load_verify_locations(cafile=cert_file)
                except Exception as e:
                    logger.warning(
                        f"[SSL] IncidentStore: failed to load SSL_CERT_FILE={cert_file}: {e}"
                    )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=self.auth,
            verify=ssl_context,
            timeout=15.0,
        )
        self._index_ensured = False
        logger.info(
            f"IncidentStore configured for {self.base_url} (verify_ssl={verify_ssl})"
        )

    async def _ensure_index(self):
        """Create pallas-incidents index if it doesn't exist."""
        if self._index_ensured:
            return
        try:
            resp = await self.client.head(f"/{self.INDEX_NAME}")
            if resp.status_code == 404:
                resp = await self.client.put(
                    f"/{self.INDEX_NAME}",
                    json=self.INDEX_MAPPING,
                )
                if resp.status_code in (200, 201):
                    logger.info(f"Created index '{self.INDEX_NAME}'")
                else:
                    logger.warning(f"Failed to create index: {resp.status_code} {resp.text}")
            self._index_ensured = True
        except Exception as e:
            logger.error(f"Index ensure failed: {e}")

    async def get_incident(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get incident by alert_id. Returns None if not found."""
        await self._ensure_index()
        try:
            query = {
                "query": {"term": {"alert_id": alert_id}},
                "size": 1
            }
            resp = await self.client.post(
                f"/{self.INDEX_NAME}/_search",
                json=query,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                return None
            doc = hits[0]["_source"]
            doc["_doc_id"] = hits[0]["_id"]
            return doc
        except Exception as e:
            logger.error(f"Get incident failed: {e}")
            return None

    async def create_incident(self, alert_id: str, alert: Dict[str, Any], analyst_name: str) -> Dict[str, Any]:
        """Create or update incident with acknowledged status."""
        await self._ensure_index()

        existing = await self.get_incident(alert_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            # Already exists - return current state
            return existing

        # Parse alert timestamp for MTTD calculation
        alert_ts = alert.get("timestamp") or alert.get("alert_timestamp", "")
        mttd_seconds = None
        if alert_ts:
            try:
                if alert_ts.endswith("Z"):
                    alert_dt = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
                else:
                    alert_dt = datetime.fromisoformat(alert_ts)
                if alert_dt.tzinfo is None:
                    alert_dt = alert_dt.replace(tzinfo=timezone.utc)
                mttd_seconds = (datetime.now(timezone.utc) - alert_dt).total_seconds()
            except Exception:
                pass

        incident_id = str(uuid.uuid4())
        doc = {
            "incident_id": incident_id,
            "alert_id": alert_id,
            "alert_source": alert.get("source", "unknown"),
            "rule_id": alert.get("rule_id", ""),
            "rule_description": alert.get("rule_description", ""),
            "severity": alert.get("severity", ""),
            "severity_num": alert.get("severity_num", 0),
            "source_ip": alert.get("source_ip", "") or None,
            "dest_ip": alert.get("dest_ip", "") or None,
            "agent_id": alert.get("agent_id", ""),
            "agent_name": alert.get("agent_name", ""),
            "alert_timestamp": alert_ts or now,
            "status": "acknowledged",
            "acknowledged_at": now,
            "acknowledged_by": analyst_name,
            "jira_ticket_id": None,
            "jira_ticket_url": None,
            "jira_created_at": None,
            "observability_completed": False,
            "observability_verdict": None,
            "observability_completed_at": None,
            "waf_blocked": False,
            "waf_blocked_at": None,
            "waf_blocked_ip": None,
            "waf_error": None,
            "firewall_blocked": False,
            "firewall_blocked_at": None,
            "firewall_blocked_ip": None,
            "firewall_error": None,
            "closed_at": None,
            "assigned_to": None,
            "assigned_at": None,
            "assigned_by": None,
            "fp_reason": None,
            "fp_dismissed_by": None,
            "fp_dismissed_at": None,
            "closed_by": None,
            "resolution_notes": None,
            "mttd_seconds": mttd_seconds,
            "mttr_seconds": None,
            "actions_log": [
                {
                    "action": "acknowledged",
                    "timestamp": now,
                    "actor": analyst_name,
                    "details": f"Alert acknowledged by {analyst_name}",
                }
            ],
            "created_at": now,
            "updated_at": now,
        }

        try:
            # `refresh=wait_for` makes OpenSearch hold the response until the new
            # document is visible in the next refresh cycle (default 1s). Without
            # this, a follow-up get_incident() called within the refresh window
            # returns None — which surfaces to the analyst as
            #   "Incident not found. Acknowledge the alert first."
            # on Create Ticket even though /acknowledge returned 200 OK moments
            # earlier. The cost is up to ~1s of latency on the ack call, which
            # is acceptable for an analyst action.
            resp = await self.client.post(
                f"/{self.INDEX_NAME}/_doc?refresh=wait_for",
                json=doc,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Incident created: {incident_id} for alert {alert_id}")
            else:
                # Previously this branch only logged and let the caller return
                # the in-memory `doc` to the client — so the analyst saw a fake
                # 200 with a UUID, then Create Ticket failed minutes later
                # because nothing was actually persisted. Raise instead so the
                # ack endpoint returns 500 and the analyst can retry. The full
                # OpenSearch error body is included for debugging.
                err_body = resp.text[:500] if resp.text else "(empty body)"
                logger.error(
                    f"Incident create failed: {resp.status_code} {err_body}"
                )
                raise RuntimeError(
                    f"Failed to persist incident in OpenSearch "
                    f"(status {resp.status_code}): {err_body}"
                )
        except RuntimeError:
            # Already logged above — re-raise so the API layer returns 500.
            raise
        except Exception as e:
            logger.error(f"Incident create exception: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to persist incident in OpenSearch: {type(e).__name__}: {e}"
            )

        return doc

    async def update_incident(self, alert_id: str, updates: Dict[str, Any], action_log: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Update incident fields and optionally append to actions_log."""
        await self._ensure_index()

        existing = await self.get_incident(alert_id)
        if not existing:
            return None

        doc_id = existing.get("_doc_id")
        if not doc_id:
            return None

        now = datetime.now(timezone.utc).isoformat()
        updates["updated_at"] = now

        # Build update script for actions_log append
        if action_log:
            script = {
                "script": {
                    "source": "ctx._source.putAll(params.updates); ctx._source.actions_log.add(params.action_log)",
                    "params": {
                        "updates": updates,
                        "action_log": action_log,
                    }
                }
            }
        else:
            script = {"doc": updates}

        try:
            if action_log:
                resp = await self.client.post(
                    f"/{self.INDEX_NAME}/_update/{doc_id}",
                    json=script,
                )
            else:
                resp = await self.client.post(
                    f"/{self.INDEX_NAME}/_update/{doc_id}",
                    json={"doc": updates},
                )

            if resp.status_code in (200, 201):
                return await self.get_incident(alert_id)
            else:
                logger.error(f"Incident update failed: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Incident update exception: {e}")
            return None

    async def get_incidents_batch(self, alert_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple incidents by alert_ids in a single query."""
        await self._ensure_index()
        if not alert_ids:
            return {}

        try:
            query = {
                "query": {"terms": {"alert_id": alert_ids}},
                "size": len(alert_ids),
            }
            resp = await self.client.post(
                f"/{self.INDEX_NAME}/_search",
                json=query,
            )
            if resp.status_code != 200:
                return {}

            data = resp.json()
            result = {}
            for hit in data.get("hits", {}).get("hits", []):
                doc = hit["_source"]
                doc["_doc_id"] = hit["_id"]
                result[doc["alert_id"]] = doc
            return result
        except Exception as e:
            logger.error(f"Batch get failed: {e}")
            return {}

