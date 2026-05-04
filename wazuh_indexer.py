"""
Wazuh Indexer client for vulnerability queries (Wazuh 4.8.0+).

Since Wazuh 4.8.0, vulnerability data is stored in the Wazuh Indexer
(Elasticsearch/OpenSearch) instead of being available via the Wazuh Manager API.

The vulnerability API endpoint (/vulnerability/*) was deprecated in 4.7.0
and removed in 4.8.0. This client queries the wazuh-states-vulnerabilities-*
index directly.
"""

import logging
from typing import Dict, Any, Optional
import httpx
import ssl

logger = logging.getLogger(__name__)

# Vulnerability index pattern for Wazuh 4.8+
VULNERABILITY_INDEX = "wazuh-states-vulnerabilities-*"
# Alert index pattern
ALERT_INDEX = "wazuh-alerts-*"


class WazuhIndexerClient:
    """
    Client for querying the Wazuh Indexer (Elasticsearch/OpenSearch).

    Required for vulnerability queries in Wazuh 4.8.0 and later.
    Supports HTTPS with flexible SSL verification.
    """

    def __init__(
        self,
        host: str,
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: int = 30
    ):
        # Clean host - remove any protocol prefix
        self.host = host.replace('https://', '').replace('http://', '').strip()
        self.port = port
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False

        logger.info(f"Indexer client configured for https://{self.host}:{self.port} (verify_ssl={verify_ssl})")

    @property
    def base_url(self) -> str:
        """Get the base URL for the Wazuh Indexer."""
        return f"https://{self.host}:{self.port}"

    async def initialize(self):
        """Initialize the HTTP client with proper SSL handling."""
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)

        # Create custom SSL context that handles self-signed/untrusted certificates
        if not self.verify_ssl:
            # Create a permissive SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Relax cipher requirements for older TLS versions
            try:
                ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
            except Exception:
                # If setting ciphers fails, continue anyway
                pass

            # Set minimum TLS version to support older configurations
            try:
                ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            except Exception:
                pass
        else:
            ssl_context = None

        # Initialize httpx client with SSL configuration
        self.client = httpx.AsyncClient(
            verify=ssl_context if ssl_context else True,
            timeout=httpx.Timeout(
                connect=self.timeout,
                read=self.timeout,
                write=self.timeout,
                pool=self.timeout
            ),
            auth=auth,
            http2=False,  # Disable HTTP/2 for better compatibility
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10
            )
        )
        self._initialized = True
        logger.info(f"WazuhIndexerClient initialized for {self.base_url}")

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self._initialized = False

    async def _ensure_initialized(self):
        """Ensure client is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _search(self, index: str, query: Dict[str, Any], size: int = 100, sort: list = None) -> Dict[str, Any]:
        """
        Execute a search query against the Wazuh Indexer.

        Args:
            index: Index pattern to search
            query: Elasticsearch query DSL
            size: Maximum number of results
            sort: Optional sort clauses (e.g. [{"@timestamp": "asc"}])

        Returns:
            Search results from the indexer
        """
        await self._ensure_initialized()

        url = f"{self.base_url}/{index}/_search"
        body = {
            "query": query,
            "size": size
        }
        if sort:
            body["sort"] = sort

        try:
            response = await self.client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Indexer search failed: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Indexer query failed: {e.response.status_code}")
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Wazuh Indexer: {e}")
            raise ConnectionError(f"Cannot connect to Wazuh Indexer at {self.base_url}. Check SSL/TLS settings.")
        except httpx.TimeoutException:
            raise ConnectionError(f"Timeout connecting to Wazuh Indexer at {self.base_url}")
        except Exception as e:
            logger.error(f"Unexpected error querying Indexer: {e}")
            raise

    # ===================================================================
    # ALERT METHODS (NEW - for Wazuh 4.8+ where /alerts API is removed)
    # ===================================================================

    async def search_alerts(
        self,
        query: Optional[str] = None,
        time_range: str = "24h",
        size: int = 100,
        **params
    ) -> Dict[str, Any]:
        """
        Search alerts in Wazuh alert indices.

        Args:
            query: Query string (optional)
            time_range: Time range (e.g., "1h", "24h", "7d")
            size: Maximum number of results
            **params: Additional filters (rule_id, level, agent_id, etc.)

        Returns:
            Alert data matching the criteria
        """
        # Build query
        must_clauses = []

        # Add time range filter
        must_clauses.append({
            "range": {
                "@timestamp": {
                    "gte": f"now-{time_range}",
                    "lte": "now"
                }
            }
        })

        # Add query string if provided
        if query:
            must_clauses.append({
                "query_string": {
                    "query": query
                }
            })

        # Add specific filters from params
        if params.get("rule_id"):
            must_clauses.append({"term": {"rule.id": params["rule_id"]}})
        if params.get("level"):
            level_val = int(params["level"]) if isinstance(params["level"], str) else params["level"]
            must_clauses.append({"term": {"rule.level": level_val}})
        if params.get("min_level"):
            min_val = int(params["min_level"]) if isinstance(params["min_level"], str) else params["min_level"]
            must_clauses.append({"range": {"rule.level": {"gte": min_val}}})
        if params.get("agent_id"):
            must_clauses.append({"term": {"agent.id": params["agent_id"]}})

        # Build the query
        search_query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}

        sort_order = params.get("sort")
        result = await self._search(ALERT_INDEX, search_query, size=size, sort=sort_order)

        # Transform to standard Wazuh API format
        hits = result.get("hits", {})
        alerts = []

        for hit in hits.get("hits", []):
            source = hit.get("_source", {})
            # Preserve OpenSearch's stable doc _id for downstream alert_id use.
            # Position-based ids (`wz_{idx}_{rule}`) collided across fetches and
            # stuck stale incident_store records onto fresh alerts.
            source["_doc_id"] = hit.get("_id")
            alerts.append(source)

        return {
            "data": {
                "affected_items": alerts,
                "total_affected_items": hits.get("total", {}).get("value", len(alerts)),
                "total_failed_items": 0,
                "failed_items": []
            }
        }

    async def get_alert_aggregation(
        self,
        time_range: str = "24h",
        group_by: str = "rule.level"
    ) -> Dict[str, Any]:
        """
        Get alert aggregation grouped by field.

        Args:
            time_range: Time range for aggregation
            group_by: Field to group by

        Returns:
            Aggregated alert data
        """
        await self._ensure_initialized()

        # Fields that should be used as-is (already keyword type or numeric in Wazuh mapping)
        direct_fields = {"rule.level", "rule.firedtimes", "data.alert.severity", "rule.id", "rule.groups", "agent.id"}
        if group_by in direct_fields:
            field_name = group_by
        elif "." in group_by:
            field_name = f"{group_by}.keyword"
        else:
            field_name = group_by

        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": f"now-{time_range}",
                        "lte": "now"
                    }
                }
            },
            "aggs": {
                "group_by": {
                    "terms": {
                        "field": field_name,
                        "size": 50
                    }
                }
            }
        }

        try:
            response = await self.client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Alert aggregation failed: {e.response.status_code}")
            raise ValueError(f"Alert aggregation query failed: {e.response.status_code}")

    # ===================================================================
    # VULNERABILITY METHODS (Original - for Wazuh 4.8+)
    # ===================================================================

    async def get_vulnerabilities(
        self,
        agent_id: Optional[str] = None,
        severity: Optional[str] = None,
        cve_id: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get vulnerabilities from the Wazuh Indexer.

        Args:
            agent_id: Filter by agent ID
            severity: Filter by severity (Critical, High, Medium, Low)
            cve_id: Filter by specific CVE ID
            limit: Maximum number of results

        Returns:
            Vulnerability data matching the criteria
        """
        # Build query
        must_clauses = []

        if agent_id:
            # Use term for exact ID matching (not analyzed)
            must_clauses.append({"term": {"agent.id": str(agent_id)}})

        if severity:
            # Normalize severity to match indexer format
            severity_normalized = severity.capitalize()
            must_clauses.append({"match": {"vulnerability.severity": severity_normalized}})

        if cve_id:
            must_clauses.append({"match": {"vulnerability.id": cve_id}})

        # Build the query
        if must_clauses:
            query = {"bool": {"must": must_clauses}}
        else:
            query = {"match_all": {}}

        result = await self._search(VULNERABILITY_INDEX, query, size=limit)

        # Transform to standard format
        hits = result.get("hits", {})
        vulnerabilities = []

        for hit in hits.get("hits", []):
            source = hit.get("_source", {})
            vulnerabilities.append({
                "id": source.get("vulnerability", {}).get("id"),
                "severity": source.get("vulnerability", {}).get("severity"),
                "description": source.get("vulnerability", {}).get("description"),
                "reference": source.get("vulnerability", {}).get("reference"),
                "status": source.get("vulnerability", {}).get("status"),
                "detected_at": source.get("vulnerability", {}).get("detected_at"),
                "published_at": source.get("vulnerability", {}).get("published_at"),
                "agent": {
                    "id": source.get("agent", {}).get("id"),
                    "name": source.get("agent", {}).get("name"),
                },
                "package": {
                    "name": source.get("package", {}).get("name"),
                    "version": source.get("package", {}).get("version"),
                    "architecture": source.get("package", {}).get("architecture"),
                }
            })

        return {
            "data": {
                "affected_items": vulnerabilities,
                "total_affected_items": hits.get("total", {}).get("value", len(vulnerabilities)),
                "total_failed_items": 0,
                "failed_items": []
            }
        }

    async def get_critical_vulnerabilities(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get critical severity vulnerabilities.

        Args:
            limit: Maximum number of results

        Returns:
            Critical vulnerability data
        """
        return await self.get_vulnerabilities(severity="Critical", limit=limit)

    async def get_vulnerability_summary(self) -> Dict[str, Any]:
        """
        Get vulnerability summary statistics.

        Returns:
            Summary with counts by severity
        """
        await self._ensure_initialized()

        # Aggregation query for severity counts
        url = f"{self.base_url}/{VULNERABILITY_INDEX}/_search"
        body = {
            "size": 0,
            "aggs": {
                "by_severity": {
                    "terms": {
                        "field": "vulnerability.severity",
                        "size": 10
                    }
                },
                "by_agent": {
                    "cardinality": {
                        "field": "agent.id"
                    }
                },
                "total_vulnerabilities": {
                    "value_count": {
                        "field": "vulnerability.id"
                    }
                }
            }
        }

        try:
            response = await self.client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            # Parse aggregations
            aggs = result.get("aggregations", {})
            severity_buckets = aggs.get("by_severity", {}).get("buckets", [])

            severity_counts = {}
            for bucket in severity_buckets:
                severity_counts[bucket.get("key", "unknown")] = bucket.get("doc_count", 0)

            return {
                "data": {
                    "total_vulnerabilities": aggs.get("total_vulnerabilities", {}).get("value", 0),
                    "affected_agents": aggs.get("by_agent", {}).get("value", 0),
                    "by_severity": severity_counts,
                    "critical": severity_counts.get("Critical", 0),
                    "high": severity_counts.get("High", 0),
                    "medium": severity_counts.get("Medium", 0),
                    "low": severity_counts.get("Low", 0)
                }
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Vulnerability summary query failed: {e.response.status_code}")
            raise ValueError(f"Vulnerability summary query failed: {e.response.status_code}")
        except httpx.ConnectError:
            raise ConnectionError(f"Cannot connect to Wazuh Indexer at {self.base_url}")

    # ===================================================================
    # ROUND 6: ADVANCED ANALYTICS METHODS
    # ===================================================================

    async def get_rule_trigger_counts(self, time_range: str = "24h", limit: int = 20) -> Dict[str, Any]:
        """Get top rule trigger counts with severity sub-aggregation."""
        await self._ensure_initialized()
        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            "aggs": {
                "by_rule": {
                    "terms": {"field": "rule.id", "size": limit, "order": {"_count": "desc"}},
                    "aggs": {
                        "level": {"terms": {"field": "rule.level", "size": 5}},
                        "description": {"terms": {"field": "rule.description.keyword", "size": 1}},
                        "mitre_technique": {"terms": {"field": "rule.mitre.technique.keyword", "size": 5}},
                        "mitre_tactic": {"terms": {"field": "rule.mitre.tactic.keyword", "size": 5}},
                        "groups": {"terms": {"field": "rule.groups.keyword", "size": 5}}
                    }
                },
                "total_alerts": {"value_count": {"field": "_id"}}
            }
        }
        try:
            response = await self.client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            aggs = result.get("aggregations", {})
            total = aggs.get("total_alerts", {}).get("value", 0)
            buckets = aggs.get("by_rule", {}).get("buckets", [])
            rules = []
            for b in buckets:
                level_buckets = b.get("level", {}).get("buckets", [])
                top_level = level_buckets[0]["key"] if level_buckets else 0
                desc_buckets = b.get("description", {}).get("buckets", [])
                desc = desc_buckets[0]["key"] if desc_buckets else "N/A"
                mitre_t = [m["key"] for m in b.get("mitre_technique", {}).get("buckets", [])]
                mitre_tac = [m["key"] for m in b.get("mitre_tactic", {}).get("buckets", [])]
                groups = [g["key"] for g in b.get("groups", {}).get("buckets", [])]
                rules.append({
                    "rule_id": b["key"], "count": b["doc_count"], "level": top_level,
                    "description": desc, "mitre_techniques": mitre_t,
                    "mitre_tactics": mitre_tac, "groups": groups
                })
            return {"rules": rules, "total_alerts": total}
        except Exception as e:
            logger.error(f"Rule trigger count query failed: {e}")
            return {"rules": [], "total_alerts": 0, "error": str(e)}

    async def get_mitre_coverage(self, time_range: str = "7d") -> Dict[str, Any]:
        """Get MITRE ATT&CK technique and tactic coverage from alerts."""
        await self._ensure_initialized()
        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            "aggs": {
                "by_technique": {
                    "terms": {"field": "rule.mitre.technique.keyword", "size": 200},
                    "aggs": {
                        "tactics": {"terms": {"field": "rule.mitre.tactic.keyword", "size": 10}},
                        "rule_ids": {"terms": {"field": "rule.id", "size": 5}}
                    }
                },
                "by_tactic": {
                    "terms": {"field": "rule.mitre.tactic.keyword", "size": 50}
                }
            }
        }
        try:
            response = await self.client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            aggs = result.get("aggregations", {})
            techniques = []
            for b in aggs.get("by_technique", {}).get("buckets", []):
                tactics = [t["key"] for t in b.get("tactics", {}).get("buckets", [])]
                rule_ids = [r["key"] for r in b.get("rule_ids", {}).get("buckets", [])]
                techniques.append({"technique": b["key"], "count": b["doc_count"],
                                   "tactics": tactics, "rule_ids": rule_ids})
            tactic_counts = {b["key"]: b["doc_count"]
                            for b in aggs.get("by_tactic", {}).get("buckets", [])}
            return {"techniques": techniques, "tactic_counts": tactic_counts}
        except Exception as e:
            logger.error(f"MITRE coverage query failed: {e}")
            return {"techniques": [], "tactic_counts": {}, "error": str(e)}

    async def get_alert_timeline(self, time_range: str = "24h", interval: str = "1h") -> Dict[str, Any]:
        """Get alert counts over time using date_histogram."""
        await self._ensure_initialized()
        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            "aggs": {
                "timeline": {
                    "date_histogram": {"field": "@timestamp", "fixed_interval": interval,
                                       "min_doc_count": 0}
                },
                "by_severity": {
                    "range": {
                        "field": "rule.level",
                        "ranges": [
                            {"key": "low", "from": 0, "to": 4},
                            {"key": "medium", "from": 4, "to": 7},
                            {"key": "high", "from": 7, "to": 12},
                            {"key": "critical", "from": 12}
                        ]
                    }
                }
            }
        }
        try:
            response = await self.client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            aggs = result.get("aggregations", {})
            buckets = []
            for b in aggs.get("timeline", {}).get("buckets", []):
                buckets.append({"timestamp": b.get("key_as_string", b.get("key")),
                                "count": b.get("doc_count", 0)})
            severity = {b["key"]: b["doc_count"]
                        for b in aggs.get("by_severity", {}).get("buckets", [])}
            return {"buckets": buckets, "severity_breakdown": severity}
        except Exception as e:
            logger.error(f"Alert timeline query failed: {e}")
            return {"buckets": [], "severity_breakdown": {}, "error": str(e)}

    async def get_agent_event_counts(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get per-agent event counts for log source health."""
        await self._ensure_initialized()
        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            "aggs": {
                "by_agent": {
                    "terms": {"field": "agent.id", "size": 200},
                    "aggs": {"agent_name": {"terms": {"field": "agent.name.keyword", "size": 1}}}
                }
            }
        }
        try:
            response = await self.client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            buckets = result.get("aggregations", {}).get("by_agent", {}).get("buckets", [])
            agents = []
            for b in buckets:
                name_b = b.get("agent_name", {}).get("buckets", [])
                name = name_b[0]["key"] if name_b else "N/A"
                agents.append({"agent_id": b["key"], "agent_name": name, "event_count": b["doc_count"]})
            return {"agents": agents}
        except Exception as e:
            logger.error(f"Agent event count query failed: {e}")
            return {"agents": [], "error": str(e)}

    async def get_fim_events(self, agent_id: str = None, time_range: str = "24h",
                             limit: int = 50) -> Dict[str, Any]:
        """Get File Integrity Monitoring events from syscheck rule group."""
        await self._ensure_initialized()
        must_clauses = [
            {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            {"term": {"rule.groups.keyword": "syscheck"}}
        ]
        if agent_id:
            must_clauses.append({"term": {"agent.id": agent_id}})
        search_query = {"bool": {"must": must_clauses}}
        try:
            result = await self._search(ALERT_INDEX, search_query, size=limit)
            hits = result.get("hits", {}).get("hits", [])
            total = result.get("hits", {}).get("total", {})
            total_count = total.get("value", 0) if isinstance(total, dict) else total
            events = []
            for hit in hits:
                src = hit.get("_source", {})
                syscheck = src.get("syscheck", {}) if isinstance(src.get("syscheck"), dict) else {}
                agent = src.get("agent", {}) if isinstance(src.get("agent"), dict) else {}
                events.append({
                    "timestamp": src.get("@timestamp") or src.get("timestamp"),
                    "agent_id": agent.get("id"), "agent_name": agent.get("name"),
                    "path": syscheck.get("path", "N/A"),
                    "event": syscheck.get("event", "N/A"),
                    "user_name": syscheck.get("uname", "N/A"),
                    "group_name": syscheck.get("gname", "N/A"),
                    "size_after": syscheck.get("size_after"),
                    "md5_after": syscheck.get("md5_after"),
                    "sha1_after": syscheck.get("sha1_after"),
                    "rule_description": src.get("rule", {}).get("description", "N/A"),
                    "rule_level": src.get("rule", {}).get("level", 0)
                })
            return {"events": events, "total": total_count}
        except Exception as e:
            logger.error(f"FIM events query failed: {e}")
            return {"events": [], "total": 0, "error": str(e)}

    async def get_alerts_by_agent_severity(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get alert counts grouped by agent and severity for cross-correlation."""
        await self._ensure_initialized()
        url = f"{self.base_url}/{ALERT_INDEX}/_search"
        body = {
            "size": 0,
            "query": {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            "aggs": {
                "by_agent": {
                    "terms": {"field": "agent.id", "size": 200},
                    "aggs": {
                        "agent_name": {"terms": {"field": "agent.name.keyword", "size": 1}},
                        "severity": {
                            "range": {
                                "field": "rule.level",
                                "ranges": [
                                    {"key": "low", "from": 0, "to": 4},
                                    {"key": "medium", "from": 4, "to": 7},
                                    {"key": "high", "from": 7, "to": 12},
                                    {"key": "critical", "from": 12}
                                ]
                            }
                        }
                    }
                }
            }
        }
        try:
            response = await self.client.post(url, json=body, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            buckets = result.get("aggregations", {}).get("by_agent", {}).get("buckets", [])
            agents = []
            for b in buckets:
                name_b = b.get("agent_name", {}).get("buckets", [])
                name = name_b[0]["key"] if name_b else "N/A"
                sev = {s["key"]: s["doc_count"] for s in b.get("severity", {}).get("buckets", [])}
                agents.append({"agent_id": b["key"], "agent_name": name,
                               "total_alerts": b["doc_count"], "severity": sev})
            return {"agents": agents}
        except Exception as e:
            logger.error(f"Alerts by agent severity query failed: {e}")
            return {"agents": [], "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Wazuh Indexer health status.

        Returns:
            Health status information
        """
        await self._ensure_initialized()

        try:
            response = await self.client.get(f"{self.base_url}/_cluster/health")
            response.raise_for_status()
            health = response.json()

            return {
                "status": health.get("status"),
                "cluster_name": health.get("cluster_name"),
                "number_of_nodes": health.get("number_of_nodes"),
                "active_shards": health.get("active_shards")
            }

        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e)
            }


class IndexerNotConfiguredError(Exception):
    """Raised when Wazuh Indexer is not configured but required."""

    def __init__(self, message: str = None):
        default_message = (
            "Wazuh Indexer not configured. "
            "Vulnerability and alert tools require the Wazuh Indexer for Wazuh 4.8.0+.\n\n"
            "Please set the following environment variables:\n"
            "  WAZUH_INDEXER_HOST=test-indexer.athenasecuritygrp.com (or your indexer IP)\n"
            "  WAZUH_INDEXER_USER=admin\n"
            "  WAZUH_INDEXER_PASS=<your_password>\n"
            "  WAZUH_INDEXER_PORT=9200 (optional, default: 9200)\n"
            "  WAZUH_VERIFY_SSL=false (for self-signed certificates)\n\n"
            "Note: The /alerts and /vulnerability APIs were removed in Wazuh 4.8.0. "
            "Data must be queried from the Wazuh Indexer."
        )
        super().__init__(message or default_message)
