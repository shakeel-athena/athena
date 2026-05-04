"""Suricata IDS client optimized for Suricata Elasticsearch integration"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import httpx

from wazuh_mcp_server.config import WazuhConfig
from wazuh_mcp_server.resilience import CircuitBreaker, CircuitBreakerConfig, RetryConfig

logger = logging.getLogger(__name__)


class SuricataClient:
    """Suricata Elasticsearch client for IDS alerts."""

    def __init__(self, config: WazuhConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

        # Use indexer configuration for Suricata
        if not config.wazuh_indexer_host:
            raise ValueError("WAZUH_INDEXER_HOST required for Suricata integration")

        # Clean hostname
        indexer_host = config.wazuh_indexer_host.replace('https://', '').replace('http://', '').strip()

        self.base_url = f"https://{indexer_host}:{config.wazuh_indexer_port}"
        self.index_pattern = "suricata-1.1.0-*"
        self.username = config.wazuh_indexer_user
        self.password = config.wazuh_indexer_pass
        self.verify_ssl = config.verify_ssl
        self.timeout = config.request_timeout_seconds

        self._rate_limiter = asyncio.Semaphore(config.max_connections)
        self._request_times = []
        self._max_requests_per_minute = getattr(config, "max_requests_per_minute", 100)

        cb_cfg = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
        )
        self._circuit_breaker = CircuitBreaker(cb_cfg)

        # Event type field — auto-detected during initialize()
        self.event_type_field = "event_type.keyword"  # default; overridden by _detect_event_type_field

        logger.info(f"SuricataClient initialized: {self.base_url}/{self.index_pattern}")

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    async def initialize(self):
        """Initialize HTTP client."""
        # Create SSL context
        import ssl
        if not self.verify_ssl:
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
            ssl_context = None

        self.client = httpx.AsyncClient(
            verify=ssl_context if ssl_context else True,
            timeout=self.timeout,
            auth=(self.username, self.password) if self.username and self.password else None,
            http2=False,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Test connection
        try:
            await self.health_check()
            logger.info("Suricata Elasticsearch connection successful")
        except Exception as e:
            logger.warning(f"Suricata Elasticsearch connection test failed: {e}")

        # Auto-detect the event type field name
        await self._detect_event_type_field()

    async def _detect_event_type_field(self):
        """Auto-detect which field holds the event type (alert/http/tls/etc).

        Different Suricata index templates use different field names:
        - event_type / event_type.keyword  (Filebeat Suricata module)
        - event.subtype / event.subtype.keyword  (some ECS mappings)
        - event_type (raw Suricata JSON via Logstash)
        """
        candidates = [
            "event_type.keyword",
            "event_type",
            "event.subtype.keyword",
            "event.subtype",
        ]
        try:
            # Query a single doc to check which fields exist
            body = {
                "size": 0,
                "aggs": {}
            }
            for field in candidates:
                agg_name = field.replace(".", "_").replace(" ", "_")
                body["aggs"][agg_name] = {
                    "terms": {"field": field, "size": 5}
                }

            resp = await self._request("POST", f"/{self.index_pattern}/_search", json=body)
            aggs = resp.get("aggregations", {})

            # Pick the first candidate that returned buckets
            for field in candidates:
                agg_name = field.replace(".", "_").replace(" ", "_")
                buckets = aggs.get(agg_name, {}).get("buckets", [])
                if buckets:
                    self.event_type_field = field
                    event_types = [b.get("key") for b in buckets]
                    logger.info(f"Suricata event type field detected: '{field}' → {event_types}")
                    return

            logger.warning("Could not detect Suricata event type field — using default 'event_type.keyword'")
        except Exception as e:
            logger.warning(f"Event type field detection failed: {e} — using default 'event_type.keyword'")

    # ------------------------------------------------------------------
    # CORE REQUEST
    # ------------------------------------------------------------------

    async def _rate_limit(self):
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._max_requests_per_minute:
            await asyncio.sleep(1)
        self._request_times.append(now)

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        async with self._rate_limiter:
            await self._rate_limit()
            return await self._circuit_breaker._call(
                self._execute_request, method, endpoint, **kwargs
            )

    @RetryConfig.WAZUH_API_RETRY
    async def _execute_request(self, method: str, endpoint: str, **kwargs):
        """Execute HTTP request to Elasticsearch."""
        url = f"{self.base_url}{endpoint}"

        try:
            resp = await self.client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Suricata ES error {e.response.status_code}: {e.response.text}")
            raise ValueError(
                f"Suricata ES request failed: {e.response.status_code} - {e.response.text}"
            )

    # ------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Check cluster health."""
        try:
            response = await self._request("GET", "/_cluster/health")
            return {
                "status": response.get("status"),
                "cluster_name": response.get("cluster_name"),
                "number_of_nodes": response.get("number_of_nodes"),
                "active_shards": response.get("active_shards")
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # SURICATA ALERTS
    # ------------------------------------------------------------------

    async def get_alerts(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        signature: Optional[str] = None,
        src_ip: Optional[str] = None,
        dest_ip: Optional[str] = None,
        time_range: str = "24h",
        **params
    ) -> Dict[str, Any]:
        """
        Get Suricata IDS alerts.

        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity (1=Critical, 2=High, 3=Medium, 4+=Low)
            category: Filter by alert category
            signature: Filter by signature name
            src_ip: Filter by source IP
            dest_ip: Filter by destination IP
            time_range: Time range (e.g., "1h", "24h", "7d")
        """
        # Build query
        must_clauses = []
        should_clauses = []

        # CRITICAL: Only return actual alert events (not HTTP/TLS/flow events)
        must_clauses.append({"term": {self.event_type_field: "alert"}})

        # Time range filter
        must_clauses.append({
            "range": {
                "@timestamp": {
                    "gte": f"now-{time_range}",
                    "lte": "now"
                }
            }
        })

        # Severity filter
        if severity:
            try:
                sev_val = int(severity)
                must_clauses.append({"term": {"alert.severity": sev_val}})
            except ValueError:
                # Handle "5+" case
                if severity == "5+":
                    must_clauses.append({"range": {"alert.severity": {"gte": 5}}})

        # Category filter (multi-field support)
        if category:
            should_clauses.extend([
                {"term": {"alert.category.keyword": category}},
                {"term": {"alert.category": category}},
                {"match_phrase": {"alert.category": category}}
            ])

        # Signature filter
        if signature:
            should_clauses.extend([
                {"term": {"alert.signature.keyword": signature}},
                {"term": {"alert.signature": signature}},
                {"match_phrase": {"alert.signature": signature}}
            ])

        # Source IP filter (multi-field support)
        if src_ip:
            should_clauses.extend([
                {"term": {"src_ip.keyword": src_ip}},
                {"term": {"src_ip": src_ip}},
                {"term": {"source.ip": src_ip}}
            ])

        # Destination IP filter
        if dest_ip:
            should_clauses.extend([
                {"term": {"dest_ip.keyword": dest_ip}},
                {"term": {"dest_ip": dest_ip}},
                {"term": {"destination.ip": dest_ip}}
            ])

        # Build final query
        query = {"bool": {"must": must_clauses}}
        if should_clauses:
            query["bool"]["should"] = should_clauses
            query["bool"]["minimum_should_match"] = 1

        # Execute search
        body = {
            "query": query,
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "track_total_hits": True
        }

        response = await self._request(
            "POST",
            f"/{self.index_pattern}/_search",
            json=body
        )

        # Transform to standard format
        hits = response.get("hits", {})
        alerts = []

        for hit in hits.get("hits", []):
            source = hit.get("_source", {})
            # Preserve OpenSearch's stable doc _id for downstream alert_id use.
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

    # ------------------------------------------------------------------
    # ALERT SUMMARY & STATISTICS
    # ------------------------------------------------------------------

    async def get_alert_summary(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get alert summary with severity breakdown."""
        query = {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": f"now-{time_range}",
                                "lte": "now"
                            }
                        }
                    },
                    {"term": {self.event_type_field: "alert"}}
                ]
            }
        }

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "severity_breakdown": {
                    "terms": {
                        "field": "alert.severity",
                        "size": 20
                    }
                },
                "top_categories": {
                    "terms": {
                        "field": "alert.category.keyword",
                        "size": 50
                    }
                },
                "top_signatures": {
                    "terms": {
                        "field": "alert.signature.keyword",
                        "size": 50
                    }
                },
                "timeline": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "calendar_interval": "1h",
                        "min_doc_count": 0,
                        "extended_bounds": {
                            "min": f"now-{time_range}",
                            "max": "now"
                        }
                    }
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST",
            f"/{self.index_pattern}/_search",
            json=body
        )

        # Process aggregations
        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        severity_breakdown = {}
        for bucket in aggs.get("severity_breakdown", {}).get("buckets", []):
            severity_breakdown[str(bucket.get("key"))] = bucket.get("doc_count", 0)

        top_categories = [
            {"category": bucket.get("key"), "count": bucket.get("doc_count")}
            for bucket in aggs.get("top_categories", {}).get("buckets", [])
        ]

        top_signatures = [
            {"signature": bucket.get("key"), "count": bucket.get("doc_count")}
            for bucket in aggs.get("top_signatures", {}).get("buckets", [])
        ]

        timeline = [
            {"timestamp": bucket.get("key_as_string"), "count": bucket.get("doc_count")}
            for bucket in aggs.get("timeline", {}).get("buckets", [])
        ]

        return {
            "total_alerts": total,
            "time_range": time_range,
            "severity_breakdown": severity_breakdown,
            "top_categories": top_categories,
            "top_signatures": top_signatures,
            "timeline": timeline
        }

    async def get_top_alerts_by_severity(self, severity: int, limit: int = 10, time_range: str = "24h") -> Dict[str, Any]:
        """Get top alerts for a specific severity level."""
        return await self.get_alerts(
            limit=limit,
            severity=str(severity),
            time_range=time_range
        )

    async def get_critical_alerts(self, limit: int = 50, time_range: str = "24h") -> Dict[str, Any]:
        """Get critical severity alerts (severity=1)."""
        return await self.get_alerts(limit=limit, severity="1", time_range=time_range)

    async def get_high_alerts(self, limit: int = 50, time_range: str = "24h") -> Dict[str, Any]:
        """Get high severity alerts (severity=2)."""
        return await self.get_alerts(limit=limit, severity="2", time_range=time_range)

    # ------------------------------------------------------------------
    # NETWORK ANALYSIS
    # ------------------------------------------------------------------

    async def get_network_analysis(self, time_range: str = "24h", limit: int = 10) -> Dict[str, Any]:
        """Get network analysis aggregations."""
        query = {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": f"now-{time_range}",
                                "lte": "now"
                            }
                        }
                    }
                ]
            }
        }

        # Try both native EVE (src_ip) and ECS (source.ip) field names
        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "top_source_ips": {
                    "terms": {"field": "src_ip.keyword", "size": limit}
                },
                "top_source_ips_ecs": {
                    "terms": {"field": "source.ip", "size": limit}
                },
                "top_dest_ips": {
                    "terms": {"field": "dest_ip.keyword", "size": limit}
                },
                "top_dest_ips_ecs": {
                    "terms": {"field": "destination.ip", "size": limit}
                },
                "top_services": {
                    "terms": {"field": "app_proto.keyword", "size": limit}
                },
                "top_services_ecs": {
                    "terms": {"field": "network.protocol.keyword", "size": limit}
                },
                "top_src_hostnames": {
                    "terms": {"field": "src_hostname.keyword", "size": limit, "missing": "N/A"}
                },
                "top_dest_hostnames": {
                    "terms": {"field": "dest_hostname.keyword", "size": limit, "missing": "N/A"}
                }
            }
        }

        response = await self._request(
            "POST",
            f"/{self.index_pattern}/_search",
            json=body
        )

        aggs = response.get("aggregations", {})

        # Merge native EVE and ECS field results (use whichever has data)
        def _pick_buckets(primary_key, fallback_key):
            buckets = aggs.get(primary_key, {}).get("buckets", [])
            if not buckets:
                buckets = aggs.get(fallback_key, {}).get("buckets", [])
            return buckets

        src_buckets = _pick_buckets("top_source_ips", "top_source_ips_ecs")
        dst_buckets = _pick_buckets("top_dest_ips", "top_dest_ips_ecs")
        svc_buckets = _pick_buckets("top_services", "top_services_ecs")

        return {
            "top_source_ips": [
                {"ip": b.get("key"), "count": b.get("doc_count")}
                for b in src_buckets
            ],
            "top_dest_ips": [
                {"ip": b.get("key"), "count": b.get("doc_count")}
                for b in dst_buckets
            ],
            "top_services": [
                {"service": b.get("key"), "count": b.get("doc_count")}
                for b in svc_buckets
            ],
            "top_src_hostnames": [
                {"hostname": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_src_hostnames", {}).get("buckets", [])
            ],
            "top_dest_hostnames": [
                {"hostname": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_dest_hostnames", {}).get("buckets", [])
            ]
        }

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    async def search_alerts(
        self,
        query: str,
        time_range: str = "24h",
        limit: int = 100
    ) -> Dict[str, Any]:
        """Search alerts using multi-match query."""
        search_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "alert.signature",
                                "alert.category",
                                "src_ip",
                                "dest_ip"
                            ],
                            "type": "phrase_prefix"
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": f"now-{time_range}",
                                "lte": "now"
                            }
                        }
                    },
                    {"term": {self.event_type_field: "alert"}}
                ]
            }
        }

        body = {
            "query": search_query,
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "track_total_hits": True
        }

        response = await self._request(
            "POST",
            f"/{self.index_pattern}/_search",
            json=body
        )

        hits = response.get("hits", {})
        alerts = [hit.get("_source", {}) for hit in hits.get("hits", [])]

        return {
            "data": {
                "affected_items": alerts,
                "total_affected_items": hits.get("total", {}).get("value", len(alerts)),
                "total_failed_items": 0,
                "failed_items": []
            }
        }

    # ------------------------------------------------------------------
    # HTTP TRAFFIC ANALYSIS
    # ------------------------------------------------------------------

    async def get_http_analysis(self, time_range: str = "24h", limit: int = 10) -> Dict[str, Any]:
        """Get HTTP traffic analysis aggregations."""
        query = {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
                    {"term": {self.event_type_field: "http"}}
                ]
            }
        }


        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "top_urls": {
                    "terms": {"field": "http.url.keyword", "size": limit}
                },
                "top_methods": {
                    "terms": {"field": "http.http_method.keyword", "size": 10}
                },
                "top_status_codes": {
                    "terms": {"field": "http.status", "size": 20}
                },
                "top_user_agents": {
                    "terms": {"field": "http.http_user_agent.keyword", "size": limit}
                },
                "top_hostnames": {
                    "terms": {"field": "http.hostname.keyword", "size": limit}
                },
                "top_content_types": {
                    "terms": {"field": "http.http_content_type.keyword", "size": 10}
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        return {
            "total_events": total,
            "time_range": time_range,
            "top_urls": [
                {"url": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_urls", {}).get("buckets", [])
            ],
            "top_methods": [
                {"method": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_methods", {}).get("buckets", [])
            ],
            "top_status_codes": [
                {"status_code": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_status_codes", {}).get("buckets", [])
            ],
            "top_user_agents": [
                {"user_agent": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_user_agents", {}).get("buckets", [])
            ],
            "top_hostnames": [
                {"hostname": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_hostnames", {}).get("buckets", [])
            ],
            "top_content_types": [
                {"content_type": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_content_types", {}).get("buckets", [])
            ]
        }

    async def search_http_events(
        self,
        time_range: str = "24h",
        limit: int = 100,
        url: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        user_agent: Optional[str] = None,
        hostname: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search HTTP events with optional filters."""
        filter_clauses = [
            {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            {"term": {self.event_type_field: "http"}}
        ]

        if method:
            filter_clauses.append({"term": {"http.http_method.keyword": method.upper()}})
        if status_code:
            filter_clauses.append({"term": {"http.status": status_code}})
        if hostname:
            filter_clauses.append({"term": {"http.hostname.keyword": hostname}})

        must_clauses = []
        if url:
            must_clauses.append({"match_phrase": {"http.url": url}})
        if user_agent:
            must_clauses.append({"match_phrase": {"http.http_user_agent": user_agent}})

        query = {"bool": {"filter": filter_clauses}}
        if must_clauses:
            query["bool"]["must"] = must_clauses

        body = {
            "query": query,
            "size": limit,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        hits = response.get("hits", {})
        events = [hit.get("_source", {}) for hit in hits.get("hits", [])]

        return {
            "data": {
                "affected_items": events,
                "total_affected_items": hits.get("total", {}).get("value", len(events)),
                "total_failed_items": 0,
                "failed_items": []
            }
        }

    # ------------------------------------------------------------------
    # TLS/SSL ANALYSIS
    # ------------------------------------------------------------------

    async def get_tls_analysis(self, time_range: str = "24h", limit: int = 10) -> Dict[str, Any]:
        """Get TLS analysis with JA3/JA3S/JA4 fingerprints."""
        query = {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
                    {"term": {self.event_type_field: "tls"}}
                ]
            }
        }

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "tls_versions": {
                    "terms": {"field": "tls.version.keyword", "size": 10}
                },
                "top_ja3": {
                    "terms": {"field": "tls.ja3.hash.keyword", "size": limit}
                },
                "top_ja3s": {
                    "terms": {"field": "tls.ja3s.hash.keyword", "size": limit}
                },
                "top_ja4": {
                    "terms": {"field": "tls.ja4.keyword", "size": limit}
                },
                "top_dest_services": {
                    "terms": {"field": "service_name.keyword", "size": limit}
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        return {
            "total_events": total,
            "time_range": time_range,
            "tls_versions": [
                {"version": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("tls_versions", {}).get("buckets", [])
            ],
            "top_ja3": [
                {"hash": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_ja3", {}).get("buckets", [])
            ],
            "top_ja3s": [
                {"hash": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_ja3s", {}).get("buckets", [])
            ],
            "top_ja4": [
                {"fingerprint": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_ja4", {}).get("buckets", [])
            ],
            "top_dest_services": [
                {"service": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_dest_services", {}).get("buckets", [])
            ]
        }

    # ------------------------------------------------------------------
    # MITRE ATT&CK ANALYSIS
    # ------------------------------------------------------------------

    async def get_mitre_analysis(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get MITRE ATT&CK analysis from Suricata alert metadata."""
        query = {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
                    {"exists": {"field": "alert.metadata.mitre_tactic_id"}}
                ]
            }
        }

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "tactics": {
                    "terms": {"field": "alert.metadata.mitre_tactic_name.keyword", "size": 20}
                },
                "tactic_ids": {
                    "terms": {"field": "alert.metadata.mitre_tactic_id.keyword", "size": 20}
                },
                "techniques": {
                    "terms": {"field": "alert.metadata.mitre_technique_name.keyword", "size": 50}
                },
                "technique_ids": {
                    "terms": {"field": "alert.metadata.mitre_technique_id.keyword", "size": 50}
                },
                "severity_breakdown": {
                    "terms": {"field": "alert.severity", "size": 10}
                },
                "top_signatures": {
                    "terms": {"field": "alert.signature.keyword", "size": 20}
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        # Build tactic-ID mapping
        tactic_names = {b.get("key"): b.get("doc_count") for b in aggs.get("tactics", {}).get("buckets", [])}
        tactic_ids = {b.get("key"): b.get("doc_count") for b in aggs.get("tactic_ids", {}).get("buckets", [])}

        tactics = []
        for name, count in tactic_names.items():
            tactics.append({"tactic_name": name, "count": count})

        technique_names = {b.get("key"): b.get("doc_count") for b in aggs.get("techniques", {}).get("buckets", [])}
        technique_id_map = {b.get("key"): b.get("doc_count") for b in aggs.get("technique_ids", {}).get("buckets", [])}

        techniques = []
        for name, count in technique_names.items():
            techniques.append({"technique_name": name, "count": count})

        return {
            "total_mitre_alerts": total,
            "time_range": time_range,
            "tactics": tactics,
            "tactic_ids": [
                {"tactic_id": tid, "count": cnt}
                for tid, cnt in tactic_ids.items()
            ],
            "techniques": techniques,
            "technique_ids": [
                {"technique_id": tid, "count": cnt}
                for tid, cnt in technique_id_map.items()
            ],
            "severity_breakdown": {
                str(b.get("key")): b.get("doc_count", 0)
                for b in aggs.get("severity_breakdown", {}).get("buckets", [])
            },
            "top_signatures": [
                {"signature": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_signatures", {}).get("buckets", [])
            ]
        }

    # ------------------------------------------------------------------
    # JA3 FINGERPRINT ANALYSIS
    # ------------------------------------------------------------------

    async def get_ja3_analysis(
        self,
        time_range: str = "24h",
        limit: int = 20,
        ja3_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get JA3 fingerprint deep analysis with associated IPs."""
        filter_clauses = [
            {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            {"term": {self.event_type_field: "tls"}}
        ]
        if ja3_hash:
            filter_clauses.append({"term": {"tls.ja3.hash.keyword": ja3_hash}})

        query = {"bool": {"filter": filter_clauses}}

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "ja3_fingerprints": {
                    "terms": {"field": "tls.ja3.hash.keyword", "size": limit},
                    "aggs": {
                        "top_src_ips": {
                            "terms": {"field": "src_ip.keyword", "size": 5}
                        },
                        "top_dest_ips": {
                            "terms": {"field": "dest_ip.keyword", "size": 5}
                        },
                        "top_services": {
                            "terms": {"field": "service_name.keyword", "size": 5}
                        },
                        "tls_versions": {
                            "terms": {"field": "tls.version.keyword", "size": 5}
                        }
                    }
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        fingerprints = []
        for bucket in aggs.get("ja3_fingerprints", {}).get("buckets", []):
            fingerprints.append({
                "ja3_hash": bucket.get("key"),
                "count": bucket.get("doc_count"),
                "top_src_ips": [
                    {"ip": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_src_ips", {}).get("buckets", [])
                ],
                "top_dest_ips": [
                    {"ip": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_dest_ips", {}).get("buckets", [])
                ],
                "top_services": [
                    {"service": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_services", {}).get("buckets", [])
                ],
                "tls_versions": [
                    {"version": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("tls_versions", {}).get("buckets", [])
                ]
            })

        return {
            "total_tls_events": total,
            "time_range": time_range,
            "fingerprints": fingerprints
        }

    # ------------------------------------------------------------------
    # JA4 FINGERPRINT DEEP ANALYSIS (Pallas 3.3)
    # ------------------------------------------------------------------

    async def get_ja4_analysis(
        self,
        time_range: str = "24h",
        limit: int = 20,
        ja4_fingerprint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get JA4 fingerprint deep analysis with associated IPs and services."""
        filter_clauses = [
            {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            {"term": {self.event_type_field: "tls"}},
            {"exists": {"field": "tls.ja4"}}
        ]
        if ja4_fingerprint:
            filter_clauses.append({"term": {"tls.ja4.keyword": ja4_fingerprint}})

        query = {"bool": {"filter": filter_clauses}}

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "ja4_fingerprints": {
                    "terms": {"field": "tls.ja4.keyword", "size": limit},
                    "aggs": {
                        "top_src_ips": {
                            "terms": {"field": "src_ip.keyword", "size": 5}
                        },
                        "top_dest_ips": {
                            "terms": {"field": "dest_ip.keyword", "size": 5}
                        },
                        "top_services": {
                            "terms": {"field": "service_name.keyword", "size": 5}
                        },
                        "tls_versions": {
                            "terms": {"field": "tls.version.keyword", "size": 5}
                        }
                    }
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        fingerprints = []
        for bucket in aggs.get("ja4_fingerprints", {}).get("buckets", []):
            fingerprints.append({
                "ja4_fingerprint": bucket.get("key"),
                "count": bucket.get("doc_count"),
                "top_src_ips": [
                    {"ip": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_src_ips", {}).get("buckets", [])
                ],
                "top_dest_ips": [
                    {"ip": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_dest_ips", {}).get("buckets", [])
                ],
                "top_services": [
                    {"service": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("top_services", {}).get("buckets", [])
                ],
                "tls_versions": [
                    {"version": b.get("key"), "count": b.get("doc_count")}
                    for b in bucket.get("tls_versions", {}).get("buckets", [])
                ]
            })

        return {
            "total_tls_events": total,
            "time_range": time_range,
            "fingerprints": fingerprints
        }

    # ------------------------------------------------------------------
    # FLOW / CONVERSATION ANALYSIS (Pallas 3.3)
    # ------------------------------------------------------------------

    async def get_flow_analysis(
        self,
        time_range: str = "24h",
        limit: int = 20,
        src_ip: Optional[str] = None,
        dest_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze network flow/conversation patterns between IP pairs."""
        filter_clauses = [
            {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}},
            {"term": {self.event_type_field: "alert"}}
        ]
        if src_ip:
            filter_clauses.append({"term": {"src_ip.keyword": src_ip}})
        if dest_ip:
            filter_clauses.append({"term": {"dest_ip.keyword": dest_ip}})

        query = {"bool": {"filter": filter_clauses}}

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "conversations": {
                    "composite": {
                        "size": limit,
                        "sources": [
                            {"src_ip": {"terms": {"field": "src_ip.keyword"}}},
                            {"dest_ip": {"terms": {"field": "dest_ip.keyword"}}}
                        ]
                    },
                    "aggs": {
                        "alert_count": {"value_count": {"field": "@timestamp"}},
                        "unique_signatures": {"cardinality": {"field": "alert.signature.keyword"}},
                        "top_signatures": {"terms": {"field": "alert.signature.keyword", "size": 3}},
                        "min_severity": {"min": {"field": "alert.severity"}},
                        "protocols": {"terms": {"field": "proto.keyword", "size": 5}},
                        "dest_ports": {"terms": {"field": "dest_port", "size": 5}}
                    }
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        conversations = []
        for bucket in aggs.get("conversations", {}).get("buckets", []):
            key = bucket.get("key", {})
            conversations.append({
                "src_ip": key.get("src_ip"),
                "dest_ip": key.get("dest_ip"),
                "alert_count": bucket.get("alert_count", {}).get("value", 0),
                "unique_signatures": bucket.get("unique_signatures", {}).get("value", 0),
                "top_signatures": [
                    b.get("key") for b in bucket.get("top_signatures", {}).get("buckets", [])
                ],
                "min_severity": bucket.get("min_severity", {}).get("value"),
                "protocols": [
                    b.get("key") for b in bucket.get("protocols", {}).get("buckets", [])
                ],
                "dest_ports": [
                    b.get("key") for b in bucket.get("dest_ports", {}).get("buckets", [])
                ]
            })

        # Sort by alert count descending
        conversations.sort(key=lambda x: x["alert_count"], reverse=True)

        return {
            "total_alerts": total,
            "time_range": time_range,
            "conversations": conversations,
            "unique_conversations": len(conversations)
        }

    # ------------------------------------------------------------------
    # SUSPICIOUS ACTIVITY DETECTION
    # ------------------------------------------------------------------

    async def get_suspicious_activity(self, time_range: str = "24h") -> Dict[str, Any]:
        """Detect suspicious HTTP patterns and TLS anomalies."""
        time_filter = {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}}

        # Query 1: Scanner-like user agents
        scanner_ua_body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [time_filter, {"term": {self.event_type_field: "http"}}],
                    "should": [
                        {"match_phrase": {"http.http_user_agent": "curl"}},
                        {"match_phrase": {"http.http_user_agent": "wget"}},
                        {"match_phrase": {"http.http_user_agent": "python-requests"}},
                        {"match_phrase": {"http.http_user_agent": "nikto"}},
                        {"match_phrase": {"http.http_user_agent": "nmap"}},
                        {"match_phrase": {"http.http_user_agent": "sqlmap"}},
                        {"match_phrase": {"http.http_user_agent": "gobuster"}},
                        {"match_phrase": {"http.http_user_agent": "dirbuster"}},
                        {"match_phrase": {"http.http_user_agent": "masscan"}},
                        {"match_phrase": {"http.http_user_agent": "zgrab"}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "suspicious_agents": {
                    "terms": {"field": "http.http_user_agent.keyword", "size": 20},
                    "aggs": {
                        "src_ips": {"terms": {"field": "src_ip.keyword", "size": 5}}
                    }
                }
            }
        }

        # Query 2: Legacy TLS versions
        legacy_tls_body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [time_filter, {"term": {self.event_type_field: "tls"}}],
                    "should": [
                        {"term": {"tls.version.keyword": "TLS 1.0"}},
                        {"term": {"tls.version.keyword": "TLS 1.1"}},
                        {"term": {"tls.version.keyword": "SSLv3"}},
                        {"term": {"tls.version.keyword": "SSLv2"}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "legacy_versions": {
                    "terms": {"field": "tls.version.keyword", "size": 10},
                    "aggs": {
                        "src_ips": {"terms": {"field": "src_ip.keyword", "size": 5}},
                        "dest_ips": {"terms": {"field": "dest_ip.keyword", "size": 5}}
                    }
                }
            }
        }

        # Query 3: Unusual HTTP methods
        unusual_methods_body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [time_filter, {"term": {self.event_type_field: "http"}}],
                    "should": [
                        {"term": {"http.http_method.keyword": "DELETE"}},
                        {"term": {"http.http_method.keyword": "PATCH"}},
                        {"term": {"http.http_method.keyword": "OPTIONS"}},
                        {"term": {"http.http_method.keyword": "TRACE"}},
                        {"term": {"http.http_method.keyword": "CONNECT"}},
                        {"term": {"http.http_method.keyword": "PUT"}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "methods": {
                    "terms": {"field": "http.http_method.keyword", "size": 10},
                    "aggs": {
                        "top_urls": {"terms": {"field": "http.url.keyword", "size": 5}},
                        "src_ips": {"terms": {"field": "src_ip.keyword", "size": 5}}
                    }
                }
            }
        }

        # Query 4: HTTP error responses (4xx, 5xx)
        http_errors_body = {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        time_filter,
                        {"term": {self.event_type_field: "http"}},
                        {"range": {"http.status": {"gte": 400}}}
                    ]
                }
            },
            "aggs": {
                "error_codes": {
                    "terms": {"field": "http.status", "size": 20},
                    "aggs": {
                        "src_ips": {"terms": {"field": "src_ip.keyword", "size": 5}}
                    }
                }
            }
        }

        # Execute all queries
        results = await asyncio.gather(
            self._request("POST", f"/{self.index_pattern}/_search", json=scanner_ua_body),
            self._request("POST", f"/{self.index_pattern}/_search", json=legacy_tls_body),
            self._request("POST", f"/{self.index_pattern}/_search", json=unusual_methods_body),
            self._request("POST", f"/{self.index_pattern}/_search", json=http_errors_body),
            return_exceptions=True
        )

        # Process scanner user agents
        suspicious_user_agents = []
        if not isinstance(results[0], Exception):
            for b in results[0].get("aggregations", {}).get("suspicious_agents", {}).get("buckets", []):
                suspicious_user_agents.append({
                    "user_agent": b.get("key"),
                    "count": b.get("doc_count"),
                    "src_ips": [s.get("key") for s in b.get("src_ips", {}).get("buckets", [])]
                })

        # Process legacy TLS
        legacy_tls = []
        if not isinstance(results[1], Exception):
            for b in results[1].get("aggregations", {}).get("legacy_versions", {}).get("buckets", []):
                legacy_tls.append({
                    "version": b.get("key"),
                    "count": b.get("doc_count"),
                    "src_ips": [s.get("key") for s in b.get("src_ips", {}).get("buckets", [])],
                    "dest_ips": [s.get("key") for s in b.get("dest_ips", {}).get("buckets", [])]
                })

        # Process unusual methods
        unusual_http_methods = []
        if not isinstance(results[2], Exception):
            for b in results[2].get("aggregations", {}).get("methods", {}).get("buckets", []):
                unusual_http_methods.append({
                    "method": b.get("key"),
                    "count": b.get("doc_count"),
                    "top_urls": [s.get("key") for s in b.get("top_urls", {}).get("buckets", [])],
                    "src_ips": [s.get("key") for s in b.get("src_ips", {}).get("buckets", [])]
                })

        # Process HTTP errors
        http_errors = []
        if not isinstance(results[3], Exception):
            for b in results[3].get("aggregations", {}).get("error_codes", {}).get("buckets", []):
                http_errors.append({
                    "status_code": b.get("key"),
                    "count": b.get("doc_count"),
                    "src_ips": [s.get("key") for s in b.get("src_ips", {}).get("buckets", [])]
                })

        return {
            "time_range": time_range,
            "suspicious_user_agents": suspicious_user_agents,
            "legacy_tls": legacy_tls,
            "unusual_http_methods": unusual_http_methods,
            "http_errors": http_errors
        }

    # ------------------------------------------------------------------
    # TRAFFIC OVERVIEW
    # ------------------------------------------------------------------

    async def get_traffic_overview(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get overall traffic distribution by event type, protocol, and service."""
        query = {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gte": f"now-{time_range}", "lte": "now"}}}
                ]
            }
        }

        body = {
            "size": 0,
            "query": query,
            "aggs": {
                "event_types": {
                    "terms": {"field": self.event_type_field, "size": 20}
                },
                "protocols": {
                    "terms": {"field": "proto.keyword", "size": 10}
                },
                "services": {
                    "terms": {"field": "service_name.keyword", "size": 20}
                },
                "traffic_locality": {
                    "terms": {"field": "traffic_locality.keyword", "size": 5}
                },
                "top_interfaces": {
                    "terms": {"field": "in_iface.keyword", "size": 5}
                }
            },
            "track_total_hits": True
        }

        response = await self._request(
            "POST", f"/{self.index_pattern}/_search", json=body
        )

        aggs = response.get("aggregations", {})
        total = response.get("hits", {}).get("total", {}).get("value", 0)

        return {
            "total_events": total,
            "time_range": time_range,
            "event_types": [
                {"type": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("event_types", {}).get("buckets", [])
            ],
            "protocols": [
                {"protocol": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("protocols", {}).get("buckets", [])
            ],
            "services": [
                {"service": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("services", {}).get("buckets", [])
            ],
            "traffic_locality": [
                {"locality": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("traffic_locality", {}).get("buckets", [])
            ],
            "interfaces": [
                {"interface": b.get("key"), "count": b.get("doc_count")}
                for b in aggs.get("top_interfaces", {}).get("buckets", [])
            ]
        }

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    async def validate_connection(self) -> Dict[str, Any]:
        """Validate connection to Suricata Elasticsearch."""
        try:
            health = await self.health_check()
            status = "connected" if health.get("status") in ["green", "yellow"] else "degraded"
        except Exception as e:
            status = "failed"
            health = {"error": str(e)}

        return {
            "status": status,
            "cluster_health": health.get("status", "unknown"),
            "cluster_name": health.get("cluster_name", "unknown"),
            "index_pattern": self.index_pattern
        }

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
