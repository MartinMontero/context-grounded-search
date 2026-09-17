"""SSRF-hardened outbound fetching.

Defences, in order:
1. Scheme allow-list (http/https only), no userinfo, port allow-list.
2. Hostname deny-list (localhost, .local, .internal, ...).
3. DNS resolution *before* connecting; every A/AAAA record must be public
   (private, loopback, link-local, CGNAT, multicast, reserved, IPv4-mapped IPv6
   and unique-local ranges are all rejected).
4. The connection is pinned to the validated IP (Host header + SNI carry the
   original name), so DNS rebinding between check and connect is impossible.
5. Redirects are followed manually and every hop repeats steps 1-4.
6. Response size is capped while streaming; content-type is allow-listed;
   timeouts bound every phase.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from rag_common.errors import ForbiddenTargetError, PayloadTooLargeError, UpstreamError

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Awaitable[list[IPAddress]]]

ALLOWED_SCHEMES = frozenset({"http", "https"})
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".home", ".lan", ".corp")
BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal", "instance-data"})
_EXTRA_BLOCKED_NETWORKS = [
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",  # "this" network
        "100.64.0.0/10",  # carrier-grade NAT
        "192.0.0.0/24",  # IETF protocol assignments
        "198.18.0.0/15",  # benchmarking
        "240.0.0.0/4",  # reserved
        "64:ff9b::/96",  # NAT64
        "2001:db8::/32",  # documentation
    )
]
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "message/rfc822")


def is_blocked_ip(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or getattr(ip, "is_site_local", False)
    ):
        return True
    return any(ip in net for net in _EXTRA_BLOCKED_NETWORKS)


@dataclass(frozen=True)
class ResolvedTarget:
    url: str
    scheme: str
    host: str
    port: int
    ip: IPAddress

    @property
    def pinned_url(self) -> str:
        """The same URL with the host replaced by the validated IP literal."""
        parts = urlsplit(self.url)
        ip_literal = f"[{self.ip}]" if self.ip.version == 6 else str(self.ip)
        netloc = (
            ip_literal if self.port == _default_port(self.scheme) else f"{ip_literal}:{self.port}"
        )
        return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))

    @property
    def host_header(self) -> str:
        return self.host if self.port == _default_port(self.scheme) else f"{self.host}:{self.port}"


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


async def system_resolver(host: str, port: int) -> list[IPAddress]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


async def validate_target_url(
    url: str,
    *,
    allowed_ports: list[int],
    resolver: Resolver = system_resolver,
) -> ResolvedTarget:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ForbiddenTargetError(f"scheme not allowed: {scheme or 'missing'}")
    if parts.username or parts.password:
        raise ForbiddenTargetError("credentials in URL are not allowed")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise ForbiddenTargetError("missing host")
    try:
        port = parts.port or _default_port(scheme)
    except ValueError as exc:
        raise ForbiddenTargetError("invalid port") from exc
    if port not in allowed_ports:
        raise ForbiddenTargetError(f"port not allowed: {port}")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise ForbiddenTargetError(f"host not allowed: {host}")

    try:
        literal: IPAddress | None = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [literal]
    else:
        try:
            addresses = await resolver(host, port)
        except (socket.gaierror, OSError) as exc:
            raise ForbiddenTargetError(f"dns resolution failed for {host}") from exc
        if not addresses:
            raise ForbiddenTargetError(f"dns resolution returned no addresses for {host}")
    for ip in addresses:  # every record must be public: no "one public, one private" tricks
        if is_blocked_ip(ip):
            raise ForbiddenTargetError(
                "target resolves to a non-public address", details={"host": host, "ip": str(ip)}
            )
    return ResolvedTarget(url=url.strip(), scheme=scheme, host=host, port=port, ip=addresses[0])


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    redirects: int


class SafeFetcher:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        max_bytes: int,
        max_redirects: int,
        allowed_ports: list[int],
        user_agent: str,
        resolver: Resolver = system_resolver,
    ) -> None:
        self.client = client
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.allowed_ports = allowed_ports
        self.user_agent = user_agent
        self.resolver = resolver

    async def fetch(self, url: str) -> FetchResult:
        current = url
        for hop in range(self.max_redirects + 1):
            target = await validate_target_url(
                current, allowed_ports=self.allowed_ports, resolver=self.resolver
            )
            request = self.client.build_request(
                "GET",
                target.pinned_url,
                headers={
                    "Host": target.host_header,
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
                    "Accept-Encoding": "gzip",
                },
                extensions={"sni_hostname": target.host} if target.scheme == "https" else {},
            )
            try:
                response = await self.client.send(request, stream=True, follow_redirects=False)
            except httpx.HTTPError as exc:
                raise UpstreamError(f"fetch failed: {type(exc).__name__}") from exc
            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise UpstreamError("redirect without Location header")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise UpstreamError(f"upstream returned HTTP {response.status_code}")
                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if content_type and not content_type.startswith(ALLOWED_CONTENT_TYPES):
                    raise ForbiddenTargetError(f"content-type not allowed: {content_type}")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self.max_bytes:
                    raise PayloadTooLargeError(
                        f"content-length {declared} exceeds {self.max_bytes}"
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        raise PayloadTooLargeError(f"response exceeds {self.max_bytes} bytes")
                return FetchResult(
                    url=url,
                    final_url=current,
                    status_code=response.status_code,
                    content_type=content_type or "application/octet-stream",
                    body=bytes(body),
                    redirects=hop,
                )
            finally:
                await response.aclose()
        raise ForbiddenTargetError(f"too many redirects (> {self.max_redirects})")
