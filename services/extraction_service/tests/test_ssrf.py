from __future__ import annotations

import ipaddress

import httpx
import pytest

from extraction_service.ssrf import SafeFetcher, is_blocked_ip, validate_target_url
from rag_common.errors import ForbiddenTargetError, PayloadTooLargeError, UpstreamError

PUBLIC = ipaddress.ip_address("93.184.216.34")
PUBLIC_V6 = ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946")
TABLE = {
    "public.example": [PUBLIC],
    "public6.example": [PUBLIC_V6],
    "mixed.example": [PUBLIC, ipaddress.ip_address("10.0.0.5")],
    "mapped.example": [ipaddress.ip_address("::ffff:192.168.1.10")],
    "cgnat.example": [ipaddress.ip_address("100.64.0.1")],
}


async def resolver(host: str, port: int) -> list:
    if host not in TABLE:
        raise OSError("nxdomain")
    return TABLE[host]


@pytest.mark.parametrize(
    "ip",
    [
        "10.1.2.3",
        "172.16.0.1",
        "192.168.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104
        "100.64.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "fd12::1",
        "::ffff:10.0.0.1",
        "64:ff9b::a00:1",
        "2001:db8::1",
    ],
)
def test_private_and_special_ranges_are_blocked(ip: str) -> None:
    assert is_blocked_ip(ipaddress.ip_address(ip))


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700::1111"])
def test_public_ips_allowed(ip: str) -> None:
    assert not is_blocked_ip(ipaddress.ip_address(ip))


async def test_validate_accepts_public_host_and_pins_ip() -> None:
    target = await validate_target_url(
        "https://public.example/path?q=1#frag", allowed_ports=[80, 443], resolver=resolver
    )
    assert (target.host, target.port, str(target.ip)) == ("public.example", 443, "93.184.216.34")
    assert target.pinned_url == "https://93.184.216.34/path?q=1"
    assert target.host_header == "public.example"
    v6 = await validate_target_url(
        "http://public6.example:80/", allowed_ports=[80], resolver=resolver
    )
    assert v6.pinned_url == f"http://[{PUBLIC_V6}]/"


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("ftp://public.example/x", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("http://user:pw@public.example/", "credentials"),
        ("http://public.example:22/", "port"),
        ("http://localhost/", "host not allowed"),
        ("http://metadata.google.internal/", "host not allowed"),
        ("http://svc.cluster.local/", "host not allowed"),
        ("http://127.0.0.1/", "non-public"),
        ("http://[::1]/", "non-public"),
        ("http://0x7f000001/", "dns"),
        ("http://mixed.example/", "non-public"),
        ("http://mapped.example/", "non-public"),
        ("http://cgnat.example/", "non-public"),
        ("http://does-not-exist.example/", "dns"),
    ],
)
async def test_validate_rejects(url: str, match: str) -> None:
    with pytest.raises(ForbiddenTargetError, match=match):
        await validate_target_url(url, allowed_ports=[80, 443], resolver=resolver)


def _fetcher(handler, max_bytes: int = 1000, max_redirects: int = 3) -> SafeFetcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SafeFetcher(
        client=client,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        allowed_ports=[80, 443],
        user_agent="test-agent",
        resolver=resolver,
    )


async def test_fetch_connects_to_pinned_ip_with_original_host_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, text="<p>hi</p>"
        )

    result = await _fetcher(handler).fetch("https://public.example/a")
    assert result.body == b"<p>hi</p>" and result.content_type == "text/html"
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["host"] == "public.example"
    assert seen[0].headers["user-agent"] == "test-agent"
    assert seen[0].extensions.get("sni_hostname") == "public.example"


async def test_fetch_revalidates_every_redirect_hop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://10.0.0.1/internal"})
        return httpx.Response(200, text="never")

    with pytest.raises(ForbiddenTargetError, match="non-public"):
        await _fetcher(handler).fetch("http://public.example/start")


async def test_fetch_redirect_limit_size_limit_and_content_type() -> None:
    def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "http://public.example/again"})

    with pytest.raises(ForbiddenTargetError, match="too many redirects"):
        await _fetcher(loop, max_redirects=2).fetch("http://public.example/")

    def big(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 5000)

    with pytest.raises(PayloadTooLargeError):
        await _fetcher(big, max_bytes=1000).fetch("http://public.example/")

    def binary(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b"\x00"
        )

    with pytest.raises(ForbiddenTargetError, match="content-type"):
        await _fetcher(binary).fetch("http://public.example/")

    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(UpstreamError, match="503"):
        await _fetcher(server_error).fetch("http://public.example/")
