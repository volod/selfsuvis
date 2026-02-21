import ipaddress
import socket
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests

from pipeline.config import settings


def _iter_resolved_ips(host: str) -> Iterable[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            out.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    return out


def _is_ip_allowed(ip: ipaddress._BaseAddress) -> bool:
    if settings.ALLOW_PRIVATE_URLS:
        return True
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    return True


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("URL hostname required")
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain credentials")

    host = parsed.hostname
    # Reject single-label hostnames unless explicitly allowed.
    if "." not in host and not host.isdigit() and not settings.ALLOW_PRIVATE_URLS:
        raise ValueError("Hostname not allowed")

    ips = list(_iter_resolved_ips(host))
    if not ips:
        raise ValueError("Hostname could not be resolved")
    for ip in ips:
        if not _is_ip_allowed(ip):
            raise ValueError("Target IP not allowed")


def safe_request(
    method: str,
    url: str,
    *,
    timeout: int,
    max_redirects: Optional[int] = None,
    stream: bool = False,
    **kwargs,
) -> requests.Response:
    max_redirects = settings.MAX_REDIRECTS if max_redirects is None else max_redirects
    session = requests.Session()
    current = url
    for _ in range(max_redirects + 1):
        validate_url(current)
        resp = session.request(
            method,
            current,
            timeout=timeout,
            allow_redirects=False,
            stream=stream,
            **kwargs,
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise ValueError("Redirect without location")
            current = urljoin(current, location)
            continue
        return resp
    raise ValueError("Too many redirects")
