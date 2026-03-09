from __future__ import annotations

import ipaddress

from fastapi import Request

from app.core.config import settings

# This module helps the app figure out the real user IP address
# when another server is standing in front of the app.
# Avoids trusting fake IP headers from untrusted clients!!!

def _is_ip_in_trusted_networks(host: str) -> bool:
    """
    Check if IP belongs to trusty proxy from my config.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False

    for item in settings.TRUSTED_PROXIES:
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError:
            continue
        if ip in network:
            return True

    return False


def is_trusted_proxy(host: str | None) -> bool:
    if not host:
        return False

    normalized = host.strip()
    if not normalized:
        return False

    if normalized in settings.TRUSTED_PROXIES:
        return True

    return _is_ip_in_trusted_networks(normalized)


def get_client_ip(request: Request) -> str:
    """
    This returns the safest best guess of the real client IP.
    """

    # direct_host - IP of whatever connected directly to my app
    #             - browser IP or reverse proxy IP
    direct_host = request.client.host if request.client else None
    if not direct_host:
        return "unknown"

    if not is_trusted_proxy(direct_host):
        return direct_host

    # If connected through trusted proxy, inspect real IP and return
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        parts = [part.strip() for part in forwarded_for.split(",") if part.strip()]
        if parts:
            return parts[0]

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    return direct_host
