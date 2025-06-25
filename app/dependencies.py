"""
Dependency Utilities for FastAPI RAG System
===========================================

This module provides reusable dependency functions for the FastAPI application, including rate limiting and client IP extraction.

Key Responsibilities:
- Enforce per-client rate limiting to protect the API from abuse.
- Provide utility functions for extracting client IP addresses in a proxy-aware manner.

Usage:
Import and use these dependencies in FastAPI endpoint definitions via the `Depends` mechanism.
"""

from time import time
from collections import defaultdict
from fastapi import Request, HTTPException
from typing import Dict, List

# In-memory record of request timestamps per client IP
rate_limit_records: Dict[str, List[float]] = defaultdict(list)

def get_client_ip(request: Request) -> str:
    """
    Extract Client IP Address
    ------------------------

    Returns the real client IP address, accounting for proxy headers if present.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        str: The client IP address.
    """
    if x_forwarded_for := request.headers.get("x-forwarded-for"):
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host

async def rate_limiter(request: Request):
    """
    Rate Limiter Dependency
    ----------------------

    Enforces a simple in-memory rate limit per client IP address.
    Raises HTTP 429 if the client exceeds the allowed request rate.

    Args:
        request (Request): The FastAPI request object.

    Raises:
        HTTPException: If the client exceeds the allowed rate.
    """
    client_ip = get_client_ip(request)
    settings = request.app.state.settings
    now = time()
    rate_limit_records[client_ip] = [t for t in rate_limit_records[client_ip] if t > now - settings.RATE_LIMIT_TIMEFRAME_SECONDS]
    if len(rate_limit_records[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    rate_limit_records[client_ip].append(now)