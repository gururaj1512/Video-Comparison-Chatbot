"""
URL parsing and validation utilities.

Detects platform (YouTube vs Instagram) and extracts video IDs
from various URL formats.
"""

import re
from typing import Optional, Tuple
from enum import Enum


class Platform(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    UNKNOWN = "unknown"


# YouTube URL patterns
# Supports: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID
YOUTUBE_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]

# Instagram URL patterns
# Supports: instagram.com/reel/ID, instagram.com/reels/ID, instagram.com/p/ID
INSTAGRAM_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/(?:reel|reels|p)/([a-zA-Z0-9_-]+)"),
]


def detect_platform(url: str) -> Platform:
    url_lower = url.lower()

    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return Platform.YOUTUBE
    elif "instagram.com" in url_lower:
        return Platform.INSTAGRAM

    return Platform.UNKNOWN


def extract_video_id(url: str) -> Optional[str]:
    platform = detect_platform(url)

    if platform == Platform.YOUTUBE:
        for pattern in YOUTUBE_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

    elif platform == Platform.INSTAGRAM:
        for pattern in INSTAGRAM_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

    return None


def parse_url(url: str) -> Tuple[Platform, Optional[str]]:
    platform = detect_platform(url)
    video_id = extract_video_id(url)
    return platform, video_id


def validate_video_url(url: str, expected_platform: Optional[Platform] = None) -> bool:
    platform, video_id = parse_url(url)

    if platform == Platform.UNKNOWN or video_id is None:
        return False

    if expected_platform and platform != expected_platform:
        return False

    return True
