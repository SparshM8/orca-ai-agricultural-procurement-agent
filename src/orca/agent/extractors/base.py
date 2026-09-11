"""Base interface and timing normalization for conversational offer extractors."""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from orca.domain.schemas import ExtractedOffer


def parse_pickup_timing(timing_str: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse and normalize human-readable pickup timing strings into (datetime, normalized_str).

    Handles:
    - 'tomorrow at 10 AM' -> (datetime, 'tomorrow at 10:00 AM')
    - 'Friday at 2 PM' -> (datetime, 'Friday at 2:00 PM')
    - 'tomorrow' -> (datetime, 'tomorrow')
    - 'this Friday at 2 PM' -> (datetime, 'this Friday at 2:00 PM')
    """
    if not timing_str:
        return None, None

    # 1. Normalize casing of days of week
    days_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    normalized_str = timing_str.strip()
    for d in days_map:
        normalized_str = re.sub(rf'\b{d}\b', d.capitalize(), normalized_str, flags=re.IGNORECASE)

    # 2. Normalize time component to H:MM AM/PM
    time_match = re.search(r'(?i)\bat\s+(?P<hour>\d{1,2})(?::(?P<mins>\d{2}))?\s*(?P<meridiem>am|pm)?\b', normalized_str)

    hour = None
    mins = 0
    if time_match:
        h = int(time_match.group('hour'))
        mins = int(time_match.group('mins') or 0)
        meridiem = (time_match.group('meridiem') or '').upper()
        if meridiem == 'PM' and h < 12:
            hour = h + 12
        elif meridiem == 'AM' and h == 12:
            hour = 0
        else:
            hour = h

        # Format time consistently as 'at H:MM AM/PM'
        norm_time = f'at {h}:{mins:02d} {meridiem}'.strip()
        normalized_str = normalized_str[:time_match.start()] + norm_time + normalized_str[time_match.end():]
        normalized_str = re.sub(r'\s+', ' ', normalized_str).strip()

    # 3. Calculate datetime object
    now = datetime.now(timezone.utc)
    target_dt = None
    lower_norm = normalized_str.lower()

    if 'tomorrow' in lower_norm:
        base_date = now + timedelta(days=1)
        h = hour if hour is not None else 9
        target_dt = base_date.replace(hour=h, minute=mins, second=0, microsecond=0)
    elif 'today' in lower_norm:
        h = hour if hour is not None else 17
        target_dt = now.replace(hour=h, minute=mins, second=0, microsecond=0)
    else:
        for d_name, d_idx in days_map.items():
            if d_name in lower_norm:
                days_ahead = (d_idx - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                base_date = now + timedelta(days=days_ahead)
                h = hour if hour is not None else 9
                target_dt = base_date.replace(hour=h, minute=mins, second=0, microsecond=0)
                break

    return target_dt, normalized_str


class BaseOfferExtractor(ABC):
    """Abstract interface for natural-language offer extraction.
    
    Allows swapping in AI providers (OpenAI-compatible, Ollama, local models)
    or deterministic rule-based pattern extractors.
    """

    @abstractmethod
    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Extract structured offer attributes from natural language message.
        
        Args:
            text: Inbound natural language message from farmer.
            current_offer: Previously accumulated offer details for context preservation.
            
        Returns:
            ExtractedOffer containing updated extracted fields.
        """
        pass
