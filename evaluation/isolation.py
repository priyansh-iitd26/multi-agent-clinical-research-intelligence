from typing import Any, List, Dict
import functools

def offline_guard(func):
    """
    Decorator to ensure no network calls are made during evaluation.
    In a real implementation, this would wrap socket calls or use a proxy.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Implementation: check if any network requests are attempted
        # and raise an OfflineViolationError.
        return func(*args, **kwargs)
    return wrapper

class OfflineViolationError(Exception):
    """Raised when a forbidden external dependency is accessed."""
    pass
