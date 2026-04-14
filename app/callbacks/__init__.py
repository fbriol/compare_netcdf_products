"""Callbacks package for the Dash application.

Sub-modules:
    views          View builder functions (breadcrumb, group browser,
                   detail view, global summary, geo map).
    _registration  Callback registration (register_callbacks).
"""

from ._registration import register_callbacks

__all__ = ["register_callbacks"]
