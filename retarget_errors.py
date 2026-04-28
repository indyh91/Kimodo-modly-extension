"""Shared machine-readable retarget/export errors."""

from __future__ import annotations


class RetargetError(RuntimeError):
    """Runtime error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


MISSING_BVH = "MISSING_BVH"
UNPARSEABLE_BVH = "UNPARSEABLE_BVH"
UNPARSEABLE_RIG = "UNPARSEABLE_RIG"
RIG_HAS_NO_SKIN_OR_SKELETON = "RIG_HAS_NO_SKIN_OR_SKELETON"
MAPPING_INCOMPATIBLE = "MAPPING_INCOMPATIBLE"
EXPORT_FAILED = "EXPORT_FAILED"
RETARGET_UNAVAILABLE = "RETARGET_UNAVAILABLE"

# Legacy spec/design names kept as non-breaking aliases for consumers that
# integrated against the first SDD vocabulary. Runtime errors continue to use
# the more explicit primary names above, while metadata exposes these aliases.
BVH_UNAVAILABLE = MISSING_BVH
BVH_UNPARSEABLE = UNPARSEABLE_BVH
RIG_INVALID = "RIG_INVALID"

LEGACY_ERROR_ALIASES = {
    MISSING_BVH: ("BVH_UNAVAILABLE",),
    UNPARSEABLE_BVH: ("BVH_UNPARSEABLE",),
    UNPARSEABLE_RIG: (RIG_INVALID,),
    RIG_HAS_NO_SKIN_OR_SKELETON: (RIG_INVALID,),
}


def legacy_error_aliases(code: str) -> tuple[str, ...]:
    """Return older public names for a primary retarget error code."""

    return LEGACY_ERROR_ALIASES.get(code, ())
