"""Bittensor transport and round orchestration for Witness."""

__all__ = ["WitnessTask"]


def __getattr__(name):
    # Process cancellation is also used by direct HTTP tools. Importing that
    # utility must not initialize chain machinery as a package side effect.
    if name == "WitnessTask":
        from .protocol import WitnessTask
        return WitnessTask
    raise AttributeError(name)
