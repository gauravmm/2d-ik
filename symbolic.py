#!python3


from typing import Any

from datamodel import Position


class IKSymbolic:
    """Implements a symbolic solver for inverse kinematics, making ."""

    def __call__(self, desired: Position) -> Any:
        pass
