from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import hypothesis.strategies as st
from hypothesis import given

if TYPE_CHECKING:
    pass


else:
    pass


@given(
    datetimes=st.lists(
        st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
        min_size=1,
        max_size=3,
    )
)
def test_replace_valid() -> None:
    """Replace with valid time components."""
