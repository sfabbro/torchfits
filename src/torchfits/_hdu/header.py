"""FITS header as a dict-like mapping from keyword to value."""

from __future__ import annotations

from typing import Any

from ._repr import render_html_table
from .card import Card

# HISTORY / COMMENT carry no value: they legitimately repeat, and the mapping
# view tracks the most recent line (see .cards / get_comment / get_history for
# the lossless list).
_COMMENTARY_KEYS = frozenset({"HISTORY", "COMMENT"})


def _normalize_header_value(value: Any) -> Any:
    """Coerce numpy scalars / 0-d arrays to plain Python scalars.

    ponytail: handles np.generic via .item() so header writes never silently skip;
    ceiling is that exotic dtypes fall through to C++ which now raises.
    """
    if type(value) in (str, bool, int, float, bytes):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
    except Exception:
        pass
    return value


class Header(dict[str, Any]):
    def __init__(self, cards: Any = None) -> None:
        super().__init__()
        self._version = 0
        self._cards: list[Card] = []
        if cards:
            if isinstance(cards, Header):
                for card in cards.cards:
                    self._append_card(card, update_mapping=True, bump=False)
            elif isinstance(cards, dict):
                for k, v in cards.items():
                    if (
                        not isinstance(v, (str, bytes))
                        and isinstance(v, tuple)
                        and len(v) == 2
                    ):
                        value, comment = v
                    else:
                        value = v
                        comment = ""
                    self._set_card(str(k), value, str(comment), bump=False)
            elif isinstance(cards, (list, tuple)):
                # One path for tuples and Card-likes alike: the mapping update
                # goes through _set_mapping_for_card, so a header built here and
                # one built with add_comment/add_history agree on what the
                # mapping exposes.
                for card in cards:
                    if type(card) is tuple and len(card) == 3:
                        key, value, comment = card
                        parsed = Card(
                            str(key),
                            _normalize_header_value(value),
                            "" if comment is None else str(comment),
                        )
                    else:
                        parsed = self._coerce_card(card)
                    self._append_card(parsed, update_mapping=True, bump=False)

    def __setitem__(self, key: str, value: Any) -> None:
        if (
            not isinstance(value, (str, bytes))
            and isinstance(value, tuple)
            and len(value) == 2
        ):
            card_value, comment = value
        else:
            card_value = value
            comment = ""
        self._set_card(str(key), card_value, str(comment), bump=False)
        self._version += 1

    def __delitem__(self, key: str) -> None:
        # HISTORY/COMMENT (and any duplicated key) must clear all cards; a single
        # del left orphans in .cards after #225's remove_all fast path.
        self.remove(str(key), remove_all=True)

    def update(self, *args: Any, **kwargs: Any) -> None:
        other = dict(*args, **kwargs)
        for key, value in other.items():
            if (
                not isinstance(value, (str, bytes))
                and isinstance(value, tuple)
                and len(value) == 2
            ):
                card_value, comment = value
            else:
                card_value = value
                comment = ""
            self._set_card(str(key), card_value, str(comment), bump=False)
        if other:
            self._version += 1

    def clear(self) -> None:
        super().clear()
        self._cards.clear()
        self._version += 1

    def pop(self, *args: Any) -> Any:
        if not args:
            raise TypeError("pop expected at least 1 argument")
        key = str(args[0])
        if key not in self:
            if len(args) > 1:
                return args[1]
            raise KeyError(key)
        res = self[key]
        self.remove(key, remove_all=True)
        return res

    def popitem(self) -> tuple[str, Any]:
        key, value = super().popitem()
        key_s = str(key)
        # Mapping entry is gone; drop every card for that key (HISTORY/COMMENT
        # and any insert()-created duplicates).
        self._cards = [c for c in self._cards if c.key != key_s]
        self._version += 1
        return key_s, value

    def setdefault(self, key: str, default: Any = None) -> Any:
        key_s = str(key)
        if key_s in self:
            res = self[key_s]
        else:
            self._set_card(key_s, default, "", bump=False)
            res = default
        self._version += 1
        return res

    def add_history(self, value: Any) -> None:
        self._append_card(Card("HISTORY", str(value), ""), update_mapping=True)

    def add_comment(self, value: Any) -> None:
        self._append_card(Card("COMMENT", str(value), ""), update_mapping=True)

    def get_history(self) -> list[Any]:
        return [c[1] for c in self._cards if c[0] == "HISTORY"]

    def get_comment(self) -> list[Any]:
        return [c[1] for c in self._cards if c[0] == "COMMENT"]

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self._cards)

    def append(self, card: Card | tuple[str, Any] | tuple[str, Any, str]) -> None:
        self._append_card(self._coerce_card(card), update_mapping=True)

    def insert(
        self, index: int, card: Card | tuple[str, Any] | tuple[str, Any, str]
    ) -> None:
        parsed = self._coerce_card(card)
        self._cards.insert(int(index), parsed)
        # Rebuild rather than set: inserting ahead of existing cards changes
        # which occurrence is first, and first is what the mapping reports.
        self._rebuild_mapping_for_key(parsed.key)
        self._version += 1

    def remove(
        self,
        key: str,
        *,
        ignore_missing: bool = False,
        remove_all: bool = False,
    ) -> None:
        key_s = str(key)
        matches = [idx for idx, card in enumerate(self._cards) if card.key == key_s]
        if not matches:
            if ignore_missing:
                return
            raise KeyError(key)
        # remove_all rebuilds once (O(N)); repeated del is O(N*K) for huge HISTORY.
        if remove_all:
            self._cards = [c for c in self._cards if c.key != key_s]
        else:
            del self._cards[matches[0]]
        self._rebuild_mapping_for_key(key_s)
        self._version += 1

    def card(self, key: str) -> Card:
        key_s = str(key)
        for card in self._cards:
            if card.key == key_s:
                return card
        raise KeyError(key)

    def comments(self, key: str) -> list[str]:
        key_s = str(key)
        return [card.comment for card in self._cards if card.key == key_s]

    @staticmethod
    def _coerce_card(card: Card | tuple[str, Any] | tuple[str, Any, str]) -> Card:
        if isinstance(card, Card):
            # Normalize Card values that may hold numpy scalars.
            norm = _normalize_header_value(card.value)
            if norm is not card.value:
                return Card(card.key, norm, card.comment)
            return card
        if not isinstance(card, (list, tuple)):
            raise TypeError("card must be a Card or tuple")
        if len(card) == 3:
            key, value, comment = card
        elif len(card) == 2:
            key, value = card
            comment = ""
        else:
            raise ValueError("card tuples must have 2 or 3 items")
        return Card(
            str(key),
            _normalize_header_value(value),
            "" if comment is None else str(comment),
        )

    def _append_card(
        self, card: Card, *, update_mapping: bool, bump: bool = True
    ) -> None:
        self._cards.append(card)
        if update_mapping:
            self._set_mapping_for_card(card)
        if bump:
            self._version += 1

    def _set_card(self, key: str, value: Any, comment: str, *, bump: bool) -> None:
        value = _normalize_header_value(value)
        card = Card(key, value, comment)
        if key in _COMMENTARY_KEYS:
            self._append_card(card, update_mapping=True, bump=bump)
            return

        if key in self:
            for idx, existing in enumerate(self._cards):
                if existing.key == key:
                    self._cards[idx] = card
                    break
        else:
            self._cards.append(card)

        super().__setitem__(key, value)
        if bump:
            self._version += 1

    def _set_mapping_for_card(self, card: Card) -> None:
        if card.key in _COMMENTARY_KEYS:
            super().__setitem__(card.key, card.value)
            return
        # Value keywords resolve to the FIRST occurrence, matching CFITSIO
        # (fits_read_keyword) and astropy. A name-keyed map that kept the last
        # occurrence made read_header() and read_keys() disagree on the same
        # file, which is how a duplicated keyword silently changed meaning
        # depending on which API you asked.
        if card.key not in self:
            super().__setitem__(card.key, card.value)

    def _rebuild_mapping_for_key(self, key: str) -> None:
        remaining = [card for card in self._cards if card.key == key]
        if remaining:
            pick = remaining[-1] if key in _COMMENTARY_KEYS else remaining[0]
            super().__setitem__(key, pick.value)
        elif key in self:
            super().__delitem__(key)

    def _repr_html_(self) -> str:
        return render_html_table(
            "FITS Header",
            ["Keyword", "Value", "Comment"],
            (
                (card.key, card.value if card.value is not None else "", card.comment)
                for card in self._cards
            ),
            cell_extra=["", "", "opacity: 0.7;"],
            first_col_bold=True,
        )
