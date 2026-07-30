"""FITS header as a dict-like mapping from keyword to value."""

from __future__ import annotations

from typing import Any

from .card import Card


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
                append = self._cards.append
                setitem = super().__setitem__
                for card in cards:
                    if type(card) is tuple and len(card) == 3:
                        key, value, comment = card
                        card_obj = Card(
                            str(key),
                            value,
                            "" if comment is None else str(comment),
                        )
                        append(card_obj)
                        if key not in {"HISTORY", "COMMENT"}:
                            setitem(key, value)
                    else:
                        try:
                            parsed = self._coerce_card(card)
                        except (TypeError, ValueError):
                            continue
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
        key_s = str(key)
        super().__delitem__(key)
        for idx, card in enumerate(self._cards):
            if card.key == key_s:
                del self._cards[idx]
                break
        self._version += 1

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
        res = super().pop(*args)
        for idx, card in enumerate(self._cards):
            if card.key == key:
                del self._cards[idx]
                break
        self._version += 1
        return res

    def popitem(self) -> tuple[str, Any]:
        key, value = super().popitem()
        key_s = str(key)
        # Prefer the card whose value matches the mapping entry we just popped
        # (insert() can leave duplicate keys; first-key linear search is wrong).
        match_idx: int | None = None
        for idx, card in enumerate(self._cards):
            if card.key == key_s and card.value == value:
                match_idx = idx
                break
        if match_idx is None:
            for idx, card in enumerate(self._cards):
                if card.key == key_s:
                    match_idx = idx
                    break
        if match_idx is not None:
            del self._cards[match_idx]
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
        self._set_mapping_for_card(parsed)
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
        if remove_all and len(matches) > 1:
            self._cards = [c for c in self._cards if c.key != key_s]
        else:
            remove_indices = sorted(
                matches if remove_all else [matches[0]], reverse=True
            )
            for idx in remove_indices:
                del self._cards[idx]
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
        return Card(str(key), value, "" if comment is None else str(comment))

    def _append_card(
        self, card: Card, *, update_mapping: bool, bump: bool = True
    ) -> None:
        self._cards.append(card)
        if update_mapping:
            self._set_mapping_for_card(card)
        if bump:
            self._version += 1

    def _set_card(self, key: str, value: Any, comment: str, *, bump: bool) -> None:
        card = Card(key, value, comment)
        if key in {"HISTORY", "COMMENT"}:
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
        super().__setitem__(card.key, card.value)

    def _rebuild_mapping_for_key(self, key: str) -> None:
        remaining = [card for card in self._cards if card.key == key]
        if remaining:
            super().__setitem__(key, remaining[-1].value)
        elif key in self:
            super().__delitem__(key)

    def _repr_html_(self) -> str:
        import html as pyhtml

        html_parts = [
            '<div tabindex="0" aria-label="FITS Header" style=\'max-height: 400px; overflow: auto; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;\'>',
            "<table style='border-collapse: collapse; width: 100%; margin: 0;'>",
            "<thead><tr>",
        ]
        headers = ["Keyword", "Value", "Comment"]
        for h in headers:
            html_parts.append(
                f'<th scope="col" style=\'text-align: left; padding: 8px; position: sticky; top: 0; '
                f"background-color: var(--theme-ui-colors-background, white); "
                f"border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;'>{h}</th>"
            )
        html_parts.append("</tr></thead><tbody>")

        for card in self._cards:
            k = pyhtml.escape(str(card.key))
            v = pyhtml.escape(str(card.value)) if card.value is not None else ""
            c = pyhtml.escape(str(card.comment))
            html_parts.append("<tr>")
            html_parts.append(
                f"<th scope=\"row\" style='text-align: left; padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); font-weight: bold;'>{k}</th>"
            )
            html_parts.append(
                f"<td style='padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>{v}</td>"
            )
            html_parts.append(
                f"<td style='padding: 8px; border-bottom: 1px solid rgba(128, 128, 128, 0.2); opacity: 0.7;'>{c}</td>"
            )
            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div>")
        return "".join(html_parts)
