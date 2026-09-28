"""TS-012: the documented HTTP environment-variable contracts were untested.

``docs/architecture.md`` documents three environment variables and, for two of
them, a *precedence rule*:

===========================================  ========  ==============================
``TORCHFITS_HTTP_TIMEOUT``                   ``120``  download / Range timeout (s)
``TORCHFITS_HTTP_AUTHORIZATION``             unset    full header value; **wins**
``TORCHFITS_HTTP_TOKEN``                     unset    sent as ``Bearer <token>``
===========================================  ========  ==============================

``http_util.http_timeout`` and ``http_util.auth_headers`` implement them and had
**no test at all** — the only reference to any of the three anywhere under
``tests/`` was a single ``monkeypatch.setenv("TORCHFITS_HTTP_TOKEN", ...)`` in
``test_remote_http_range.py``, which never asserts the header that comes out.
So a refactor that dropped the ``Bearer`` prefix, inverted the precedence, or
broke the unparseable-value fallback would all have been silent.

Every value asserted below was measured against the real functions before being
written down; the matrix is in the audit ledger.
"""

from __future__ import annotations

import pytest

from torchfits.http_util import auth_headers, http_timeout

_TIMEOUT_ENV = "TORCHFITS_HTTP_TIMEOUT"
_AUTH_ENV = "TORCHFITS_HTTP_AUTHORIZATION"
_TOKEN_ENV = "TORCHFITS_HTTP_TOKEN"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (_TIMEOUT_ENV, _AUTH_ENV, _TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)


def test_http_timeout_default_is_the_documented_120():
    """`docs/architecture.md` states the default; the function must agree."""
    assert http_timeout() == 120.0


def test_http_timeout_honours_a_non_default_default_argument():
    """The `default` parameter is the fallback, not a hardcoded 120."""
    assert http_timeout(7.5) == 7.5


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("120", 120.0),
        ("30.5", 30.5),
        ("1e3", 1000.0),
        ("  45  ", 45.0),  # surrounding whitespace is stripped
        ("0.001", 0.001),  # small but positive is legitimate
    ],
)
def test_http_timeout_parses_usable_numeric_values(monkeypatch, raw, expected):
    """Anything a positive finite number parses to is passed through."""
    monkeypatch.setenv(_TIMEOUT_ENV, raw)
    assert http_timeout() == expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_http_timeout_blank_value_uses_the_supplied_default(monkeypatch, raw):
    """Blank means "unset", so *this* call's default applies, not 120."""
    monkeypatch.setenv(_TIMEOUT_ENV, raw)
    assert http_timeout(3.0) == 3.0
    assert http_timeout() == 120.0


# TS-019: the unparseable and non-positive cases used to be silent, and four of
# them reached the transport as raw low-level errors. Both the fallback and the
# warning are now pinned.


@pytest.mark.parametrize("raw", ["abc", "30s", "12,5", "None", "1 2"])
def test_unparseable_timeout_warns_and_falls_back(monkeypatch, raw):
    """A typo must not look like it took effect."""
    monkeypatch.setenv(_TIMEOUT_ENV, raw)
    with pytest.warns(UserWarning, match="is not a number") as record:
        assert http_timeout() == 120.0
    message = str(record[0].message)
    assert _TIMEOUT_ENV in message and repr(raw) in message
    assert "120" in message, "the warning must state the fallback in force"


@pytest.mark.parametrize("raw", ["0", "-5", "-0.5", "nan", "inf", "-inf"])
def test_non_positive_or_non_finite_timeout_warns_and_falls_back(monkeypatch, raw):
    """These reached the socket layer as errno 36 / ValueError / OverflowError.

    Measured against a real socket: ``0`` made the socket non-blocking
    (``BlockingIOError [Errno 36] Operation now in progress``, the connection
    never established), ``-5`` raised ``ValueError: Timeout value out of
    range``, ``nan`` raised ``ValueError: Invalid value NaN`` and ``inf``
    raised ``OverflowError: timestamp out of range for C PyTime_t`` — none of
    which names the variable that caused it. They now fall back, loudly.
    """
    monkeypatch.setenv(_TIMEOUT_ENV, raw)
    with pytest.warns(UserWarning, match="not a positive, finite number"):
        assert http_timeout() == 120.0


def test_unusable_timeout_warning_names_the_supplied_default(monkeypatch):
    """The warning reports the fallback actually in force, not a hardcoded 120."""
    monkeypatch.setenv(_TIMEOUT_ENV, "nope")
    with pytest.warns(UserWarning) as record:
        assert http_timeout(7.0) == 7.0
    assert "7" in str(record[0].message)


def test_usable_timeout_never_warns(monkeypatch, recwarn):
    """The common path must stay silent — no warning on a good value."""
    monkeypatch.setenv(_TIMEOUT_ENV, "45")
    assert http_timeout() == 45.0
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_unset_timeout_never_warns(recwarn):
    """Absent configuration is not a misconfiguration."""
    assert http_timeout() == 120.0
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_auth_headers_absent_by_default():
    """No env means no Authorization header at all (not an empty one)."""
    assert auth_headers() == {}


def test_auth_token_is_sent_as_a_bearer_credential(monkeypatch):
    monkeypatch.setenv(_TOKEN_ENV, "test-token")
    assert auth_headers() == {"Authorization": "Bearer test-token"}


def test_authorization_wins_over_token(monkeypatch):
    """The documented precedence: the full header value wins."""
    monkeypatch.setenv(_TOKEN_ENV, "test-token")
    monkeypatch.setenv(_AUTH_ENV, "Bearer other-scheme")
    assert auth_headers() == {"Authorization": "Bearer other-scheme"}


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_authorization_falls_through_to_the_token(monkeypatch, blank):
    """Whitespace-only counts as unset, so the token is not shadowed."""
    monkeypatch.setenv(_AUTH_ENV, blank)
    monkeypatch.setenv(_TOKEN_ENV, "test-token")
    assert auth_headers() == {"Authorization": "Bearer test-token"}


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_token_yields_no_header(monkeypatch, blank):
    """A blank token must not produce ``Authorization: "Bearer "``."""
    monkeypatch.setenv(_TOKEN_ENV, blank)
    assert auth_headers() == {}


def test_authorization_is_sent_verbatim_without_a_bearer_prefix(monkeypatch):
    """`_AUTHORIZATION` is the *whole* value, so no prefix may be added."""
    monkeypatch.setenv(_AUTH_ENV, "Basic dXNlcjpwYXNz")
    assert auth_headers() == {"Authorization": "Basic dXNlcjpwYXNz"}


def test_env_reads_are_not_cached_across_calls(monkeypatch):
    """A second read after the env changes must see the new value.

    Cheap to assert, and it is the failure mode a module-level cache would
    introduce: the test suite itself sets these variables per test.
    """
    assert auth_headers() == {}
    monkeypatch.setenv(_TOKEN_ENV, "later")
    assert auth_headers() == {"Authorization": "Bearer later"}
    monkeypatch.setenv(_TIMEOUT_ENV, "9")
    assert http_timeout() == 9.0
