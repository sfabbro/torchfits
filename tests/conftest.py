from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imported AFTER the sys.path injection so this works when torchfits is editable-installed
# (src/ on sys.path) as well as when it is also vap-installed (site-packages already has it).
from torchfits.cache import CACHE_ENV_SENTINELS  # noqa: E402


# Several CACHE_ENV_SENTINELS (KUBERNETES_SERVICE_HOST, K_SERVICE, WEBSITE_SITE_NAME, ...)
# leak into dev sandboxes and shared CI pods, where CacheConfig auto-detection would
# otherwise mis-classify the env as "cloud" / "hpc". The autouse fixture below clears the
# FULL allowlist (single source of truth: CACHE_ENV_SENTINELS imported above from
# torchfits.cache) before every test so each test starts from a clean env and can opt
# into its expected profile by setenv'ing the relevant sentinel. Tests that explicitly
# opt in (e.g. monkeypatch.setenv SLURM_JOB_ID for hpc tests, AWS_EXECUTION_ENV for cloud
# tests) still work because monkeypatch.setenv runs AFTER the fixture's delenv pass.
@pytest.fixture(autouse=True)
def _clear_env_profile(monkeypatch):
    for var in CACHE_ENV_SENTINELS:
        monkeypatch.delenv(var, raising=False)
    yield
