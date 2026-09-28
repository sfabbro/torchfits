#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERN_DIR="${ROOT_DIR}/extern"
TMP_DIR="${ROOT_DIR}/.tmp-vendor"

CFITSIO_REPO="HEASARC/cfitsio"
CFITSIO_VERSION=""
CFITSIO_SPEC_FILE=""
CFITSIO_SPEC_IS_FILE="0"
CFITSIO_SHA256=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") --cfitsio-version <versions-file>
       $(basename "$0") --cfitsio-version <tag>   # preview only, see below

Vendored dependencies are pinned: pass extern/VERSIONS.txt. A sha256 recorded
in the versions file is enforced against the downloaded tarball, and that file
is the ONLY thing this script will write. "latest" resolution was removed so
builds can never silently pick up different upstream code (H4).

Passing a bare tag instead of the versions file vendors that tag and prints
the sha256 it computed, but does NOT update extern/VERSIONS.txt. Patches are
named "<tag>-<name>.patch" and are skipped for any other tag, so a bare-tag
run is a preview of another version, not a way to move the pin: to move it,
edit extern/VERSIONS.txt deliberately and pass that file.

Examples:
  $(basename "$0") --cfitsio-version extern/VERSIONS.txt   # what CI runs
  $(basename "$0") --cfitsio-version cfitsio-4.6.2        # preview; prints hash
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfitsio-version)
      if [[ $# -lt 2 ]]; then
        echo "--cfitsio-version requires a value." >&2
        usage
        exit 1
      fi
      CFITSIO_VERSION="$2"
      CFITSIO_SPEC_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CFITSIO_VERSION}" ]]; then
  echo "--cfitsio-version is required (pass extern/VERSIONS.txt)." >&2
  usage
  exit 1
fi
if [[ -f "${CFITSIO_SPEC_FILE}" ]]; then
  CFITSIO_SPEC_IS_FILE="1"
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

resolve_cfitsio_version() {
  local spec="$1"

  if [[ -f "${spec}" ]]; then
    spec="$(cat "${spec}")"
  fi

  if [[ "${spec}" == *$'\n'* ]] || [[ "${spec}" == cfitsio_*=* ]]; then
    local tag_line
    tag_line="$(printf '%s\n' "${spec}" | grep -E '^cfitsio_tag=' | head -n1 || true)"
    if [[ -n "${tag_line}" ]]; then
      spec="${tag_line#cfitsio_tag=}"
    else
      spec="$(printf '%s\n' "${spec}" | grep -Ev '^cfitsio_repo=' | head -n1 | tr -d '[:space:]')"
    fi
  fi

  if [[ -z "${spec}" ]]; then
    echo "Failed to resolve CFITSIO version from: $1" >&2
    exit 1
  fi

  echo "${spec}"
}

require_cmd curl
require_cmd tar
# GNU coreutils sha256sum on Linux. macOS runners expose a limited
# sha256sum shim that rejects --check/--status ("usage: sha256sum
# [-bctwz]"), which broke the v1.1.0 tag build twice — select by OS,
# not by binary presence: Darwin always ships perl shasum.
IS_DARWIN="0"
[[ "$(uname -s)" == "Darwin" ]] && IS_DARWIN="1"
if [[ "${IS_DARWIN}" == "1" ]]; then
  require_cmd shasum
else
  require_cmd sha256sum
fi

# Portable sha256 helpers: digest a file / verify a "HASH  file" checklist.
sha256_file() {
  if [[ "${IS_DARWIN}" == "1" ]]; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    sha256sum "$1" | cut -d' ' -f1
  fi
}

sha256_check() {
  # $1 = expected hash, $2 = archive path; fails on mismatch.
  if [[ "${IS_DARWIN}" == "1" ]]; then
    echo "${1}  ${2}" | shasum -a 256 -c -s -
  else
    echo "${1}  ${2}" | sha256sum --check --status
  fi
}

CFITSIO_VERSION="$(resolve_cfitsio_version "${CFITSIO_VERSION}")"

fetch_and_extract() {
  local repo="$1"
  local tag="$2"
  local dest="$3"
  local archive="$4"

  rm -rf "${dest}"
  mkdir -p "${TMP_DIR}"

  echo "Downloading ${repo}@${tag}"
  curl -fL --retry 3 --retry-delay 2 \
    "https://github.com/${repo}/archive/refs/tags/${tag}.tar.gz" -o "${archive}"

  if [[ -n "${CFITSIO_SHA256}" ]]; then
    echo "Verifying sha256 (${CFITSIO_SHA256})"
    sha256_check "${CFITSIO_SHA256}" "${archive}" ||
      { echo "sha256 MISMATCH for ${repo}@${tag}: refusing to vendor" >&2; exit 1; }
  elif [[ "${TORCHFITS_VENDOR_ALLOW_UNPINNED:-0}" != "1" ]]; then
    echo "No cfitsio_sha256 recorded for ${tag}." >&2
    echo "Re-run with TORCHFITS_VENDOR_ALLOW_UNPINNED=1 to accept and record it," >&2
    echo "or pin a hash in extern/VERSIONS.txt (cfitsio_sha256=...)." >&2
    exit 1
  fi

  local extract_dir="${TMP_DIR}/extract-$(basename "${dest}")-${tag}"
  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"

  tar -xzf "${archive}" -C "${extract_dir}"
  local src_dir
  src_dir="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"

  if [[ -z "${src_dir}" ]]; then
    echo "Failed to extract ${repo}@${tag}" >&2
    exit 1
  fi

  mv "${src_dir}" "${dest}"
}


# Resolve the pinned hash from the versions file (if the user passed one).
if [[ "${CFITSIO_SPEC_IS_FILE}" == "1" ]]; then
  CFITSIO_SHA256="$(grep -E '^cfitsio_sha256=' "${CFITSIO_SPEC_FILE}" | head -n1 | cut -d= -f2- || true)"
fi

compute_archive_hash() {
  sha256_file "${CFITSIO_ARCHIVE}"
}

mkdir -p "${EXTERN_DIR}"
# Single source of truth for the archive path, shared by fetch + hash record.
CFITSIO_ARCHIVE="${TMP_DIR}/$(basename "${EXTERN_DIR}/cfitsio")-${CFITSIO_VERSION}.tar.gz"
fetch_and_extract "${CFITSIO_REPO}" "${CFITSIO_VERSION}" "${EXTERN_DIR}/cfitsio" "${CFITSIO_ARCHIVE}"

# Apply any patches for this exact vendored version.  Patch file names are
# "<tag>-<name>.patch"; a patch whose <tag> does not match the vendored
# version is skipped so stale patches never get applied.
PATCH_DIR="${EXTERN_DIR}/patches"
if [[ -d "${PATCH_DIR}" ]]; then
  require_cmd patch
  for p in "${PATCH_DIR}"/"${CFITSIO_VERSION}"-*.patch; do
    [[ -f "${p}" ]] || continue
    echo "Applying patch ${p} to ${EXTERN_DIR}/cfitsio"
    ( cd "${EXTERN_DIR}/cfitsio" && patch -p1 < "${p}" )
  done
  # A patch is named "<tag>-<name>.patch" and is applied only to that exact
  # tag, so vendoring any other tag silently yields an UNPATCHED tree. Some of
  # those patches are security fixes (cfitsio-4.7.0-plio-cbuf is a CFITSIO heap
  # overflow on PLIO-compressed incompressible data), so say what was left out
  # instead of letting it disappear into a passing build.
  skipped=0
  for p in "${PATCH_DIR}"/*.patch; do
    [[ -f "${p}" ]] || continue
    case "$(basename "${p}")" in
      "${CFITSIO_VERSION}"-*) continue ;;
    esac
    if [[ "${skipped}" -eq 0 ]]; then
      echo "WARNING: ${CFITSIO_VERSION} has no matching patch; these are being skipped:" >&2
    fi
    skipped=$((skipped + 1))
    echo "  $(basename "${p}")" >&2
  done
  if [[ "${skipped}" -gt 0 ]]; then
    echo "  -> ${EXTERN_DIR}/cfitsio is NOT patched. If this tag is meant to" >&2
    echo "     replace ${CFITSIO_VERSION}, add <tag>-<name>.patch copies first." >&2
  fi
fi

RECORDED_HASH="$(compute_archive_hash)"
if [[ "${CFITSIO_SPEC_IS_FILE}" == "1" ]]; then
  # The tag came out of the file, so this can only ever rewrite the same tag;
  # a changed upstream hash would have failed the check above. It is kept for
  # the "pin a new tag by hand" flow, where the file is edited first and the
  # recorded hash refreshed by running this.
  cat > "${CFITSIO_SPEC_FILE}" <<VERSIONS
cfitsio_repo=${CFITSIO_REPO}
cfitsio_tag=${CFITSIO_VERSION}
cfitsio_sha256=${RECORDED_HASH}
VERSIONS
  echo "Recorded versions in ${CFITSIO_SPEC_FILE}"
else
  # Bare tag: report, do not record. Writing here is how a stray
  # `--cfitsio-version cfitsio-X.Y.Z` used to repoint the tracked pin and
  # silently drop every patch written for the pinned tag.
  echo "cfitsio_tag=${CFITSIO_VERSION}"
  echo "cfitsio_sha256=${RECORDED_HASH}"
  echo "Not recorded: ${EXTERN_DIR}/VERSIONS.txt is only written when it is the"
  echo "--cfitsio-version argument. To move the pin, edit it and re-run with it."
fi

echo "Vendored deps prepared in ${EXTERN_DIR}"
