#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERN_DIR="${ROOT_DIR}/extern"
TMP_DIR="${ROOT_DIR}/.tmp-vendor"

CFITSIO_REPO="HEASARC/cfitsio"
CFITSIO_VERSION=""
CFITSIO_SPEC_FILE=""
CFITSIO_SHA256=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") --cfitsio-version <tag-or-versions-file>

Vendored dependencies are pinned: pass an exact tag or a versions file
(extern/VERSIONS.txt). A sha256 recorded in the versions file is enforced
against the downloaded tarball; fetching a tag with no recorded hash
requires TORCHFITS_VENDOR_ALLOW_UNPINNED=1 (the hash is then computed and
recorded for the next run). "latest" resolution was removed so builds can
never silently pick up different upstream code (H4).

Examples:
  $(basename "$0") --cfitsio-version extern/VERSIONS.txt
  $(basename "$0") --cfitsio-version cfitsio-4.6.2   # requires ALLOW_UNPINNED
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfitsio-version)
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
# GNU coreutils sha256sum on Linux; macOS runners ship only perl shasum
# (its coreutils namesake rejects --check/--status, which broke the first
# tagged build that exercised checksum verification).
if ! command -v sha256sum >/dev/null 2>&1; then
  require_cmd shasum
fi

# Portable sha256 helpers: digest a file / verify a "HASH  file" checklist.
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

sha256_check() {
  # $1 = expected hash, $2 = archive path; fails on mismatch.
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${1}  ${2}" | sha256sum --check --status
  else
    echo "${1}  ${2}" | shasum -a 256 -c -s -
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
if [[ -n "${CFITSIO_SPEC_FILE}" && -f "${CFITSIO_SPEC_FILE}" ]]; then
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
fi

RECORDED_HASH="$(compute_archive_hash)"
cat > "${EXTERN_DIR}/VERSIONS.txt" <<VERSIONS
cfitsio_repo=${CFITSIO_REPO}
cfitsio_tag=${CFITSIO_VERSION}
cfitsio_sha256=${RECORDED_HASH}
VERSIONS

echo "Vendored deps prepared in ${EXTERN_DIR}"
echo "Recorded versions in ${EXTERN_DIR}/VERSIONS.txt"
