#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERN_DIR="${ROOT_DIR}/extern"
TMP_DIR="${ROOT_DIR}/.tmp-vendor"

CFITSIO_REPO="HEASARC/cfitsio"
CFITSIO_VERSION="latest"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--cfitsio-version <tag>]

Defaults to latest published release for each dependency.
Examples:
  $(basename "$0")
  $(basename "$0") --cfitsio-version cfitsio-4.6.2
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfitsio-version)
      CFITSIO_VERSION="$2"
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

require_cmd curl
require_cmd tar

latest_tag() {
  local repo="$1"
  local tag

  tag="$(curl -fsSL "https://api.github.com/repos/${repo}/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n1)"

  if [[ -z "${tag}" ]]; then
    tag="$(curl -fsSL "https://api.github.com/repos/${repo}/tags" \
      | sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      | head -n1)"
  fi

  if [[ -z "${tag}" ]]; then
    echo "Failed to resolve latest tag for ${repo}" >&2
    exit 1
  fi

  echo "${tag}"
}


fetch_and_extract() {
  local repo="$1"
  local tag="$2"
  local dest="$3"
  local archive="${TMP_DIR}/$(basename "${dest}")-${tag}.tar.gz"

  rm -rf "${dest}"
  mkdir -p "${TMP_DIR}"

  echo "Downloading ${repo}@${tag}"
  curl -fL "https://github.com/${repo}/archive/refs/tags/${tag}.tar.gz" -o "${archive}"

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


if [[ "${CFITSIO_VERSION}" == "latest" ]]; then
  CFITSIO_VERSION="$(latest_tag "${CFITSIO_REPO}")"
fi

mkdir -p "${EXTERN_DIR}"
fetch_and_extract "${CFITSIO_REPO}" "${CFITSIO_VERSION}" "${EXTERN_DIR}/cfitsio"

cat > "${EXTERN_DIR}/VERSIONS.txt" <<VERSIONS
cfitsio_repo=${CFITSIO_REPO}
cfitsio_tag=${CFITSIO_VERSION}
VERSIONS

echo "Vendored deps prepared in ${EXTERN_DIR}"
echo "Recorded versions in ${EXTERN_DIR}/VERSIONS.txt"
