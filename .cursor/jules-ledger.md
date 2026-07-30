# Jules theme ledger

Themes already covered by Jules (or integrated Jules PRs). Weekly Jules must
**not** reopen these without a new repro and human approval.

| Theme | Evidence | Status |
|---|---|---|
| probe SSRF private/loopback guard | #216 | landed |
| hoist inner classes (cosmetic) | #214 | landed — do not repeat |
| HDU `repr_html` / HTML a11y | #213 / #219 | landed — do not repeat a11y drive-bys |
| shared cached `fitsfile*` cross-thread | CFITSIO R2 / rc2 | landed (open-per-read) |
| HTTP Range remote cutouts + vos/vault bridge | main `83c64d4` | landed |
| SubsetReader mmap cutouts (uncompressed 2D) | main `d4b419c` | landed |
| Lupton RGB Astropy-parity peak clip | main `91b30b0` | landed |
| Deep review 1.0 triage (alias/spectral hard-remove, bench honesty) | main (pre-1.0 triage) | landed — do not reopen without new repro |
| optimize Header.remove for huge HISTORY lists | #225 | landed |

When integrating a Jules PR, append a row here before closing the PR.
