# Code signing with SignPath Foundation (maintainer notes)

BasiCAM CNC is prepared for free code signing from the SignPath Foundation, which signs open-source
projects under an OSI-approved license. Until the application is approved, releases are **unsigned**
and Windows SmartScreen may warn when the program is first started.

## What is already in place

- License: GPL-3.0-or-later (`LICENSE`), with no extra restrictions (`TERMS_OF_USE.md` only explains it).
- A code signing policy and privacy statement in `README.md` (SignPath requires both to be public).
- Reproducible CI build: `.github/workflows/build.yml` builds the one-folder Nuitka program with
  `build_nuitka.py` on a GitHub-hosted runner and uploads it as an artifact. SignPath only signs
  artifacts built by CI from this public repository.
- Version and product information in the exe's file properties (from `basicam/about.py`).
- `.signpath/artifact-configuration.xml`: what to sign (only `BasiCAM CNC.exe`).
- A signing job and a release job in the workflow, switched off until `SIGNPATH_ENABLED` is `true`.

## Steps after approval

1. Apply at <https://signpath.org/apply> with the repository URL.
2. In SignPath, create the project (for example slug `BasiCAM_CNC`), connect the GitHub repository
   as a trusted build system, add the artifact configuration from `.signpath/artifact-configuration.xml`,
   and use the `release-signing` policy (or the one SignPath assigns).
3. In GitHub: **Settings > Secrets and variables > Actions**:
   - Secret `SIGNPATH_API_TOKEN`: a SignPath CI user's API token.
   - Variables `SIGNPATH_ORGANIZATION_ID`, `SIGNPATH_PROJECT_SLUG`, `SIGNPATH_POLICY_SLUG`,
     `SIGNPATH_ARTIFACT_CONFIGURATION_SLUG` (from the SignPath project).
   - Variable `SIGNPATH_ENABLED` = `true`.
4. Release: bump `APP_VERSION` in `basicam/about.py`, commit, then push a tag `vX.Y.Z`. The workflow
   builds, submits the folder for signing, waits for approval in SignPath, and attaches
   `BasiCAM-CNC-X.Y.Z-windows-x64.zip` (signed) to the release.
5. Once the first signed release is out, change the README code signing section from "will be" to "is".

## Rules SignPath Foundation expects us to keep

- Sign only what this project builds; never third-party binaries.
- Every signing request is approved by a project approver (the repository owner).
- No component that downloads or runs other code without the user asking, and no data sent anywhere
  without the user asking (see the privacy statement in `README.md`).
