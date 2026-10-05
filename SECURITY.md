# Security and Privacy

LectureLens handles classroom audio and transcripts, which can contain personal or sensitive information.

## Current privacy model

- microphone access is requested only when the user chooses Record
- imported and recorded audio is attached to a local browser object URL
- semantic embeddings are computed in the browser
- the most recent transcript may be stored in browser local storage
- no account is required

## Reporting a vulnerability

Do not include real classroom recordings, student information, credentials, or private transcripts in a public GitHub issue. Use synthetic data when reproducing issues.

Any future feature that uploads, synchronizes, or remotely processes classroom audio or transcripts must update this document and the product disclosure before release.
