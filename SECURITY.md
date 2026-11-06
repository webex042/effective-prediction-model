# Security Policy

## Scope

EPM is a research project. It trains and evaluates models on local, registered
datasets, and it ships two network-facing components:

- the local experiment API and dashboard (`python run.py --serve`), intended for
  `localhost` use only;
- the standalone prediction chatbot (`epm_chatbot/`), which can be embedded on a
  site behind an origin allowlist.

Those two are where a security issue is most likely to matter. The research code
itself (`src/`, `experiments/`, the stage demos) is not intended for production
deployment, and the README says so.

## Reporting a vulnerability

Please **do not** open a public issue.

Email **ranpranshu65@gmail.com** with:

- what the issue is and which component it affects,
- the steps to reproduce it,
- what an attacker could do with it.

You should get an acknowledgement within 7 days. The project is maintained by
one person alongside full-time study, so please allow reasonable time for a fix.
Credit will be given in the changelog unless you prefer otherwise.

## Known limitations

- The dashboard and local experiment API have no authentication and are meant
  to be bound to localhost. Do not expose them to a public network.
- `epm_chatbot` restricts embedding by origin allowlist; it is not a substitute
  for authentication in front of sensitive data.
- Dataset downloads fetch from third-party sources (scikit-learn, UCI, OpenML,
  NAB). Each registry entry records the source; verify with
  `python run.py --verify-datasets`.
