# Contributing

This repository follows a feature-branch-first workflow. The goal is fast, small, test-covered changes with clear PRs and reviews.

## Branch naming
- `feature/<short-description>` — new features or increments (e.g. `feature/messaging/realtime-init`)
- `fix/<short-description>` — bug fixes
- `chore/<short-description>` — tooling, formatting, non-functional changes
- `hotfix/<short-description>` — urgent fixes to `main`

## Pull requests
- Open PRs from feature branches into `dev` (integration) or `main` (release hotfixes).
- Require at least one reviewer for feature PRs and two for production-impacting PRs.
- All tests must pass on CI before merging. If tests are flaky, stabilize in a follow-up PR.

## Commits
- Use concise, prefixed commit messages: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`
- Keep commits small and logically grouped. Prefer multiple small commits over one large commit.

## Code style & checks
- Run linters and formatters locally before committing.
- We recommend installing Husky pre-commit hooks (the repo may include a hook to run `npm test` or `npm run lint`).

## Releases
- `main` is the production branch. `dev` is the integration branch for ongoing work.

If you're uncertain about a change, open a draft PR and add notes on risk and testing.
