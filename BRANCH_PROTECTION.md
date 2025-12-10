# Branch Protection Rules for Main Branch

This document outlines the branch protection rules that should be configured for the `main` branch to ensure code quality and security.

## Overview

Branch protection rules prevent force pushes, enforce code review requirements, and ensure that all changes go through proper validation before being merged into the main branch.

## Recommended Settings

### 1. Require Pull Request Reviews Before Merging

**Setting**: ✅ Enabled

- **Required approving reviews**: 1
- **Dismiss stale pull request approvals when new commits are pushed**: ✅ Enabled
- **Require review from Code Owners**: ✅ Enabled (see `.github/CODEOWNERS`)
- **Restrict who can dismiss pull request reviews**: Repository admins only

**Why**: Ensures that all code changes are reviewed by at least one other person, reducing bugs and improving code quality.

### 2. Require Status Checks to Pass Before Merging

**Setting**: ✅ Enabled

- **Require branches to be up to date before merging**: ✅ Enabled
- **Required status checks**:
  - `lint-and-test` (from CI/CD workflow)
  - `security-check` (from CI/CD workflow)

**Why**: Automated tests and security checks must pass before code can be merged, preventing broken or vulnerable code from entering the main branch.

### 3. Require Conversation Resolution Before Merging

**Setting**: ✅ Enabled

**Why**: Ensures all PR comments and discussions are resolved before merging, preventing overlooked issues.

### 4. Require Signed Commits

**Setting**: ⚠️ Optional (Recommended for high-security projects)

**Why**: Verifies the identity of commit authors, preventing unauthorized code changes.

### 5. Require Linear History

**Setting**: ⚠️ Optional

**Why**: Enforces a linear git history by preventing merge commits, making the history cleaner and easier to understand.

### 6. Include Administrators

**Setting**: ✅ Enabled

**Why**: Even administrators must follow the same rules, ensuring consistency across all contributions.

### 7. Restrict Who Can Push to Matching Branches

**Setting**: ✅ Enabled

- **Restrict pushes that create matching branches**: Only allow repository collaborators
- **Allow force pushes**: ❌ Disabled
- **Allow deletions**: ❌ Disabled

**Why**: Prevents accidental or malicious deletion of the main branch and ensures all changes go through pull requests.

## How to Configure These Settings

### Via GitHub Web Interface

1. Navigate to your repository on GitHub
2. Go to **Settings** → **Branches**
3. Click **Add rule** or edit existing rule for `main` branch
4. Configure the settings as described above
5. Click **Create** or **Save changes**

### Visual Configuration Checklist

```
Branch Protection Rule for: main

✅ Require a pull request before merging
   ✅ Require approvals: 1
   ✅ Dismiss stale pull request approvals when new commits are pushed
   ✅ Require review from Code Owners
   ✅ Require approval of the most recent reviewable push

✅ Require status checks to pass before merging
   ✅ Require branches to be up to date before merging
   Status checks:
   - lint-and-test
   - security-check

✅ Require conversation resolution before merging

❌ Require signed commits (optional)

❌ Require linear history (optional)

✅ Require deployments to succeed before merging (if applicable)

✅ Lock branch (for archived projects only)

✅ Do not allow bypassing the above settings
   (Apply rules to administrators)

✅ Restrict who can push to matching branches
   - Restrict pushes
   ❌ Allow force pushes
   ❌ Allow deletions
```

## Additional Protections

### CODEOWNERS File

Located at `.github/CODEOWNERS`, this file defines who must review changes to specific files or directories.

**Current configuration**:
- All files require review from @ujjawalkaushik1110
- Source code, data, documentation, and GitHub config all protected

### Automated CI/CD

Located at `.github/workflows/ci.yml`, this workflow:
- Runs on every push and pull request to `main` and `develop` branches
- Checks code formatting with Black
- Lints code with flake8
- Runs security scans with Safety
- Runs unit tests (if available)

### Dependabot

Located at `.github/dependabot.yml`, this configuration:
- Automatically checks for dependency updates weekly
- Creates pull requests for Python dependencies
- Updates GitHub Actions versions
- Assigns PRs to repository owner for review

## Benefits of These Rules

1. **Code Quality**: All code is reviewed and tested before merging
2. **Security**: Automated security scans catch vulnerabilities early
3. **Collaboration**: Encourages discussion and knowledge sharing
4. **Accountability**: Clear ownership and review process
5. **Stability**: Prevents breaking changes in production code
6. **Traceability**: All changes are documented and reviewable

## Enforcement

These rules are enforced automatically by GitHub. Any attempt to:
- Push directly to `main` will be rejected
- Merge without required reviews will be blocked
- Merge with failing tests will be prevented

## Exceptions

In emergency situations (e.g., critical security fixes), repository administrators can temporarily disable branch protection. However, this should be:
1. Documented in the commit message
2. Re-enabled immediately after the fix
3. Followed up with a post-mortem review

## Updating These Rules

To modify branch protection rules:
1. Create a pull request updating this document
2. Get approval from repository owner
3. Update GitHub settings to match
4. Document changes in commit message

## Questions?

For questions about branch protection rules, please:
- Open an issue using the question label
- Contact @ujjawalkaushik1110
- Refer to [GitHub's Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

---

**Last Updated**: December 2024  
**Maintained By**: @ujjawalkaushik1110
