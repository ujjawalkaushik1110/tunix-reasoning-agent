# Branch Protection Setup Guide

This guide will help you configure branch protection rules for the `main` branch.

## Quick Setup (Recommended)

### Method 1: Using GitHub CLI (Fastest)

```bash
# 1. Install GitHub CLI if not already installed
# macOS:
brew install gh

# Windows:
winget install GitHub.cli

# Linux:
# See: https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# 2. Authenticate with GitHub
gh auth login

# 3. Run the setup script
cd .github
./setup-branch-protection.sh
```

### Method 2: Using GitHub Web Interface

1. **Navigate to Repository Settings**
   - Go to: `https://github.com/ujjawalkaushik1110/tunix-reasoning-agent/settings/branches`
   - Or: Repository → Settings → Branches

2. **Add Branch Protection Rule**
   - Click "Add rule" or "Add branch protection rule"
   - Branch name pattern: `main`

3. **Configure Protection Settings**

   #### Protect matching branches:
   
   ✅ **Require a pull request before merging**
   - ✅ Require approvals: `1`
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require review from Code Owners
   - ✅ Require approval of the most recent reviewable push
   
   ✅ **Require status checks to pass before merging**
   - ✅ Require branches to be up to date before merging
   - Search and add these status checks:
     - `Lint and Test`
     - `Security Scan`
   
   ✅ **Require conversation resolution before merging**
   
   ✅ **Do not allow bypassing the above settings**
   
   ✅ **Rules applied to everyone, including administrators**

4. **Additional Settings**
   
   Under "Rules applied to everyone including administrators":
   - ❌ Allow force pushes (keep unchecked)
   - ❌ Allow deletions (keep unchecked)

5. **Create/Save**
   - Click "Create" or "Save changes"

## Verification

After setup, verify the protection rules:

```bash
# Using GitHub CLI
gh api repos/ujjawalkaushik1110/tunix-reasoning-agent/branches/main/protection

# Or visit:
# https://github.com/ujjawalkaushik1110/tunix-reasoning-agent/settings/branch_protection_rules
```

## Testing Protection Rules

Try these tests to confirm protection is working:

### Test 1: Direct Push (Should Fail)
```bash
# This should be rejected
git checkout main
echo "test" >> test.txt
git add test.txt
git commit -m "test"
git push origin main
# Expected: Error about protected branch
```

### Test 2: Pull Request (Should Work)
```bash
# This should work
git checkout -b test-branch
echo "test" >> test.txt
git add test.txt
git commit -m "test"
git push origin test-branch
# Then create PR via GitHub interface
```

## What's Protected

With these rules enabled:

✅ **What You CAN Do:**
- Create feature branches
- Push to feature branches
- Create pull requests
- Review and comment on PRs
- Merge PRs after approval and passing checks

❌ **What You CANNOT Do:**
- Push directly to `main` branch
- Force push to `main` branch
- Delete `main` branch
- Merge PRs without approval
- Merge PRs with failing status checks
- Bypass protection rules (even as admin)

## Troubleshooting

### Status Checks Not Appearing

If `Lint and Test` or `Security Scan` don't appear in the status checks list:

1. Push a commit to trigger the CI/CD workflow
2. Wait for the workflow to run at least once
3. Go back to branch protection settings
4. The status checks should now be available in the dropdown

### Cannot Set Up Protection Rules

If you get permission errors:

1. Verify you have admin access to the repository
2. Check you're authenticated with correct account
3. For organization repos, check organization settings

### Automation Not Working

If GitHub Actions workflows aren't running:

1. Check that Actions are enabled:
   - Settings → Actions → General → Allow all actions
2. Verify workflow file syntax:
   ```bash
   gh workflow list
   ```

## Additional Resources

- 📖 [Branch Protection Documentation](BRANCH_PROTECTION.md) - Detailed explanation of all rules
- 🤝 [Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- 🔒 [Security Policy](SECURITY.md) - Security best practices
- 🔧 [Pull Request Template](pull_request_template.md) - PR guidelines

## Support

If you encounter issues:
1. Check [GitHub's Branch Protection Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
2. Open an issue with the `question` label
3. Contact @ujjawalkaushik1110

---

**Setup Time**: ~5-10 minutes  
**Maintenance**: Review quarterly or when team changes
