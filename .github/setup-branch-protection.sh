#!/bin/bash

# Branch Protection Setup Script
# This script configures branch protection rules for the main branch using GitHub CLI
# 
# Prerequisites:
# - GitHub CLI (gh) must be installed: https://cli.github.com/
# - User must be authenticated: gh auth login
# - User must have admin permissions on the repository
#
# Usage:
#   ./setup-branch-protection.sh [REPO] [BRANCH]
#   
# Examples:
#   ./setup-branch-protection.sh
#   ./setup-branch-protection.sh owner/repo main

set -e

# Default values
DEFAULT_REPO="ujjawalkaushik1110/tunix-reasoning-agent"
DEFAULT_BRANCH="main"

# Accept command-line arguments or use defaults
REPO="${1:-$DEFAULT_REPO}"
BRANCH="${2:-$DEFAULT_BRANCH}"

echo "🔒 Setting up branch protection rules for $REPO..."
echo "Branch: $BRANCH"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is installed and authenticated"
echo ""

# Apply branch protection rules
echo "📋 Applying branch protection rules..."

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/$REPO/branches/$BRANCH/protection" \
  --input .github/branch-protection-config.json

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Branch protection rules successfully applied!"
    echo ""
    echo "📌 Configured settings:"
    echo "   - Require pull request reviews (1 approval required)"
    echo "   - Require review from code owners"
    echo "   - Dismiss stale reviews on new commits"
    echo "   - Require status checks (Lint and Test, Security Scan)"
    echo "   - Require branches to be up-to-date before merging"
    echo "   - Require conversation resolution before merging"
    echo "   - Block force pushes"
    echo "   - Block branch deletion"
    echo "   - Enforce rules for administrators"
    echo ""
    echo "🎉 Main branch is now protected!"
    echo ""
    echo "View settings: https://github.com/$REPO/settings/branches"
else
    echo ""
    echo "❌ Failed to apply branch protection rules."
    echo "Please check:"
    echo "   1. You have admin permissions on the repository"
    echo "   2. The repository exists and is accessible"
    echo "   3. The branch name is correct"
    exit 1
fi
