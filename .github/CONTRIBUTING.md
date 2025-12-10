# Contributing to Tunix Reasoning Agent

Thank you for your interest in contributing to the Tunix Reasoning Agent! We welcome contributions from the community.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/ujjawalkaushik1110/tunix-reasoning-agent/issues)
2. If not, create a new issue using the Bug Report template
3. Provide detailed information to help us reproduce the issue

### Suggesting Features

1. Check if the feature has already been suggested
2. Create a new issue using the Feature Request template
3. Clearly describe the feature and its benefits

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/ujjawalkaushik1110/tunix-reasoning-agent.git
   cd tunix-reasoning-agent
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Write clear, commented code
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run linting
   flake8 src/
   black src/
   
   # Run tests
   pytest tests/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```
   
   Commit message format:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes
   - `Test:` for test additions/changes

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Use the Pull Request template
   - Link related issues
   - Provide a clear description of changes

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage

### Documentation

- Update README.md if adding new features
- Add inline comments for complex logic
- Update docstrings when changing function behavior

## Branch Protection Rules

The `main` branch is protected with the following rules:

1. **Required Reviews**: At least 1 approval required
2. **Status Checks**: CI/CD pipeline must pass
3. **No Direct Push**: All changes via Pull Request
4. **Up-to-date Branch**: Branch must be up-to-date before merging
5. **Code Owner Review**: Changes must be reviewed by CODEOWNERS

## Review Process

1. Automated checks run on all PRs
2. Code review by maintainers
3. Feedback addressed by contributor
4. Approval and merge by maintainers

## Questions?

If you have questions, feel free to:
- Open an issue with the question label
- Reach out to @ujjawalkaushik1110

Thank you for contributing! 🎉
