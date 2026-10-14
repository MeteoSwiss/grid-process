# Contributing Guide

Thank you for your interest in contributing to **grid-process**! We welcome contributions from the community and are grateful for your support.

## Project Purpose
This library is developed primarily to support the operational, scientific, and research needs of MeteoSwiss.
While the project is open source and welcomes external contributions, it is intended to be released, maintained, and distributed under the MeteoSwiss name. Contributions should align with this mission and the long‑term goals of the organization.

As equivalent functionality becomes available in the [ECMWF earthkit ecosystem](https://earthkit.ecmwf.int/), parts of this library, potentially all of it, will be gradually deprecated in favour of earthkit. Before starting work on a larger new feature, please open an issue to discuss it first, as functionality that overlaps with earthkit may be better contributed there.

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please read the full [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and what you expected to see
- Include any error messages or stack traces
- Note your environment (Python version, OS, etc.)

You can report bugs using our **bug report template**.

### Suggesting Features

Feature suggestions are welcome! Please create an issue using our *feature request template* and include:

- A clear and descriptive title
- A detailed description of the proposed feature
- Explain why this feature would be useful
- Provide examples of how it would be used

### Contributing Code

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes following our coding standards
4. Write or update tests as needed
5. Ensure all tests pass and code quality checks succeed
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher (up to 3.13)
- Poetry (for dependency management)
- Git

### Setting Up Your Environment

1. Clone your fork of the repository:

   ```console
   $ git clone https://github.com/YOUR-USERNAME/grid-process.git
   $ cd grid-process
   ```

2. Install dependencies:

   ```console
   $ poetry install
   ```

3. Verify your setup by running the tests:

   ```console
   $ poetry run pytest
   ```

## Code Style and Quality

This project uses several tools to maintain code quality:

### Formatting with Ruff

We use [Ruff](https://docs.astral.sh/ruff/formatter) for code formatting with a 120-character line limit.

Format your code before committing:

```console
$ poetry run ruff format
```

### Linting with Pylint

Run pylint to check for code quality issues:

```console
$ poetry run pylint grid_process
```

The project configuration is in `pyproject.toml`. We disable certain docstring requirements but maintain other quality standards.

### Type Checking with Mypy

We enforce type hints throughout the codebase:

```console
$ poetry run mypy grid_process
```

All function definitions must include type annotations.

## Testing Guidelines

### Writing Tests

- Write unit tests for all new code
- Place tests in the `test/` directory
- Follow the existing test structure and naming conventions
- Use descriptive test names that explain what is being tested
- Aim for high code coverage (check with `pytest-cov`)

### Running Tests

Run all tests:

```console
$ poetry run pytest
```

Run tests with coverage:

```console
$ poetry run pytest --cov=grid_process --cov-report=html
```

## Documentation

- Update documentation for any changed functionality
- Add docstrings to new functions, classes, and modules
- Follow the existing documentation style
- Build documentation locally to verify changes:

  ```console
  $ poetry run sphinx-build doc doc/_build
  ```

## Pull Request Process

1. **Update the CHANGELOG**: Add a brief description of your changes in **CHANGELOG**

2. **Update AUTHORS**: If this is your first contribution, add your name to the **AUTHORS** file

3. **Create a Pull Request**: Use our **pull request template** and ensure you:

   - Provide a clear description of the changes
   - Reference any related issues (e.g., "Closes #123")
   - Complete the checklist in the PR template
   - Ensure all CI checks pass

4. **Code Review**: A maintainer will review your PR. Be prepared to make changes based on feedback

5. **Merge**: Once approved and all checks pass, a maintainer will merge your PR

### Commit Messages

Write clear, concise commit messages:

- Use the imperative mood ("Add feature" not "Added feature")
- Start with a capital letter
- Keep the first line under 50 characters
- Add a blank line followed by a detailed description if needed

Example:

```text
Add validation for input parameters

- Validate that input is not None
- Raise ValueError for invalid types
- Add unit tests for validation logic
```

## License

By contributing to this project, you agree that your contributions will be licensed under the BSD-3-Clause License.

## Questions?

If you have questions about contributing, please:

- Check existing issues and discussions
- Review the documentation
- Create a new issue with your question

Thank you for contributing to grid-process!