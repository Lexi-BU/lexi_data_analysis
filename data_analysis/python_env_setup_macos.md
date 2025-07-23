# 🐍 Setting Up a Python Environment on macOS (Apple Silicon or Intel)

This guide provides a concise and reproducible workflow for setting up a Python environment, managing Python versions, and installing dependencies using `requirements.txt` or `pyproject.toml`.

---

## 📦 Prerequisites

Ensure the following tools are installed:

- **[Homebrew](https://brew.sh/)**
- **[pyenv](https://github.com/pyenv/pyenv)**
- **[virtualenv](https://virtualenv.pypa.io/)** (optional if using `venv`)
- **Xcode Command Line Tools**
```bash
xcode-select --install
```

---

## 📌 Install `pyenv` and Python Versions

```bash
# Install pyenv using Homebrew
brew update
brew install pyenv

# Add pyenv to your shell config (~/.zshrc or ~/.bashrc)
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Restart your terminal or source the config
source ~/.zprofile
source ~/.zshrc

# List all installable Python versions
pyenv install --list

# Install Python 3.10.x
pyenv install 3.10.13

# (Optional) Install another version (e.g., 3.11.x)
pyenv install 3.11.9
```

---

## 🛠️ Set Local or Global Python Version

```bash
# Set Python 3.10 globally
pyenv global 3.10.13

# Or use Python 3.10 only in a project folder
cd your-project/
pyenv local 3.10.13
```

---

## 🧪 Create Virtual Environment

```bash
# Using built-in venv
python -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 🧾 Installing Dependencies

### If you have `requirements.txt`
```bash
pip install -r requirements.txt
```

### If you have `pyproject.toml`
```bash
pip install pipx
pipx install poetry  # or pip install poetry
poetry install
```

---

## 🧪 Check All Installed Python Versions on macOS

```bash
# List all pyenv-installed versions
pyenv versions

# List all Python executables on the system
ls -l /usr/local/bin/python*
ls -l /opt/homebrew/bin/python*

# Or use 'which' or 'where'
which -a python
which -a python3
```

---

## 📁 Sample `README.md` Template for Your Project

```markdown
# Python Environment Setup

## Python Version

This project uses **Python 3.10** managed via `pyenv`.

```bash
pyenv install 3.10.13
pyenv local 3.10.13
```

## Creating the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Installing Dependencies

### Using requirements.txt

```bash
pip install -r requirements.txt
```

### Using pyproject.toml (with Poetry)

```bash
pipx install poetry
poetry install
```

## Notes

- Ensure `pyenv` is initialized in your shell.
- Use `pyenv versions` to verify installed Python versions.
- Activate your virtual environment before running or installing anything.

---

## Troubleshooting

# 📦 Creating a Python 3.10 Environment When System Python Is 3.9

If your macOS system has Python 3.9 as the default, but you want to use Python 3.10 in an isolated environment, follow these steps:

---

## ✅ 1. Install Python 3.10 via `pyenv`

```bash
# Install pyenv (if not already installed)
brew install pyenv

# Install Python 3.10.x
pyenv install 3.10.13
```

---

## ✅ 2. Set Python 3.10 Locally in Your Project Folder

```bash
cd /path/to/your/project
pyenv local 3.10.13
```

> This ensures the `python` command in this folder refers to Python 3.10.13 only.

---

## ✅ 3. Create a Virtual Environment with Python 3.10

```bash
python -m venv .venv
source .venv/bin/activate
```

> ✅ Since `pyenv local 3.10.13` is active, the `python` used to create the virtual environment will be Python 3.10.

---

## ✅ 4. Verify the Python Version

```bash
python --version
# Output should be: Python 3.10.13
```

---

🎉 You now have a Python 3.10 virtual environment, isolated from your system’s Python 3.9 installation.
---

# 🔁 Creating a Python 3.10 Environment When Python 3.11 Is Installed

If Python 3.11 is installed on your system, but you want to create a virtual environment using **Python 3.10**, here is how you can do it:

---

## ✅ 1. Install Python 3.10 with `pyenv`

```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.10.x
pyenv install 3.10.13
```

---

## ✅ 2. Set Python 3.10 in Your Project Directory

```bash
cd /path/to/your/project
pyenv local 3.10.13
```

> This makes Python 3.10 the default version only within this folder.

---

## ✅ 3. Create a Virtual Environment with Python 3.10

```bash
python -m venv .venv
source .venv/bin/activate
```

> 🔧 Even if Python 3.11 is globally installed, using `pyenv local` ensures that `python` refers to 3.10 here.

---

## ✅ 4. Confirm the Environment Uses Python 3.10

```bash
python --version
# Output should be: Python 3.10.13
```

---

🎯 You now have a Python 3.10 virtual environment, even though your system or global Python version is 3.11.
---

