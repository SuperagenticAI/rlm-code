# Security & Best Practices

## Overview

RLM Code is designed with security in mind to protect your files and personal data. This guide explains our security measures and best practices.

## 🔒 Core Security Principles

### 1. CWD-Only Operation

**ALL data stays in your current working directory.**

```
my-dspy-project/          # Everything in here!
├── .venv/                # Packages
├── .dspy_cache/          # DSPy cache  
├── .dspy_code/           # rlm-code data
├── generated/            # Your code
└── dspy_config.yaml      # Config
```

**Never accesses:**
- ❌ Your home directory (`~/`)
- ❌ User directories (Desktop, Documents, Downloads, Pictures)
- ❌ System directories (`/System`, `/Library`, `/usr`)
- ❌ Files outside your project

### 2. Path Boundary Validation

Every file operation is validated to ensure it stays within your project directory:

- ✅ **Depth limiting**: Maximum 10 levels of recursion
- ✅ **Symlink protection**: Verifies files are actually within project
- ✅ **Permission checks**: Verifies read access before accessing files
- ✅ **Explicit boundaries**: Operations confined to CWD

### 3. Startup Safety Checks

RLM Code validates your working directory before starting:

```bash
# Running from home directory - shows warning
cd ~/
rlm-code
# 🚨 SECURITY WARNING: You are running from your home directory!

# Running from project directory - safe
cd ~/projects/my-project
rlm-code
# ✅ Safe to proceed
```

## 📦 Recommended Setup

### Always Use Virtual Environment IN Project

**Critical for security and isolation:**

```bash
# 1. Create project directory
mkdir my-dspy-project
cd my-dspy-project

# 2. Create venv IN this directory (not elsewhere!)
python -m venv .venv

# 3. Activate it
source .venv/bin/activate  # For fish: source .venv/bin/activate.fish

# 4. Install rlm-code (goes into .venv/ here)
pip install rlm-code

# 5. Run rlm-code (everything stays here!)
rlm-code
```

### Why This Matters

When you create the virtual environment **inside** your project:

1. **Packages are local**: Installed to `./venv/lib/`
2. **Cache is local**: Stored in `./.dspy_code/`
3. **RAG index is local**: Scans only `./.venv/` packages
4. **Everything contained**: One directory = one project

**Result**: Can't access files outside your project! 🔒

## 🚨 What Gets Scanned

### Safe Scanning Scope

RLM Code **only** scans these locations (all in CWD):

1. **Virtual environment packages**:
   ```
   .venv/lib/python3.x/site-packages/dspy/
   .venv/lib/python3.x/site-packages/gepa/
   .venv/lib/python3.x/site-packages/mcp/
   ```

2. **Specific project directories**:
   ```
   ./generated/
   ./modules/
   ./signatures/
   ./programs/
   ./optimizers/
   ./src/
   ```

3. **Python files in root** (non-recursive):
   ```
   ./my_program.py
   ./config.py
   ```

### Never Scanned

- ❌ Home directory
- ❌ Parent directories
- ❌ System directories
- ❌ User directories
- ❌ Hidden files (starting with `.`)
- ❌ Test directories
- ❌ Cache directories
- ❌ `node_modules/`

## 🛡️ Security Features

### Directory Blocking

**Automatic protection** against dangerous directories:

```python
# Blocked directories
- Home directory: ~/
- System dirs: /System, /Library, /usr, /private
- User dirs: ~/Desktop, ~/Documents, ~/Downloads
- Cloud dirs: ~/iCloud, ~/Dropbox, ~/Google Drive
```

### Symlink Attack Prevention

RLM Code verifies files are actually within your project:

```python
# Even if you create a symlink pointing outside:
ln -s ~/Documents/secret.txt ./link.txt

# RLM Code will detect and skip it:
⚠️  Skipping file outside base path: ./link.txt
```

### Depth Limiting

Prevents infinite recursion and excessive scanning:

```python
# Maximum depth: 10 levels
./level1/level2/.../level10/  # ✅ Scanned
./level1/level2/.../level11/  # ❌ Skipped (too deep)
```

### Permission Checks

All file operations check permissions:

```python
# If file is not readable:
if not os.access(file, os.R_OK):
    skip_file()  # Gracefully skip
```

## ⚠️ Security Warnings

### Home Directory Warning

If you run from home directory:

```
🚨 SECURITY WARNING

You are running rlm-code from your home directory!

This is dangerous as it may attempt to scan ALL files
in your home directory, including personal documents,
photos, and sensitive data.

Recommended actions:
1. Create a dedicated project directory
2. Navigate to it: cd ~/my-dspy-project
3. Create virtual environment: python -m venv .venv
4. Activate it: source .venv/bin/activate  # For fish: source .venv/bin/activate.fish
5. Install rlm-code: pip install rlm-code
6. Then run: rlm-code

Press Ctrl+C to exit, or Enter to continue at your own risk...
```

### User Directory Warning

If you run from Desktop, Documents, etc.:

```
⚠️  WARNING

You are running rlm-code from: ~/Desktop

This directory may contain personal files. For safety,
rlm-code will not scan files here.

Recommendation:
Create a dedicated project directory for your DSPy work:
mkdir ~/projects/my-dspy-project && cd ~/projects/my-dspy-project
```

### System Directory Error

If you run from system directory:

```
🚨 CRITICAL ERROR

You cannot run rlm-code from system directory: /usr

This could damage your system. Please run from a user
project directory.
```

## ✅ Best Practices

### 1. Dedicated Project Directories

**Always** create a dedicated directory for each project:

```bash
# Good
~/projects/sentiment-analysis/
~/projects/question-answering/
~/projects/rag-system/

# Bad
~/Desktop/    # Personal files here
~/            # Home directory
/tmp/         # Temp directory
```

### 2. Virtual Environment in Project

**Never** create virtual environment outside your project:

```bash
# ❌ Bad - venv outside project
cd ~/
python -m venv my_global_venv
source my_global_venv/bin/activate
cd ~/Desktop/project
rlm-code

# ✅ Good - venv inside project
cd ~/projects/my-project
python -m venv .venv
source .venv/bin/activate  # For fish: source .venv/bin/activate.fish
rlm-code
```

### 3. Add to .gitignore

Protect sensitive data:

```gitignore
# Virtual environment
.venv/

# rlm-code internal data
.dspy_code/
.dspy_cache/

# Environment variables (contains API keys!)
.env
.env.*
!.env.example

# Secrets
*.key
*.pem
secrets.json
```

### 4. Environment Variables for API Keys

**Never** commit API keys:

```yaml
# ❌ Bad - API key in config
models:
  openai:
    api_key: sk-1234567890abcdef  # DON'T DO THIS!

# ✅ Good - use environment variable
models:
  openai:
    api_key: ${OPENAI_API_KEY}  # From environment
```

```bash
# Set in environment
export OPENAI_API_KEY=sk-1234567890abcdef

# Or use .env file (add to .gitignore!)
echo "OPENAI_API_KEY=sk-1234567890abcdef" > .env
```

### 5. Regular Updates

Keep rlm-code updated for security patches:

```bash
pip install --upgrade rlm-code
```

## 🔐 Data Privacy

### What Gets Cached

**In `.dspy_code/cache/`:**
- Code signatures and metadata
- Function/class names and docstrings
- Project structure information

**NOT cached:**
- API keys or secrets
- Personal files or data
- Anything outside your project

### What Gets Sent to LLMs

When using model connections:
- Your prompts and code generation requests
- Generated code for validation
- Optimization metrics

**NOT sent:**
- Your entire codebase
- API keys or secrets
- Personal files

## 🆘 If Something Goes Wrong

### Suspected Security Issue

If you believe rlm-code is accessing files it shouldn't:

1. **Stop immediately**: Press Ctrl+C
2. **Check location**: Run `pwd` to verify directory
3. **Report**: Open issue on GitHub with details
4. **Check logs**: Look in `.dspy_code/rlm-code.log`

### Clean Up

If you want to remove all rlm-code data:

```bash
# Remove internal data
rm -rf .dspy_code/
rm -rf .dspy_cache/

# Remove virtual environment
rm -rf .venv/

# Or delete entire project
cd ..
rm -rf my-dspy-project/
```

## 📬 Reporting Security Vulnerabilities

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. **Email** security concerns to maintainers
3. **Include** steps to reproduce
4. **Allow** time for fix before disclosure

## ✨ Summary

RLM Code is designed to be **secure by default**:

- ✅ All data in current working directory
- ✅ No home directory access
- ✅ No system directory access
- ✅ Path boundary validation
- ✅ Symlink protection
- ✅ Startup safety checks
- ✅ Comprehensive testing

**Remember**: Always run from a dedicated project directory with a virtual environment inside it!

---

**Questions about security?** [Open an issue](https://github.com/SuperagenticAI/rlm-code/issues) or check our [FAQ](faq.md).
