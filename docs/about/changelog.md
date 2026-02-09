# Changelog

All notable changes to RLM Code will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-11-21

### Added

- Initial release of RLM Code (formerly DSPy CLI)
- Interactive mode with natural language interface
- Slash commands for all operations
- Code generation for Signatures, Modules, and complete programs
- Model connection support (Ollama, OpenAI, Anthropic, Gemini)
- Built-in MCP client integration
- Real GEPA optimization support
- Codebase RAG for project understanding
- Code validation and sandboxed execution
- Project initialization (fresh and existing projects)
- Session management with auto-save
- Export/import functionality
- Comprehensive documentation with MkDocs

### Changed

- Renamed from "DSPy CLI" to "RLM Code" with tagline "Claude Code for DSPy"
- Updated branding to Superagentic AI
- Made all commands interactive-only (slash commands)
- Moved codebase indexing to `/init` command
- Enhanced error handling and user feedback

### Fixed

- Context sharing between interactive session and slash commands
- Codebase indexing to use installed packages instead of reference directory
- Permission handling for restricted environments
- Model connection error messages

---

## [Unreleased]

### 🎯 MAJOR CHANGE: Everything in CWD (2024-11-24)

**BREAKING CHANGE**: All rlm-code data now stored in current working directory for better isolation and portability.

#### What Changed
- **Cache location**: Moved from `~/.dspy_cli/cache/` to `.dspy_code/cache/` in CWD
- **Session data**: Moved from `~/.dspy_cli/sessions/` to `.dspy_code/sessions/` in CWD
- **Optimization workflows**: Moved from `~/.dspy_cli/optimization/` to `.dspy_code/optimization/` in CWD
- **Export history**: Moved from `~/.dspy_cli/exports/` to `.dspy_code/exports/` in CWD
- **Command history**: Moved from `~/.dspy_code_history` to `.dspy_code/history.txt` in CWD

#### Why This Matters
✅ **True CWD-only operation**: Everything (code, cache, packages) stays in your project directory  
✅ **Enhanced security**: No home directory access at all  
✅ **Perfect isolation**: Each project is completely self-contained  
✅ **Easy cleanup**: Delete project folder to remove everything  
✅ **Portability**: Zip entire project directory to share or backup  
✅ **Simplicity**: One directory = one project with all its data  

#### Recommended Setup
**CRITICAL**: Always create virtual environment INSIDE your project:

```bash
mkdir my-dspy-project
cd my-dspy-project
python -m venv .venv          # Creates .venv IN project
source .venv/bin/activate  # For fish: source .venv/bin/activate.fish
pip install rlm-code dspy
rlm-code                      # Everything stays in my-dspy-project/
```

#### Migration
Your old data in `~/.dspy_cli/` will not be automatically migrated. This is intentional - each project now has its own isolated cache and sessions.

### 🚨 SECURITY FIXES (2024-11-24)

#### File System Access Protection
- **CRITICAL**: Fixed RAG indexer recursively scanning entire file system including personal directories
- **CRITICAL**: Added safety checks to prevent accessing iCloud Photos, Documents, Downloads, and other user directories
- **CRITICAL**: Implemented strict path boundary validation to limit operations to current working directory only

#### Security Measures Added
1. **Directory Safety Validation**
   - Blocks scanning of home directory (`~/`)
   - Blocks scanning of system directories (`/System`, `/Library`, `/usr`, `/private`)
   - Blocks scanning of user directories (Desktop, Documents, Downloads, Pictures, Photos, Music, Movies)
   - Blocks scanning of immediate home subdirectories

2. **Path Boundary Protection**
   - All file operations validated to stay within project directory
   - Symlink attack prevention - verifies files are actually within project
   - Maximum depth limiting (10 levels) to prevent infinite recursion
   - Explicit permission checks before accessing any file

3. **Startup Safety Checks**
   - Critical warning displayed if running from home directory
   - Warning displayed if running from user directories
   - Error and exit if running from system directories
   - Recommendations to use dedicated project directories with virtual environments

4. **Limited Scanning Scope**
   - RAG indexer only scans:
     - Installed packages in current virtual environment (not system-wide)
     - Specific project directories: `generated/`, `modules/`, `signatures/`, `programs/`, `optimizers/`, `src/`
     - Python files in current directory (non-recursive)
   - Excludes test files, cache directories, hidden files automatically

### Added
- Comprehensive security test suite (`tests/test_security.py`)
- Startup directory safety checks
- Path boundary validation in all file operations
- Project-specific command history
- Complete CWD isolation
- Security documentation page

### Changed
- **BREAKING**: All rlm-code internal data now in `.dspy_code/` in CWD instead of `~/.dspy_cli/`
- RAG indexer strictly limited to project directory
- Project scanner respects path boundaries
- All `rglob()` operations use safe wrappers with depth limits
- `.gitignore` updated to include `.dspy_code/` and `.dspy_cache/`
- Documentation updated to emphasize virtual environment in project directory

### Planned

- Enhanced MCP tool integration
- Additional DSPy module templates
- Improved optimization workflows
- Extended evaluation capabilities
- Performance optimizations

---

**For detailed development history, see [GitHub Commits](https://github.com/superagentic-ai/rlm-code/commits/main)**
