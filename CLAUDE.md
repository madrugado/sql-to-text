# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository contains a single installation script (`claude_code_env.sh`) for setting up Claude Code CLI on Linux/macOS systems.

## Script Overview

`claude_code_env.sh` is a self-contained installer that:

1. **Node.js Setup** - Installs Node.js via nvm (version 22 by default) if not present or if version < 18
2. **Claude Code Installation** - Installs `@anthropic-ai/claude-code` globally via npm
3. **Configuration** - Sets up `~/.claude/settings.json` with custom API settings:
   - Uses ZHIPU API (bigmodel.cn) as the base URL
   - Sets API timeout to 3000000ms
   - Disables non-essential traffic

## Running the Script

```bash
./claude_code_env.sh
```

The script will prompt for a ZHIPU API key, which can be obtained from: https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys

## Constants (Modifiable)

Key variables defined at the top of the script:
- `NODE_MIN_VERSION=18` - Minimum Node.js version required
- `NODE_INSTALL_VERSION=22` - Node.js version to install
- `API_BASE_URL="https://open.bigmodel.cn/api/anthropic"` - Custom API endpoint
- `API_TIMEOUT_MS=3000000` - Request timeout in milliseconds
