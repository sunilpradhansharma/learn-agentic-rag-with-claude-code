# Setup

---

## Requirements

- Python 3.11 or higher
- `git`
- A code editor (VS Code recommended)
- [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) installed
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

---

## Clone and install

```bash
git clone <repo url>
cd learn-agentic-rag-with-claude-code
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste your API key
```

---

## Verify your setup

Run this one-liner to confirm the `anthropic` library is importable and your API key is loaded:

```bash
python -c "import anthropic, os; from dotenv import load_dotenv; load_dotenv(); key = os.getenv('ANTHROPIC_API_KEY'); print('OK' if key and key != 'your_key_here' else 'ERROR: API key not set')"
```

Expected output:

```
OK
```

If you see `ERROR: API key not set`, open `.env` and make sure `ANTHROPIC_API_KEY` is set to your actual key (not the placeholder).

---

## Install Claude Code

Follow the official installation guide: [https://docs.claude.com/en/docs/claude-code/overview](https://docs.claude.com/en/docs/claude-code/overview)

Once installed, verify it works:

```bash
claude --version
```

You should see a version string printed to the terminal.

---

## Working through the course

1. **Do lessons in order.** Each lesson builds on code and concepts from the previous one. Skipping ahead will leave gaps.
2. **Read each lesson README completely before starting.** The README explains the goal, the background, and exactly what you need to produce.
3. **Use `solution/` folders to unstick yourself, not to skip work.** Looking at a solution after a genuine attempt is a good learning strategy. Copying it without attempting the task is not.
4. **Keep your own notes in `docs/lesson-notes/`.** Write down what confused you, what surprised you, and decisions you made differently from the solution. These notes are for you.

---

## Troubleshooting

**API key not found**
Symptom: `AuthenticationError` or the verify command prints `ERROR: API key not set`.
Fix: Confirm `.env` exists (not just `.env.example`) and contains `ANTHROPIC_API_KEY=<your actual key>`. The file must be in the repo root directory.

**Import errors after `pip install`**
Symptom: `ModuleNotFoundError` for `anthropic` or `dotenv`.
Fix: Make sure your virtual environment is activated — your terminal prompt should show `(venv)`. Run `pip install -r requirements.txt` again inside the activated environment.

**Wrong Python version**
Symptom: Syntax errors on valid code, or `python --version` shows 3.10 or lower.
Fix: Install Python 3.11+ from [python.org](https://www.python.org/downloads/). On macOS you can also use `brew install python@3.11`. Create a fresh venv after upgrading: `python3.11 -m venv venv`.

**`venv\Scripts\activate` not recognized on Windows**
Symptom: PowerShell reports a security error or `activate` is not found.
Fix: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then try activation again. Alternatively use `venv\Scripts\activate.bat` in Command Prompt instead of PowerShell.
