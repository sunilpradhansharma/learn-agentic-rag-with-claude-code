# Lesson 2 — CLAUDE.md: Teaching Claude Code About Your Project

> **You'll learn:** What CLAUDE.md is, how Claude Code uses it, and how to extend it to shape Claude Code's behavior in your own projects.
> **Time:** 45–60 minutes
> **Prerequisites:** Lessons 0–1 complete.

---

## Why this lesson exists

Claude Code can work in any repository, but it works best when you tell it about your project's conventions, constraints, and purpose. CLAUDE.md is where you do that. Without it, Claude Code has to infer context from your code and file names alone — it will often get things right, but it will also make decisions you did not intend. With a well-written CLAUDE.md, you encode those decisions once and they apply to every session automatically. It is the most important file you will write in any Claude Code project.

---

## Concepts

### What CLAUDE.md is

CLAUDE.md is a Markdown file that Claude Code reads automatically at the start of every session. It acts as project-level memory: conventions Claude Code should follow, commands it should know about, constraints it should respect, and the communication style it should use. Once it is in place, you do not have to repeat yourself in every prompt — Claude Code already knows.

Think of it as onboarding documentation for a new team member, except the team member is Claude Code. A good onboarding doc does not try to document everything; it captures the things that are non-obvious, the decisions that have already been made, and the mistakes that are easy to make. CLAUDE.md should do the same.

### Where CLAUDE.md files live

Claude Code looks for CLAUDE.md files in three places, each with a different scope:

- **Repo root** (`CLAUDE.md`) — applies to the entire repository. The one in this repo is an example.
- **Subdirectory** (e.g. `lessons/04-loading-chunking/CLAUDE.md`) — applies only when Claude Code is working in that folder. Useful for scoping conventions to a specific part of the codebase.
- **User home** (`~/.claude/CLAUDE.md`) — applies to every project on your machine. Good for personal preferences that have nothing to do with any specific repo.

### What belongs in CLAUDE.md

- **Project overview** — what the project is and who it is for
- **Directory layout** — what each folder contains and why
- **Coding conventions** — language version, style rules, which tools to use
- **Common commands** — how to install, test, and run things
- **Constraints** — explicit things Claude Code should not do in this repo
- **Communication style** — tone, explanation depth, audience assumptions

### What does NOT belong

- **Secrets, API keys, or passwords** — those go in `.env`, which Claude Code is instructed never to read or edit
- **Frequently changing content** — if it belongs in code or a config file, put it there; CLAUDE.md is for stable conventions
- **Long prose** — CLAUDE.md rewards brevity; a long file is harder to load into context and harder to maintain

---

## Your task

### Step 1: Read the existing CLAUDE.md

Open `CLAUDE.md` at the repo root and read it completely. Notice each section: Overview, Repository Layout, Conventions, Teaching Style, Commands, Files I Should Never Edit, and Current Course Progress. Each section exists for a specific reason. As you read, think about what a Claude Code session would do differently because of each bullet.

---

### Step 2: Observe CLAUDE.md in action

Start a Claude Code session:

```bash
claude
```

Then ask:

```
In one paragraph, what do you know about how to work in this repo?
Cite the source of that knowledge.
```

Claude Code should reference `CLAUDE.md` directly in its answer. That citation is proof the file is being read and applied — not guessed at from the code.

---

### Step 3: Extend CLAUDE.md together

Give Claude Code this prompt in the same session:

```
I want to add a new section to CLAUDE.md titled "## Student Interaction Rules".
Propose 4–6 bullet points that would be appropriate to add, given this repo
is a teaching resource for learners new to AI/ML.
Do NOT edit the file yet — just propose the bullets.
```

Read each proposed bullet carefully. For each one, ask yourself:

- Is this rule specific enough to change behavior? ("Be helpful" is not; "Always show the exact command to run next" is.)
- Is it appropriate for the audience — beginners who know basic Python?
- Would it ever conflict with another rule already in the file?

Discuss or revise the proposals in the session before accepting them.

---

### Step 4: Make the edit

Once you are satisfied with the proposed bullets, tell Claude Code:

```
Good. Please add the "## Student Interaction Rules" section to CLAUDE.md
with those bullets. Place it after "## Teaching Style".
```

Open `CLAUDE.md` after the edit and confirm the new section is there and reads the way you intended.

---

### Step 5: Test the new behavior

Exit the current session:

```
/exit
```

Start a new one:

```bash
claude
```

Ask:

```
What are the student interaction rules in this project?
```

Claude Code should recite the rules from the section you just added. A new session reads CLAUDE.md fresh — this is the proof that your additions are persistent and applied automatically, not just remembered from the previous conversation.

---

### Step 6: Try the `#` shortcut

Inside a Claude Code session, type this as a standalone message (the `#` must be the first character):

```
# Remember: this repo uses pip and venv, not uv or poetry.
```

The `#` prefix tells Claude Code to append the line directly to CLAUDE.md without asking. Open `CLAUDE.md` and confirm the line was added. This shortcut is useful for quick, low-ceremony additions — facts you want Claude Code to remember in future sessions without going through a full edit workflow.

---

## What you should see

- An extended `CLAUDE.md` with a new `## Student Interaction Rules` section containing 4–6 bullets you reviewed and approved.
- A line appended at the bottom via the `#` shortcut.
- A new Claude Code session that recites your new rules without you having to repeat them.

---

## Understand what happened

Create `docs/lesson-notes/lesson-02.md` and answer these questions:

1. Why does CLAUDE.md need to be short? What happens to Claude Code's behavior if it grows too long?
2. If CLAUDE.md is a new team member's onboarding doc, what is the `#` shortcut analogous to? (Think about the workflow difference, not just the outcome.)
3. Compare Claude Code's answer in Step 2 (before your edit) to its answer in Step 5 (after). What changed, and what stayed the same?

---

## Homework

1. Find a public open-source project that uses Claude Code and read its CLAUDE.md. (The Anthropic cookbook repositories are a good starting point.) In `lesson-02.md`, note two things it includes that this repo's CLAUDE.md does not, and explain whether they would be useful here.
2. Write one paragraph in `lesson-02.md`: "If I started my own Python project tomorrow, what would my CLAUDE.md look like?" Be specific — list the actual sections you would include.

---

## Stuck?

**Claude Code ignores CLAUDE.md**
Start a fresh session. CLAUDE.md is read at session start, not continuously updated during a running session. Changes you make mid-session take effect in the next session.

**The `#` shortcut doesn't work**
Confirm the `#` is the very first character of the message with no leading space. If it still doesn't work, edit `CLAUDE.md` directly in your editor — the outcome is identical.

**Want a reference**
See `solution/CLAUDE.md.example` for a version of `CLAUDE.md` that reflects what this file should look like after this lesson's edits.

---

## What's next

[Lesson 3](../03-tiny-rag/README.md) — you'll build your first RAG system: about 80 lines of Python that demonstrates the entire retrieve-then-generate pattern.
