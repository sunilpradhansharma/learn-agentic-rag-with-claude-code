# Lesson 0 — Setup and First Conversation

> **You'll learn:** How to start a Claude Code session in this repo and verify your environment is working.
> **Time:** 30–45 minutes
> **Prerequisites:** SETUP.md completed.

---

## Why this lesson exists

Before you can learn Claude Code, you need to be able to talk to it in this repo. This lesson verifies that your environment is working and introduces the core idea that Claude Code is not autocomplete — it is an agent that plans and executes tool calls. Everything that follows in this course depends on that distinction.

---

## Concepts

### What is Claude Code?

Claude Code is a command-line agent that reads your files, writes code, and runs shell commands on your behalf. You give it a goal in plain English; it figures out the steps to accomplish that goal and carries them out.

Unlike a chat interface, Claude Code acts directly on your filesystem. When you ask it to write a function, it writes the file. When you ask it to run tests, it runs them. The results appear in your terminal and in your files, not just in a conversation window.

This makes Claude Code "agentic" in a specific sense: it does not just predict the next word. It plans a sequence of steps, executes tools to carry out each step, observes the results, and adjusts. That planning-and-executing loop is what the word "agent" means throughout this course.

### What is a tool call?

When you ask Claude Code to do something, it decides which tool is appropriate and invokes it. The main tools it uses in this course are: **Read** (reads a file's contents), **Write** (creates or overwrites a file), **Edit** (makes targeted changes to an existing file), **Bash** (runs a shell command), and **Glob** / **Grep** (searches for files or text patterns).

You will see these tool calls appear in your terminal in real time, before Claude Code gives you a final response. Watching them is not noise — it is the most direct way to understand how an agent reasons and acts. Pay attention to the order and the choices Claude Code makes. That sequence is part of what you are learning.

---

## Your task

### Step 1: Verify your environment

Run the following commands and confirm the output matches what is shown:

```bash
python --version
```
Expected: `Python 3.11.x` or higher.

```bash
claude --version
```
Expected: a version number printed to the terminal (for example, `1.x.x`).

If either command fails, return to SETUP.md before continuing.

---

### Step 2: Start your first Claude Code session

From the repo root, run:

```bash
claude
```

You should see a Claude Code prompt appear in your terminal. This is an interactive session: you type prompts, Claude Code responds, and the session persists until you end it.

---

### Step 3: Ask Claude Code to explore the repo

Paste the following prompt into the Claude Code session exactly as written:

```
Please look around this repository and tell me:
1. What is this project?
2. What top-level folders exist and what is each for?
3. What files exist at the root?
Keep your answer brief and beginner-friendly.
```

Wait for Claude Code to finish before moving on.

---

### Step 4: Observe what Claude Code does

While Claude Code works, watch the terminal carefully. Notice:

- Claude Code reads `README.md` first.
- It then reads `CLAUDE.md`.
- It may explore directories or read additional files.
- Each of these actions is a tool call, printed to your terminal before the final response.

You do not need to do anything during this step. Just watch.

---

### Step 5: End the session

When you are ready, type `/exit` and press Enter, or press `Ctrl+D`.

---

## What you should see

A response from Claude Code describing the course, the 7 phases, and the folder structure. Before that response, you should have seen tool calls for reading files — at minimum `README.md` and `CLAUDE.md`, possibly others depending on what Claude Code chose to explore.

---

## Understand what happened

Create the file `docs/lesson-notes/lesson-00.md` and answer these questions in it:

1. List the tool calls Claude Code made, in order. What was each one for?
2. What information did Claude Code use to describe the project — which files, and what from each?
3. What surprised you about how Claude Code worked?

There are no right or wrong answers here. The goal is to make you look carefully at what happened, not just at the output.

---

## Homework

1. Complete `docs/lesson-notes/lesson-00.md` with your answers above.
2. Run `claude` one more time and ask it: `"What is CLAUDE.md and why does this project have one?"` Read its answer carefully — this is previewing Lesson 2.
3. Do **not** edit any files in the repo yet. This lesson is observation only.

---

## Stuck?

**`claude` command not found**
Revisit SETUP.md's "Install Claude Code" section. Make sure the install completed without errors and that your terminal has been restarted since installation.

**API key errors**
Check that `.env` exists at the repo root (not just `.env.example`) and that it contains `ANTHROPIC_API_KEY=<your actual key>`. Run the verification one-liner from SETUP.md to confirm.

**Claude Code hangs or gives errors**
Check your internet connection and verify your API key is valid at [console.anthropic.com](https://console.anthropic.com).

---

## What's next

[Lesson 1](../01-first-file/README.md) — you'll prompt Claude Code to write your first Python file and learn how to read code you didn't write.

---

*This lesson has no solution/ folder because no code is produced.*
