# Lesson 1 — Your First File Written by Claude Code

> **You'll learn:** How to prompt Claude Code to write code, and — more importantly — how to read and understand code you didn't write.
> **Time:** 45–60 minutes
> **Prerequisites:** Lesson 0 complete.

---

## Why this lesson exists

Most of this course involves reading code that Claude Code writes. That is a skill, and it does not come for free. This lesson builds it deliberately: you will generate a small script, then explain every line back — using Claude Code itself as your tutor when stuck. By the end, you will have a method for reading unfamiliar code that you will use in every lesson that follows.

---

## Concepts

### Claude Code as a code-writing agent

When you give Claude Code a coding task, it does not guess at a solution and hand it to you. It plans a sequence of steps: read relevant files to understand the current state, write or edit the target file, and sometimes run the code to verify it works. You will see each of those steps appear in your terminal as tool calls before the final response.

A key habit Claude Code has is editing existing files rather than rewriting them when possible. If you ask it to add a feature to a 100-line script, it will make a targeted change to those specific lines rather than producing a new 110-line file from scratch. This matters because it preserves work you have already done and makes changes easier to review. In Step 5 of this lesson you will watch this happen directly.

### Reading code as a skill

Professional engineers spend more time reading code than writing it. They read code written by colleagues, by open-source libraries, and — increasingly — by AI tools. Reading unfamiliar code is not a passive activity: it requires tracing execution order, looking up unfamiliar constructs, and building a mental model of what each section does. This course trains that skill deliberately. In every lesson you will be asked to explain code back in your own words before moving on.

### Context window (preview)

Everything Claude Code knows about your repo in a given session it read into its "context window" — a fixed-size buffer of text that the model can reason over at one time. In this lesson the context window is small: just a script and a text file. In Lesson 3, when we introduce RAG, you will see why this limit matters: a large document corpus cannot fit in a context window, and retrieval is how we work around that constraint.

---

## Your task

### Step 1: Start a Claude Code session

From the repo root, run:

```bash
claude
```

---

### Step 2: Give Claude Code this exact prompt

Paste the following into the Claude Code session:

```
Please create a Python script at lessons/01-first-file/word_counter.py that:

1. Takes a file path as a command-line argument.
2. Reads the file's text content.
3. Counts:
   - Total words
   - Total lines
   - The 10 most common words (ignoring case and these stopwords:
     the, a, an, is, of, and, to, in, it, that, this, for, on, with)
4. Prints the results in a clean, readable format.

Requirements:
- Use only the Python standard library.
- Add clear comments explaining each section.
- Handle the case where the file does not exist with a helpful error message.

Also create lessons/01-first-file/sample.txt with a short paragraph of Lorem
Ipsum text so I can run the script.

After creating both files, show me the exact command to run the script and
the expected output.
```

Wait for Claude Code to finish before moving on.

---

### Step 3: Run the script

Use the exact command Claude Code gave you. Confirm the output matches what it described.

If the output does not match, paste the actual output back into the Claude Code session and ask it to explain the discrepancy.

---

### Step 4: Read every line

Open `lessons/01-first-file/word_counter.py` in your code editor. Go through it line by line. For any line you do not understand, ask Claude Code to explain it. Paste the line directly into the session. For example:

```
Explain this line in plain English: `from collections import Counter`
```

Do not skip lines. If something looks obvious but you cannot explain what it does in one sentence, ask anyway.

---

### Step 5: Make a small change

Ask Claude Code to modify the script so it also prints the 5 least common non-stopword words. Use this prompt:

```
Please modify word_counter.py to also print the 5 least common words,
using the same stopword filter. Add it as a new section below the top-10
output.
```

Watch the tool calls. Notice whether Claude Code rewrites the whole file or edits specific lines.

---

## What you should see

A working script that runs against `sample.txt` and prints total words, total lines, the top 10 most common words, and (after Step 5) the 5 least common words.

---

## Understand what happened

Create `docs/lesson-notes/lesson-01.md` and answer these questions:

1. What does `from collections import Counter` do? What problem does it solve?
2. Why does the script use `argparse` (or `sys.argv`) to receive the file path instead of hardcoding it?
3. When Claude Code modified the script in Step 5, what tool call did it use? Did it rewrite the whole file or edit specific lines? Why does that distinction matter?

---

## Homework

1. Paste your final `word_counter.py` into `lesson-01.md` and annotate every section with a comment in your own words explaining what it does.
2. Write down any line you still do not understand after asking Claude Code. Bring those questions to Lesson 2.

---

## Stuck?

**Script doesn't run**
Compare your file to `solution/word_counter.py`. If yours differs, paste both into a Claude Code session and ask it to explain the difference.

**Unsure about a Python concept**
Ask Claude Code directly in the session. That is the intended workflow — using it as a tutor is a core skill this course builds.

---

## What's next

[Lesson 2](../02-claude-md/README.md) — you'll learn about `CLAUDE.md`, the file that shapes how Claude Code behaves in this repo.
