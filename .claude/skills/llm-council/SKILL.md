---
name: llm-council
description: Run a decision through a council of 5 advisor subagents, an anonymous peer review and a final verdict. Use ONLY when the user explicitly asks for it: 'council this', 'run the council', '/llm-council'. Never start it on your own, also not for 'should I ...' questions.
---

# LLM Council (bluesky-gym version)

Adapted from https://github.com/tenfoldmarc/llm-council-skill (MIT), after Karpathy's LLM Council.
Copied from the PAFs_repo version on 2026-11-19 (user request). Changes for this repo: explicit trigger only;
advisors on Opus, reviewers on Sonnet; the main session writes the verdict; terminal answer in ASD-STE100;
no HTML page unless the user asks; transcript in the scratchpad, not in the repo.

All advisors are Claude with different prompts, not different models. Say this once in the verdict.

## Cost

One council = 10 subagents (5 advisors + 5 reviewers), each starts cold. Before step 2, state in one line:
"Council: 5 Opus advisors + 5 Sonnet reviewers." Then go on without asking again (the user asked for the council).

## The five advisors

1. **Contrarian.** Looks for the fatal flaw: what is wrong, what is missing, what will fail.
2. **First Principles.** Asks what the real problem is. Removes assumptions. Can say "this is the wrong question".
3. **Expansionist.** Looks for the upside that the others miss. Does not care about risk.
4. **Outsider.** Has no context about the field. Reacts only to what is written. Finds the "curse of knowledge".
5. **Executor.** Asks only: can this be done, and what is the first concrete step?

## Step 1: frame the question

- Read the context that the question needs, at most 3 files and quick reads: the files the user names, the
  README.md of the sub-project concerned (e.g. `bluesky_rl_transformer/`, `navigation-mappo-rl/`), and the latest
  result summary that the question is about (e.g. a `results/*_summary.json`). Do not read whole large files or logs.
- Write one neutral framed question: the decision, the options, the key numbers (exact, from the files), the
  constraints (the user's rules that apply), what is at stake. Add no opinion.
- If the question is too vague, ask one clarifying question, then go on.

## Step 2: advisors (5 subagents in parallel, one message)

`Agent` with `subagent_type: "general-purpose"`, `model: "opus"`, one per advisor. Prompt:

```
You are [Advisor] on an LLM Council. Your thinking style: [description].
Question:
---
[framed question]
---
Answer from your angle only. Be direct and specific. Do not hedge. Do not read or change any file.
Write 150-300 words in ASD-STE100 (short active sentences, plain words). Keep numbers exact; invent none.
```

## Step 3: peer review (5 subagents in parallel, one message)

Map the 5 answers to letters A-E in a random order. `Agent`, `model: "sonnet"`, 5 reviewers. Prompt:

```
Five advisors answered this question:
---
[framed question]
---
Response A: ... Response E: ...
Answer, by letter, in under 200 words, in ASD-STE100:
1. Which response is the strongest? Why?
2. Which response has the biggest blind spot? What does it miss?
3. What did all five miss?
```

## Step 4: verdict (main session, no subagent)

Write the verdict in the terminal, with the session's `[<focus>]` prefix, in ASD-STE100:

- **Where the council agrees** (high-confidence points)
- **Where the council clashes** (both sides, and why)
- **Blind spots from the peer review**
- **Recommendation** (one clear answer; it may follow a minority if its reasoning is stronger)
- **First step** (one action)

Check every number against the source files. Keep the user's rules (~/.claude/CLAUDE.md): a recommendation never
launches a training run itself (give the `~/bin/notify_run.sh` command with a measured time estimate), and it adds
no new dependency without the user's approval.

## Step 5: transcript

Save `council-<YYYYmmdd-HHMM>.md` (time from `date`) in the session scratchpad: original question, framed
question, the 5 answers, the letter map, the 5 reviews, the verdict. Give its full absolute path in one line.
Make an HTML page only when the user asks (then load `artifact-design` first).
