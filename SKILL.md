# goldfish-memory

Give yourself (a coding agent) persistent, searchable memory across sessions. Everything stays on the user's machine in one SQLite file; this tool makes no network calls.

## When to use

Use this whenever you do non-trivial work in a codebase: capture decisions, fixes, and gotchas as you go, and pull relevant memories into your context at the start of every session or task. If the user says "remember this", that is a capture.

## Setup

```bash
pip install git+https://github.com/DhanushNehru/goldfish-memory.git
cd <project> && goldfish init        # creates .goldfish/memory.db
```

The embedding model is bundled; no provisioning, no API keys, works offline.

## Core loop

1. **Session start** - inject relevant context before acting:
   ```bash
   goldfish inject "<what you are about to do>" -k 6 --max-tokens 800
   ```
   Prepend the printed `## Relevant memory` block to your prompt context.
2. **During work** - capture things future sessions must not lose:
   ```bash
   goldfish add --kind decision "Chose X over Y because <reason>"
   goldfish add --kind error "Symptom Z was caused by <root cause>; fixed in <file>"
   goldfish add --kind note "API rate limit is per key, not per IP"
   ```
   Capture decisions with reasons, errors with root causes, and non-obvious facts. Do not capture secrets, credentials, or personal data.
3. **Look before you decide** - search before changing behavior:
   ```bash
   goldfish search "<the thing you are about to change>" -k 5
   ```
4. **Maintenance** - compaction runs automatically; manual control:
   ```bash
   goldfish compact --dry-run     # preview summarization of old memories
   goldfish pin <id>              # protect a memory from compaction
   goldfish forget <id>           # permanently delete a memory
   goldfish recent -n 10          # what was captured lately
   goldfish stats                 # store size and location
   ```

## Python API

```python
from goldfish_memory import MemoryEngine

with MemoryEngine() as engine:                     # .goldfish/memory.db by default
    engine.capture("Chose X over Y because ...", kind="decision")
    hits = engine.search("why did we pick X?", k=5)
    block = engine.inject("refactor the storage layer", k=6, max_tokens=800)
    print(block.text)                              # markdown, token-budgeted
```

Custom model: pass any `Embedder` implementation, or set `GOLDFISH_MODEL_DIR` to a directory with `model.onnx` + `tokenizer.json`. Models are never downloaded at runtime.

## Rules

- Keep memories factual and small; one idea per memory.
- Never capture secrets, tokens, or personal data.
- Prefer `decision`/`error` kinds - they age best.
- The store is project-local by default (`.goldfish/`); commit the directory to share memory with teammates, or gitignore it for personal memory.
