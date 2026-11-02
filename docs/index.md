# Goldfish Memory

**Local-first memory for AI agents.**

Goldfish Memory helps AI agents capture, search, and manage useful memories locally. It uses a local database and semantic search to retrieve relevant context when needed.

## Why Goldfish Memory?

- **Local-first:** Your memories stay on your machine.
- **Offline-oriented:** Designed to work without relying on external services.
- **Semantic search:** Retrieve memories based on meaning.
- **Memory management:** Pin, search, compact, and forget memories.
- **Agent integration:** Inject relevant memories into task context.

## Quickstart

Install Goldfish Memory in your Python environment:

```bash
pip install -e .
```

Initialize the database:

```bash
goldfish init
```

Add a memory:

```bash
goldfish add "I prefer concise explanations with practical examples."
```

Search memories:

```bash
goldfish search "explanation preferences"
```

## Documentation

- [Quickstart](quickstart.md)
- [CLI Reference](cli.md)
- [API Reference](api.md)

## Project

[GitHub repository](https://github.com/DhanushNehru/goldfish-memory)