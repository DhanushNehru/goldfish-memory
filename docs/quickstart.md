# Quickstart

This guide helps you get started with Goldfish Memory.

## Requirements

- Python 3.10 or newer
- Git
- A terminal

## 1. Clone the repository

```bash
git clone https://github.com/DhanushNehru/goldfish-memory.git
cd goldfish-memory
```

## 2. Install the project

Install Goldfish Memory in editable mode:

```bash
pip install -e .
```

This installs the package and its dependencies.

## 3. Initialize Goldfish Memory

```bash
goldfish init
```

This initializes the local memory database and configuration.

## 4. Add your first memory

```bash
goldfish add "I prefer concise explanations with practical examples."
```

Goldfish Memory stores this information for later retrieval.

You can also provide memory content through standard input.

## 5. Search your memories

```bash
goldfish search "explanation preferences"
```

Limit the number of results:

```bash
goldfish search "explanation preferences" -k 5
```

## 6. Inject relevant memories

Retrieve relevant memories for a task:

```bash
goldfish inject "Explain Python dictionaries"
```

Limit the number of memories:

```bash
goldfish inject "Explain Python dictionaries" -k 5
```

## 7. View recent memories

```bash
goldfish recent
```

Display more memories:

```bash
goldfish recent -n 20
```

## 8. View memory statistics

```bash
goldfish stats
```

## Next steps

- Read the [CLI Reference](cli.md) for all available commands.
- Explore the [API Reference](api.md) to integrate Goldfish Memory into Python applications.