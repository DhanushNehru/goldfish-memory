# API Reference

Goldfish Memory provides a Python API for storing, searching, and managing memories.

The API documentation below is generated from the project's Python docstrings using mkdocstrings.

## Installation

Install Goldfish Memory in your Python environment:

```bash
pip install -e .


## Basic usage

Import the public API:

```python
from goldfish_memory import Config, MemoryEngine
```

Initialize the configuration and memory engine:

```python
config = Config()
engine = MemoryEngine(config)
```

> This is an illustrative example. Refer to the class documentation below for the actual constructor parameters and methods.

## Public API

The following symbols are exported by the `goldfish_memory` package.

::: goldfish_memory
    options:
      show_root_heading: false
      show_source: false
      members:
        - Config
        - Embedder
        - MiniLMOnnxEmbedder
        - ModelNotProvisionedError
        - MemoryEngine
        - CompactionResult
        - InjectionBlock
        - Memory
        - SearchResult

## Configuration

::: goldfish_memory.config
    options:
      show_source: false
      members:
        - Config

## Memory engine

::: goldfish_memory.engine
    options:
      show_source: false
      members:
        - MemoryEngine
        - CompactionResult
        - InjectionBlock

## Memory models

::: goldfish_memory.models
    options:
      show_source: false
      members:
        - Memory
        - SearchResult

## Compaction

::: goldfish_memory.compaction
    options:
      show_source: false

## Embeddings

::: goldfish_memory.embeddings
    options:
      show_source: false

## Further reading

- [Quickstart](quickstart.md)
- [CLI Reference](cli.md)
- [GitHub repository](https://github.com/DhanushNehru/goldfish-memory)