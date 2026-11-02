
## `goldfish compact`

Summarize and archive older memories.

```bash
goldfish compact
```

### Options

| Option | Description |
|---|---|
| `--older-than DAYS` | Select memories older than the specified number of days |
| `--dry-run` | Preview changes without applying them |
| `-h`, `--help` | Show command help |

## `goldfish recent`

List the newest memories.

```bash
goldfish recent
```

### Options

| Option | Description |
|---|---|
| `-n N` | Number of memories to display |
| `--all` | Include archived memories |
| `-h`, `--help` | Show command help |

### Example

Display the latest 20 memories:

```bash
goldfish recent -n 20
```

## `goldfish pin`

Protect a memory from automatic compaction.

```bash
goldfish pin MEMORY_ID
```

### Options

| Option | Description |
|---|---|
| `--unpin` | Remove protection from a memory |
| `-h`, `--help` | Show command help |

## `goldfish forget`

Permanently delete a memory.

```bash
goldfish forget MEMORY_ID
```

### Options

| Option | Description |
|---|---|
| `-h`, `--help` | Show command help |

## `goldfish stats`

Display memory store statistics.

```bash
goldfish stats
```

## `goldfish demo`

Run a demonstration of Goldfish Memory features using a temporary store.

```bash
goldfish demo
```
