# Contributing to goldfish-memory

Thanks for helping out. The rules below keep the project's promises intact.

## Setup

```bash
git clone https://github.com/DhanushNehru/goldfish-memory.git
cd goldfish-memory
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## Checks (all run in CI)

```bash
pytest -q          # tests; sockets are disabled for the whole suite
ruff check .       # lint
ruff format .      # formatting
mypy               # type checks (strict)
```

## The two hard rules

1. **The engine never touches the network.** No downloads, no telemetry, no
   calls home - not even optional ones. CI runs the full suite inside an
   empty network namespace; a change that needs the network will fail there.
   Model weights are bundled or user-provisioned, never fetched at runtime.
2. **Every feature gets a unit test.** If you add behavior, add the test in
   the same PR.

## Good first contributions

Pick an open issue, comment that you're taking it, and open a PR against
`main`. Keep PRs focused; small and reviewed beats big and stuck.

Please be kind and constructive in issues and reviews - assume good intent,
critique code not people.
