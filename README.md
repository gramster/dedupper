# dedupper - Find (and optionally close) duplicate issues in a Github repo.

See CONTRIBUTING.md for build instructions, or install from PyPI with:

```
python -m pip install dedupper
```

Use `dedupper -h` for help.

You need to be running ollama locally as dedupper uses an LLM to make the final
decision about whether an issue is a duplicate.

State is stored in a database and subsequent runs will be incremental.

## Development

This project uses `flit`. First install `flit`:

```
python -m pip install flit
```

Then to build:

```
flit build
```

To install locally:

```
flit install
```

