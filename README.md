# dedupper - Find (and optionally close) duplicate issues in a Github repo.

See CONTRIBUTING.md for build instructions, or install from PyPI with:

```
python -m pip install dedupper
```

Use `dedupper -h` for help.

State is stored in a database and subsequent runs will be incremental.

You need to set an OPENAI_API_KEY environment variable as gpt-4o-turbo is
used to make a final decison about whether issues are dups.

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

