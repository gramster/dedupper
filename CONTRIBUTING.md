## Building

1) Install flit of needed: python -m pip install flit
2) Run 'flit install'

You should then have a local copy available as 'arxivsummary'.

## Releasing

To make a release,

  1) Update README.md and the \__version__ in arxivsummary/\__init__.py
  2) Run 'flit install'
  3) Test the installed arxivsummary locally
  4) Upload to PyPI: 'flit publish'

