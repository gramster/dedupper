"""dedupper - Find and close duplicate issues in a Github repo. """

__version__ = '0.01'

import os
import click
from .dedupper import dedup


@click.command()
@click.option('-t', '--token', help='GitHub PAT with issue read/write rights for the repo, if not from environment.', default='--')
@click.option('-d', '--database', help='Database file for persistent state.', default='')
@click.option('-c', '--close', type=bool, help='Close any duplicates found.', default=False)
@click.option('-p', '--pretend', type=bool, help='Do a dry run without making any changes.', default=False)
@click.argument('repo', required=1)
def dedupper(token, database, close, pretend, repo):
    """ REPO: Repository name in form org/repo. """    
    if token == '--':
        token = os.environ.get('GITHUB_TOKEN') or ''
    if not token:
        raise Exception("No Github token")
    if not database:
        database = repo.replace('/', '-') + '.db'
    dedup(token=token, database=database, repo=repo, close=close, pretend=pretend)


def main():
    dedupper()

