#!/usr/bin/env python3
"""
This script does the following:
  • Given a GitHub repo (e.g. "username/repo"), it fetches all issues (excluding PRs)
    and stores the issue number, title, body, state and created_at into an SQLite database.
  • It then checks every open issue versus all older issues (open or closed) and computes 
    a TF-IDF cosine similarity between their (title+body) texts. If similarity exceeds a chosen
    threshold, it is checked again by an LLM, and flagged as potentially a duplicate.
  • For any open issue that appears to be a duplicate of one or more earlier issues, the script
    posts a comment on GitHub listing the duplicate issue numbers and optionally closes the issue.
  • On subsequent runs, only new issues (not already in the database) are fetched, and duplicate
    detection is performed for the open issues that have not yet been commented on.
    
The code also uses a session with a retry adapter to try to gracefully handle rate limiting
or any transient failures.
"""

import os
import sqlite3
import time
import logging
import openai
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# GitHub API base URL
GITHUB_API = "https://api.github.com"

# A threshold for duplicate similarity. If we pass this, we still do a second
# check with an LLM.
DUPLICATE_THRESHOLD = 0.8

# Maximum number of issues per page from GitHub
PER_PAGE = 100

# Delay (in seconds) when rate limit encountered (or when HTTP 403 is received)
RATE_LIMIT_SLEEP = 60

llm_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GitHubClient:
    def __init__(self, token):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Duplicate-Issue-Detector",
            }
        )
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def request(self, method, url, **kwargs):
        while True:
            response = self.session.request(method, url, **kwargs)
            if response.status_code == 403:
                # Check for rate limiting
                if (
                    "X-RateLimit-Remaining" in response.headers
                    and response.headers["X-RateLimit-Remaining"] == "0"
                ):
                    reset_time = int(
                        response.headers.get(
                            "X-RateLimit-Reset", time.time() + RATE_LIMIT_SLEEP
                        )
                    )
                    sleep_for = max(reset_time - int(time.time()), RATE_LIMIT_SLEEP)
                    logging.warning(
                        "Rate limit reached. Sleeping for %d seconds...", sleep_for
                    )
                    time.sleep(sleep_for)
                    continue
                else:
                    logging.warning(
                        "403 received. Retrying in %d seconds...", RATE_LIMIT_SLEEP
                    )
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
            if not response.ok:
                logging.error("Error %s: %s", response.status_code, response.text)
            return response

    def get_issues(self, repo, state="all", since=None):
        """
        Generator that pages through issues from the given repo.
        Excludes pull requests.
        """
        page = 1
        params = {"state": state, "per_page": PER_PAGE, "page": page}
        if since:
            params["since"] = since
        while True:
            url = f"{GITHUB_API}/repos/{repo}/issues?state=all"
            response = self.request("GET", url, params=params)
            issues = response.json()
            if not isinstance(issues, list):
                logging.error("Error: expected list of issues but got: %s", issues)
                break
            # Exclude pull requests – these issues have a "pull_request" key.
            issues = [issue for issue in issues if "pull_request" not in issue]
            if not issues:
                break
            for issue in issues:
                yield issue
            if len(issues) == 0:  # < PER_PAGE:
                break
            page += 1
            params["page"] = page

    def add_comment(self, repo, issue_number, comment_text):
        url = f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments"
        payload = {"body": comment_text}
        response = self.request("POST", url, json=payload)
        if response.ok:
            logging.info("Added comment on issue #%d: %s", issue_number, comment_text)
        else:
            logging.error(
                "Failed to add comment on issue #%d: %s", issue_number, response.text
            )

    def close_issue(self, repo, issue_number):
        url = f"{GITHUB_API}/repos/{repo}/issues/{issue_number}"
        payload = {"state": "closed"}
        response = self.request("PATCH", url, json=payload)
        if response.ok:
            logging.info("Closed issue #%d: %s", issue_number)
        else:
            logging.error(
                "Failed to close issue #%d: %s", issue_number, response.text
            )      


def init_db(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS issues (
        id INTEGER PRIMARY KEY,        -- GitHub issue ID (unique)
        number INTEGER UNIQUE,          -- GitHub issue number
        title TEXT,
        body TEXT,
        state TEXT,                     -- open or closed
        created_at TEXT,
        duplicate_commented INTEGER DEFAULT 0
    )
    """
    )
    conn.commit()
    return conn


def issue_exists(conn, issue_number):
    c = conn.cursor()
    c.execute("SELECT 1 FROM issues WHERE number=?", (issue_number,))
    return c.fetchone() is not None


def insert_issue(conn, issue):
    c = conn.cursor()
    c.execute(
        """
        INSERT OR IGNORE INTO issues 
        (id, number, title, body, state, created_at, duplicate_commented) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            issue["id"],
            issue["number"],
            issue.get("title", ""),
            issue.get("body", ""),
            issue.get("state", ""),
            issue.get("created_at", ""),
            0,
        ),
    )
    conn.commit()


def combine_text(issue):
    # Combine the title and body (if present) into one document
    parts = [issue.get("title", "")]
    if issue.get("body"):
        parts.append(issue["body"])
    return "\n".join(parts)


def is_duplicate_candidate(doc_current, doc_prev):
    # Build a mini vectorizer for the two texts
    vectorizer = TfidfVectorizer().fit([doc_current, doc_prev])
    tfidf = vectorizer.transform([doc_current, doc_prev])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return sim >= DUPLICATE_THRESHOLD


def is_duplicate(doc_current, doc_prev):
    # Use an LLM to make the decision
    assert(llm_client is not None)
    retry = 0
    results = []
    while retry < 3:
        try:
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an experienced software engineer triaging issues in GitHub."},
                    {"role": "user", "content": f"""
Analyze the following two issues to determine if they are duplicates. Answer only Yes or No:
                     
==============
                     
ISSUE 1
                     
{doc_current}

==============
                     
ISSUE 2
         
{doc_prev}
"""}
                ]
            )
            result = response.choices[0].message.content
            results.append(result)
            if result:
                result = result.strip().lower()
                if result in ["yes", "no", "yes.", "no."]:
                    return result == "yes" or result == "yes."
            retry += 1
        except Exception as e:
            print(f"Error checking issue: {e}")
            retry += 1
    return False


def detect_duplicates(issues_db):
    """
    For every open issue that has not been commented about as a duplicate,
    check all older issues (created earlier) for a potential duplicate.
    Returns a list of tuples: (open_issue_number, list_of_duplicate_issue_numbers)
    and a dictionary of titles indexed by number.
    """
    titles = {}
    c = issues_db.cursor()
    # Get all issues from the DB with their key fields
    c.execute(
        "SELECT number, title, body, state, created_at, duplicate_commented FROM issues ORDER BY created_at ASC"
    )
    rows = c.fetchall()

    # Build a list of dictionaries for easier processing
    issues = []
    for number, title, body, state, created_at, duplicate_commented in rows:
        issues.append(
            {
                "number": number,
                "title": title,
                "body": body,
                "state": state,
                "created_at": created_at,
                "duplicate_commented": duplicate_commented,
            }
        )

    results = []
    # For each open issue that has not been commented about:
    for idx, issue in enumerate(issues):
        if issue["state"] != "open" or issue["duplicate_commented"]:
            continue
        # Make a document for the current issue
        doc_current = combine_text(issue)
        duplicates = []
        # Compare with all previous issues (by creation order)
        for prev in issues[:idx]:
            doc_prev = combine_text(prev)
            if is_duplicate_candidate(doc_current, doc_prev):
                if is_duplicate(doc_current, doc_prev):
                    titles[prev["number"]] = prev.get("title", "")
                    duplicates.append(prev["number"])
        if duplicates:
            titles[issue["number"]] = issue.get("title", "")            
            results.append((issue["number"], duplicates))
    return results, titles


def mark_issue_commented(conn, issue_number):
    c = conn.cursor()
    c.execute("UPDATE issues SET duplicate_commented=1 WHERE number=?", (issue_number,))
    conn.commit()


def dedup(token, database, repo, close=False, pretend=False):

    # Initialize database connection
    conn = init_db(database)

    # Initialize GitHub client
    gh = GitHubClient(token)

    # Optionally, you can keep track of last run time to narrow down the API calls.
    # For simplicity, here we fetch all issues for a new repo OR use "since" parameter if needed.
    # For incremental update, we can pick the latest created_at in our DB.
    c = conn.cursor()
    c.execute("SELECT MAX(created_at) FROM issues")
    row = c.fetchone()
    since = row[0] if row and row[0] else None
    if since:
        logging.info("Fetching issues updated since %s", since)
    else:
        logging.info("Fetching all issues.")

    # Fetch issues from GitHub and add new ones to the DB
    new_issue_count = 0
    for issue in gh.get_issues(repo, state="all", since=since):
        # Some issues may not be new – check by issue number
        if not issue_exists(conn, issue["number"]):
            logging.info(
                "Storing new issue #%d: %s",
                issue["number"],
                issue.get("title", "").strip(),
            )
            insert_issue(conn, issue)
            new_issue_count += 1

    logging.info("Fetched and stored %d new issues", new_issue_count)

    # Duplicate detection, for each open and not-yet-commented issue
    dup_results, titles = detect_duplicates(conn)
    if not dup_results:
        logging.info("No duplicate issues found.")
    else:
        for open_issue_num, dup_nums in dup_results:
            # Format the comment text listing the potential duplicate issue numbers.
            dup_list_text = ", ".join([f"#{num}" for num in dup_nums])
            comment = f"dedupper.py detects this as a duplicate of issue(s): {dup_list_text}"
            if close:
                comment += "\n\nPlease reopen if you disagree."
            for num in dup_nums:
                logging.info(
                    "Issue #%d: '%s' may be a dup of #%d: '%s'", 
                        open_issue_num, titles[open_issue_num], num, titles[num]
                )                           
            if not pretend:
                # Post the comment
                gh.add_comment(repo, open_issue_num, comment)
                # Close the issue by setting the state to closed
                if close:
                    gh.close_issue(repo, open_issue_num)
            
                # Mark the issue as having been commented (to avoid duplicating comments on subsequent runs)
                mark_issue_commented(conn, open_issue_num)

    conn.close()

