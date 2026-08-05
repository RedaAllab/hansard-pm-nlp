"""Text and category cleaning for the raw Hansard corpus.

The raw `contribution_text` carries embedded HTML markup (<em>, <strong>,
<sub>, and Hansard's own <span class="column-number" ...> tags), \\r\\n line
breaks, and HTML entities - all artifacts of the source, not content. The raw
column is never overwritten: cleaning always produces a new column so nothing
is destroyed. `debate_section` also has leading/trailing whitespace variants
of the same debate (e.g. "Covid-19 Update" vs " Covid-19 Update") that would
otherwise silently split one category into two.
"""

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_contribution_text(text: str) -> str:
    """Strip embedded HTML tags/entities and normalize whitespace."""
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_debate_section(debate_section: str) -> str:
    """Strip leading/trailing whitespace so duplicate-looking categories merge."""
    return debate_section.strip()
