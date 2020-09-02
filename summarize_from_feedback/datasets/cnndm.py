"""
all data originally from https://cs.nyu.edu/~kcho/DMQA/ and https://github.com/abisee/cnn-dailymail
"""
import hashlib
import json
import os
import re

import ftfy

from summarize_from_feedback.utils import blobs

dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]  # acceptable ways to end a sentence


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + "."


def clean_up_highlights(line):
    if line.startswith("NEW: "):
        return line.split("NEW: ")[1].strip()
    bad_prefixes = ["READ: ", "for all the latest "]
    for prefix in bad_prefixes:
        if line.startswith(prefix):
            return ""
    # these are bad even if they appear mid-highlight
    bad_after = [
        "CLICK HERE ",
        "Click HERE ",
        "Click here ",
        "For confidential support call the Samaritans in the UK",
    ]
    for bad in bad_after:
        if bad in line:
            line = line.split(bad)[0].strip()
    return line


def get_article_and_highlights(story_file, refs_with_bullets=False, clean_highlights=True):
    with open(story_file) as f:
        original_text = f.read()
    original_text = ftfy.fix_text(original_text)
    original_text = original_text.replace("\xa0", " ")  # hmm, not handled by ftfy?

    article_lines = []
    highlights = []
    has_seen_highlight = False

    for line in original_text.split("\n"):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            assert line == "@highlight"
            has_seen_highlight = True
        elif has_seen_highlight:
            if clean_highlights:
                line = clean_up_highlights(line)
            if not refs_with_bullets:
                line = fix_missing_period(line)
            if line:
                highlights.append(line)
        else:
            article_lines.append(line)

    article = "\n\n".join(article_lines)
    article = clean_up_start(article)

    if refs_with_bullets:
        highlights = "- " + "\n- ".join(highlights)
    else:
        highlights = " ".join(highlights)

    return article, highlights


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_info(url):
    if "dailymail.co.uk" in url or "mailonsunday.ie" in url or "lib.store.yahoo.net" in url:
        site = "dailymail"
    else:
        assert "cnn.com" in url or "cnn.hk" in url, url
        site = "cnn"
    url_hash = hashhex(url.encode("utf-8"))
    return url_hash, site


def clean_up_start(text):
    text = re.split(r"\(CNN\) +--", text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1] + text[100:]
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return text.strip()


def _cnndm_iter(split, subset="all", refs_with_bullets=False, clean_highlights=True):
    if split == "valid":
        split = "val"
    with blobs.open_file_cached(
        f"https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/cnndm/url_lists/{subset}_{split}.txt"
    ) as f:
        urls = [line.strip() for line in f]

    with blobs.open_file_cached("https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/cnndm/titles.json") as f:
        titles = json.load(f)
    urls_dir = blobs.download_directory_cached(
        f"https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/cnndm/cache_{split}"
    )

    for url in urls:
        article_id, site = get_url_info(url)
        path = os.path.join(urls_dir, site, "stories", f"{article_id}.story")
        article, ref_sents = get_article_and_highlights(
            path, refs_with_bullets=refs_with_bullets, clean_highlights=clean_highlights
        )
        yield dict(
            article=article, reference=ref_sents, id=article_id, site=site, title=titles[article_id]
        )


def cnndm_generator(split):
    yield from _cnndm_iter(split, subset="all")


def cnndm_filtered_generator(split):
    yield from _cnndm_iter(split, subset="filtered48to64", refs_with_bullets=True)


def cnndm_filtered_generator_short(split):
    yield from _cnndm_iter(split, subset="filtered24to48")
