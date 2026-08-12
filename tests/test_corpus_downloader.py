from __future__ import annotations

from scripts.download_open_access_corpus import extract_article, is_allowed_license


def test_license_filter_only_accepts_reusable_licenses() -> None:
    assert is_allowed_license("Creative Commons Attribution 4.0")
    assert is_allowed_license("CC BY-SA 4.0")
    assert is_allowed_license(
        "",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    )
    assert not is_allowed_license("CC BY-NC 4.0")
    assert not is_allowed_license("CC BY-ND 4.0")
    assert not is_allowed_license("all rights reserved")


def test_extract_article_reads_jats_text_and_license() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front>
        <article-meta>
          <title-group><article-title>Rice blast field study</article-title></title-group>
          <abstract><p>This abstract contains enough words to describe a controlled rice blast study and its field observations in detail for extraction.</p></abstract>
          <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</license></permissions>
        </article-meta>
      </front>
      <body><sec><title>Results</title><p>This paragraph contains sufficient detail about symptoms, environmental conditions, diagnosis, and management practices for the article extractor test.</p></sec></body>
    </article>"""
    content, metadata = extract_article(xml)
    assert "Rice blast field study" in content
    assert "## Results" in content
    assert metadata["license"] == "CC BY 4.0"
    assert metadata["license_url"].endswith("/by/4.0/")
