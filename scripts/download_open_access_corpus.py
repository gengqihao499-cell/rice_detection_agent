from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rice_agent.config import PROJECT_ROOT, settings


API_ROOT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_QUERY = (
    '(TITLE_ABS:rice OR TITLE_ABS:"Oryza sativa") '
    "AND (TITLE_ABS:disease OR TITLE_ABS:pathogen OR TITLE_ABS:blast "
    "OR TITLE_ABS:blight OR TITLE_ABS:pest OR TITLE_ABS:insect "
    "OR TITLE_ABS:fungus OR TITLE_ABS:fungal OR TITLE_ABS:bacterium "
    "OR TITLE_ABS:bacterial OR TITLE_ABS:virus OR TITLE_ABS:stress "
    "OR TITLE_ABS:management) "
    "AND OPEN_ACCESS:Y AND IN_EPMC:Y"
)
USER_AGENT = "RiceCare-RAG-Corpus-Builder/1.0"
ALLOWED_LICENSES = {
    "cc0",
    "cc by",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}


def _clean(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalized_license(value: str) -> str:
    normalized = value.casefold().replace("_", " ").replace("-", " ")
    normalized = normalized.replace("creative commons", "cc")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace("attribution share alike", "by sa")
    normalized = normalized.replace("attribution", "by")
    return normalized


def is_allowed_license(value: str, url: str = "") -> bool:
    combined = _normalized_license(f"{value} {url}")
    if any(token in combined for token in ("noncommercial", "by nc", "no derivatives", "by nd")):
        return False
    if "creativecommons.org/publicdomain/zero" in url.casefold():
        return True
    if "creativecommons.org/licenses/by/" in url.casefold():
        return True
    if "creativecommons.org/licenses/by-sa/" in url.casefold():
        return True
    return any(allowed in combined for allowed in ALLOWED_LICENSES)


def _license_from_xml(root: ET.Element) -> tuple[str, str]:
    values: list[str] = []
    urls: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "license":
            continue
        text = _clean(" ".join(element.itertext()))
        if text:
            values.append(text)
        for key, value in element.attrib.items():
            if key.rsplit("}", 1)[-1] == "href" and value:
                urls.append(value)
    return " | ".join(values), urls[0] if urls else ""


def extract_article(xml_bytes: bytes) -> tuple[str, dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    title = ""
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "article-title":
            title = _clean(" ".join(element.itertext()))
            if title:
                break

    sections: list[str] = []
    if title:
        sections.append(f"# {title}")
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "abstract":
            text = _clean(" ".join(element.itertext()))
            if len(text) >= 80:
                sections.extend(["## Abstract", text])
            break

    body = next(
        (
            element
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "body"
        ),
        None,
    )
    if body is not None:
        for element in body.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "title":
                text = _clean(" ".join(element.itertext()))
                if text and len(text) <= 240:
                    sections.append(f"## {text}")
            elif tag == "p":
                text = _clean(" ".join(element.itertext()))
                if len(text) >= 60:
                    sections.append(text)

    license_text, license_url = _license_from_xml(root)
    return "\n\n".join(sections).strip(), {
        "title": title,
        "license": license_text,
        "license_url": license_url,
    }


def _request(url: str, email: str, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if email:
        headers["From"] = email
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 >= retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _search_pages(query: str, email: str) -> Iterator[list[dict[str, Any]]]:
    cursor = "*"
    seen_cursors: set[str] = set()
    while cursor not in seen_cursors:
        seen_cursors.add(cursor)
        params = urllib.parse.urlencode(
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": 100,
                "cursorMark": cursor,
            }
        )
        payload = json.loads(_request(f"{API_ROOT}/search?{params}", email))
        results = payload.get("resultList", {}).get("result", [])
        if not results:
            break
        yield [item for item in results if isinstance(item, dict)]
        next_cursor = str(payload.get("nextCursorMark") or "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor


def _existing_manifest(path: Path) -> tuple[set[str], int]:
    identifiers: set[str] = set()
    total_bytes = 0
    if not path.is_file():
        return identifiers, total_bytes
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        identifiers.add(str(item.get("source_id", "")))
        total_bytes += int(item.get("bytes", 0) or 0)
    return identifiers, total_bytes


def _fetch_article(
    record: dict[str, Any],
    email: str,
) -> tuple[str, bytes | None, dict[str, Any] | None]:
    pmcid = str(record.get("pmcid") or "").strip().upper()
    if not pmcid:
        return "missing_id", None, None
    xml_url = f"{API_ROOT}/{urllib.parse.quote(pmcid)}/fullTextXML"
    try:
        xml_bytes = _request(xml_url, email)
        content, xml_metadata = extract_article(xml_bytes)
    except (ET.ParseError, urllib.error.URLError, TimeoutError):
        return "failed", None, None

    record_license = str(record.get("license") or "")
    license_text = xml_metadata["license"] or record_license
    license_url = xml_metadata["license_url"]
    if not is_allowed_license(license_text, license_url):
        return "license", None, None
    encoded = content.encode("utf-8")
    if len(encoded) < 2_000:
        return "short", None, None

    metadata: dict[str, Any] = {
        "file": f"{pmcid}.txt",
        "source_id": pmcid,
        "title": xml_metadata["title"] or str(record.get("title") or ""),
        "source_url": f"https://europepmc.org/articles/{pmcid}",
        "api_url": xml_url,
        "doi": str(record.get("doi") or ""),
        "authors": str(record.get("authorString") or ""),
        "journal": str(record.get("journalTitle") or ""),
        "publication_date": str(record.get("firstPublicationDate") or ""),
        "license": license_text,
        "license_url": license_url,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Europe PMC / PMC Open Access Subset",
    }
    return "ok", encoded, metadata


def download_corpus(
    *,
    output_dir: Path,
    target_mb: float,
    query: str,
    email: str,
    max_articles: int,
    delay: float,
    workers: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    seen, total_bytes = _existing_manifest(manifest_path)
    target_bytes = int(target_mb * 1024 * 1024)
    added = 0
    skipped_license = 0
    skipped_short = 0

    if total_bytes >= target_bytes:
        return {
            "status": "already_complete",
            "target_bytes": target_bytes,
            "corpus_bytes": total_bytes,
            "article_count": len(seen),
            "manifest": str(manifest_path),
        }

    worker_count = max(1, min(8, workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for page in _search_pages(query, email):
            candidates = [
                record
                for record in page
                if str(record.get("pmcid") or "").strip().upper() not in seen
            ]
            for start in range(0, len(candidates), worker_count):
                if total_bytes >= target_bytes or len(seen) >= max_articles:
                    break
                batch = candidates[start : start + worker_count]
                for status, encoded, metadata in executor.map(
                    lambda record: _fetch_article(record, email),
                    batch,
                ):
                    if status == "license":
                        skipped_license += 1
                        continue
                    if status == "short":
                        skipped_short += 1
                        continue
                    if status != "ok" or encoded is None or metadata is None:
                        continue
                    pmcid = str(metadata["source_id"])
                    if pmcid in seen:
                        continue
                    (output_dir / str(metadata["file"])).write_bytes(encoded)
                    with manifest_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    seen.add(pmcid)
                    total_bytes += len(encoded)
                    added += 1
                    print(
                        f"[{len(seen)}] {pmcid} +{len(encoded) / 1024:.1f}KB "
                        f"total={total_bytes / 1024 / 1024:.2f}MB",
                        flush=True,
                    )
                if delay > 0:
                    time.sleep(delay)
            if total_bytes >= target_bytes or len(seen) >= max_articles:
                break

    return {
        "status": "complete" if total_bytes >= target_bytes else "partial",
        "target_bytes": target_bytes,
        "corpus_bytes": total_bytes,
        "article_count": len(seen),
        "added_articles": added,
        "skipped_license": skipped_license,
        "skipped_short": skipped_short,
        "manifest": str(manifest_path),
        "query": query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Europe PMC Open Access Subset 下载可追踪许可的水稻语料。"
    )
    parser.add_argument("--target-mb", type=float, default=settings.knowledge_target_mb)
    parser.add_argument("--output-dir", type=Path, default=settings.knowledge_corpus_dir)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--email", default=os.getenv("EUROPE_PMC_EMAIL", ""))
    args = parser.parse_args()
    result = download_corpus(
        output_dir=args.output_dir.resolve(),
        target_mb=args.target_mb,
        query=args.query,
        email=args.email,
        max_articles=args.max_articles,
        delay=max(0.0, args.delay),
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
