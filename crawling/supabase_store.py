"""
Supabase/Postgres storage for crawled Hansung notices.

Set SUPABASE_DB_URL to a Supabase Postgres connection string, for example
the Transaction pooler URI with sslmode=require.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

SOURCE = "hansung"
DB_URL_ENV = "SUPABASE_DB_URL"


CREATE_NOTICES_SQL = """
create table if not exists notices (
    id bigserial primary key,
    source text not null default 'hansung',
    notice_id text,
    title text not null,
    url text not null unique,
    posted_at date,
    posted_date_text text,
    category text,
    category_type text[] not null default '{}',
    job_types text[] not null default '{}',
    body text,
    views integer,
    notice_score double precision,
    embedding jsonb,
    raw jsonb not null default '{}'::jsonb,
    crawled_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_notices_source_posted_at
    on notices (source, posted_at desc);

create index if not exists idx_notices_category
    on notices (category);

alter table notices add column if not exists category_type text[] not null default '{}';
alter table notices add column if not exists job_types text[] not null default '{}';
alter table notices add column if not exists notice_score double precision;
alter table notices add column if not exists embedding jsonb;

create table if not exists users (
    user_id text primary key,
    name text not null default '',
    phone text not null,
    interests text[] not null default '{}',
    track text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (phone)
);

create index if not exists idx_users_interests
    on users using gin (interests);

create table if not exists notification_deliveries (
    id bigserial primary key,
    notice_db_id bigint references notices(id) on delete cascade,
    notice_id text,
    user_id text not null references users(user_id) on delete cascade,
    channel text not null default 'imessage',
    category text not null default '',
    status text not null default 'pending',
    error text,
    sent_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (notice_db_id, user_id, channel, category)
);

alter table notification_deliveries add column if not exists category text not null default '';
alter table notification_deliveries
    drop constraint if exists notification_deliveries_notice_db_id_user_id_channel_key;
create unique index if not exists idx_notification_deliveries_notice_user_channel_category
    on notification_deliveries (notice_db_id, user_id, channel, category);

create index if not exists idx_notification_deliveries_status
    on notification_deliveries (status, created_at desc);
"""


UPSERT_NOTICE_SQL = """
insert into notices (
    source,
    notice_id,
    title,
    url,
    posted_at,
    posted_date_text,
    category,
    body,
    views,
    raw,
    crawled_at,
    updated_at
)
values (
    %(source)s,
    %(notice_id)s,
    %(title)s,
    %(url)s,
    %(posted_at)s,
    %(posted_date_text)s,
    %(category)s,
    %(body)s,
    %(views)s,
    %(raw)s,
    now(),
    now()
)
on conflict (url) do update set
    notice_id = excluded.notice_id,
    title = excluded.title,
    posted_at = excluded.posted_at,
    posted_date_text = excluded.posted_date_text,
    category = excluded.category,
    body = excluded.body,
    views = excluded.views,
    raw = excluded.raw,
    updated_at = now();
"""


def _db_url() -> str:
    value = os.getenv(DB_URL_ENV) or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            f"{DB_URL_ENV} is required. Add it to GitHub Actions secrets or your local .env."
        )
    return value


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        _db_url(),
        autocommit=False,
        connect_timeout=10,
        prepare_threshold=None,
    )


def ensure_schema() -> None:
    """Create the notices table and indexes if they do not exist."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_NOTICES_SQL)
        conn.commit()


def load_seen_urls(year: str | int) -> set[str]:
    """Load URLs already stored for the target year."""
    year_text = str(year)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select url
                from notices
                where source = %s
                  and (
                    extract(year from posted_at)::text = %s
                    or posted_date_text like %s
                  )
                """,
                (SOURCE, year_text, f"{year_text}%"),
            )
            return {row[0] for row in cur.fetchall()}


def upsert_notices(items: list[dict[str, Any]]) -> int:
    """Insert or update notices by unique URL."""
    if not items:
        return 0

    rows = [_to_row(item) for item in items]
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_NOTICE_SQL, rows)
        conn.commit()
    return len(rows)


def _to_row(item: dict[str, Any]) -> dict[str, Any]:
    clean_item = _clean_for_postgres(item)
    return {
        "source": SOURCE,
        "notice_id": _extract_notice_id(clean_item.get("url", "")),
        "title": clean_item.get("title") or "",
        "url": clean_item.get("url") or "",
        "posted_at": _parse_date(clean_item.get("date")),
        "posted_date_text": clean_item.get("date"),
        "category": clean_item.get("category"),
        "body": clean_item.get("body"),
        "views": _parse_int(clean_item.get("views")),
        "raw": Jsonb(clean_item),
    }


def _clean_for_postgres(value: Any) -> Any:
    """Postgres text/jsonb values cannot contain literal NUL bytes."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean_for_postgres(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_for_postgres(item) for key, item in value.items()}
    return value


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def _extract_notice_id(url: str) -> str | None:
    match = re.search(r"/(\d+)/artclView\.do", url)
    return match.group(1) if match else None
