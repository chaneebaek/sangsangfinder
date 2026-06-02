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
