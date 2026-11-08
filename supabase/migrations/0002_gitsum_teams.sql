-- Gitsum Team: paid per seat. Individuals are free everywhere, so only teams have billing state.
-- Written only by /api/team/* with the service role (Stripe webhook, checkout, key management).
-- Members can read their own team and its key metadata; nobody can write from the browser.
-- Numbered 0002 so it sits beside 0001_gitsum_subscriptions.sql when both apps share one Supabase project.

create table if not exists public.gitsum_teams (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(name) between 1 and 80),
  owner_user_id uuid not null references auth.users (id) on delete cascade,
  stripe_customer_id text,
  stripe_subscription_id text unique,
  status text not null default 'incomplete',
  seats integer not null default 0 check (seats >= 0 and seats <= 500),
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists gitsum_teams_owner_idx on public.gitsum_teams (owner_user_id);
create index if not exists gitsum_teams_customer_idx on public.gitsum_teams (stripe_customer_id);

create table if not exists public.gitsum_team_members (
  team_id uuid not null references public.gitsum_teams (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  role text not null default 'member' check (role in ('owner', 'admin', 'member')),
  created_at timestamptz not null default now(),
  primary key (team_id, user_id)
);

create index if not exists gitsum_team_members_user_idx on public.gitsum_team_members (user_id);

-- Team keys (gsk_…). Only the SHA-256 hex digest is stored; the plaintext is shown once at creation.
create table if not exists public.gitsum_team_keys (
  id uuid primary key default gen_random_uuid(),
  team_id uuid not null references public.gitsum_teams (id) on delete cascade,
  key_hash text not null unique check (key_hash ~ '^[0-9a-f]{64}$'),
  last4 text not null check (char_length(last4) = 4),
  label text not null default 'Team key' check (char_length(label) between 1 and 60),
  created_by uuid references auth.users (id) on delete set null,
  created_at timestamptz not null default now(),
  last_used_at timestamptz,
  revoked_at timestamptz
);

create index if not exists gitsum_team_keys_team_idx on public.gitsum_team_keys (team_id) where revoked_at is null;

-- ---------- row level security ----------
alter table public.gitsum_teams enable row level security;
alter table public.gitsum_team_members enable row level security;
alter table public.gitsum_team_keys enable row level security;

-- Membership check that doesn't recurse through the members policy.
create or replace function public.gitsum_is_team_member(t uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (select 1 from public.gitsum_team_members m where m.team_id = t and m.user_id = auth.uid());
$$;

revoke all on function public.gitsum_is_team_member(uuid) from public;
grant execute on function public.gitsum_is_team_member(uuid) to authenticated;

drop policy if exists "members read their gitsum team" on public.gitsum_teams;
create policy "members read their gitsum team"
  on public.gitsum_teams for select
  to authenticated
  using (public.gitsum_is_team_member(id));

drop policy if exists "members read their gitsum team roster" on public.gitsum_team_members;
create policy "members read their gitsum team roster"
  on public.gitsum_team_members for select
  to authenticated
  using (public.gitsum_is_team_member(team_id));

-- key_hash is a one-way digest of a high-entropy secret, but there's no reason for the browser to see it:
-- members may read key metadata only through this column-limited grant.
revoke all on public.gitsum_team_keys from anon, authenticated;
grant select (id, team_id, last4, label, created_by, created_at, last_used_at, revoked_at) on public.gitsum_team_keys to authenticated;

drop policy if exists "members read their gitsum team keys" on public.gitsum_team_keys;
create policy "members read their gitsum team keys"
  on public.gitsum_team_keys for select
  to authenticated
  using (public.gitsum_is_team_member(team_id));

-- No insert/update/delete policies: with RLS on, only the service role can write.
