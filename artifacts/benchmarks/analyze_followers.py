import json
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

with open('artifacts/benchmarks/follower_repos_batch1.json', encoding='utf-8') as f:
    b1 = json.load(f)
with open('artifacts/benchmarks/follower_repos_batch2.json', encoding='utf-8') as f:
    b2 = json.load(f)

all_repos = [r for r in b1 + b2 if 'name' in r]
print(f"Total valid repos: {len(all_repos)}")

# Categorize by license
mit_repos = [r for r in all_repos if r.get('license') == 'MIT']
audit_repos = [r for r in all_repos if any(k in r.get('license', '').upper() for k in ['APACHE', 'GPL', 'LGPL', 'AGPL'])]
other_repos = [r for r in all_repos if r not in mit_repos and r not in audit_repos]

print(f"MIT (Automatic integration candidates): {len(mit_repos)}")
print(f"Apache / GPL (Requires Audit): {len(audit_repos)}")
print(f"Other / No License: {len(other_repos)}")

# Sort by stars
mit_sorted = sorted(mit_repos, key=lambda x: x.get('stars', 0), reverse=True)
audit_sorted = sorted(audit_repos, key=lambda x: x.get('stars', 0), reverse=True)

with open('artifacts/benchmarks/follower_repos_analysis.json', 'w', encoding='utf-8') as f:
    json.dump({
        'summary': {'total': len(all_repos), 'mit': len(mit_repos), 'audit': len(audit_repos), 'other': len(other_repos)},
        'mit_top': mit_sorted[:25],
        'audit_top': audit_sorted[:25]
    }, f, indent=2)

print("\n--- TOP MIT REPOSITORIES (AUTOMATIC ELIGIBILITY) ---")
for r in mit_sorted[:12]:
    desc = r['desc'][:90] if r['desc'] else 'No description'
    print(f"[{r['license']}] {r['full_name']} ({r['stars']}* | {r['lang']}): {desc}")

print("\n--- TOP APACHE / GPL REPOSITORIES (AUDIT REQUIRED) ---")
for r in audit_sorted[:12]:
    desc = r['desc'][:90] if r['desc'] else 'No description'
    print(f"[{r['license']}] {r['full_name']} ({r['stars']}* | {r['lang']}): {desc}")
