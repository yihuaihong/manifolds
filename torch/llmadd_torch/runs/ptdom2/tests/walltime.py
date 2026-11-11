import re, glob, statistics, datetime as dt, collections
res = collections.defaultdict(list)
for f in glob.glob('/scratch/yh6210/llmadd_torch/logs/llmadd_ptdom-[AB]*-s*_*.out'):
    t = open(f, errors='replace').read()
    m = re.match(r'.*/llmadd_ptdom-(A|B1|B2|B3)-.*_(\d+)\.out', f)
    if not m: continue
    a = re.findall(r'^## 1\. inputs (\S+)$', t, re.M); d = re.findall(r'^## done (\S+)$', t, re.M)
    if not a or not d: continue
    p = lambda s: dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
    sec = (p(d[-1]) - p(a[-1])).total_seconds()
    off = 'offline LM: not run' not in t
    res[(m.group(1), off)].append(sec / 3600)
for k, v in sorted(res.items()):
    print(k, len(v), 'median %.3f  p90 %.3f  max %.3f' % (statistics.median(v), sorted(v)[int(0.9 * len(v))], max(v)))
