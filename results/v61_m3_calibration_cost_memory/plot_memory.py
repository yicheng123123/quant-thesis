from pathlib import Path
import csv, json, math

ROOT=Path(__file__).resolve().parent

def main():
    accounts=[json.loads(p.read_text()) for p in ROOT.glob('PROCESS_ACCOUNTING_*.json')]
    a=next(x for x in accounts if x['status']=='COMPLETE')
    runtime=json.loads((ROOT/'ACTUAL_RUNTIME.json').read_text()); pid=runtime['pid']; samples={}
    with (ROOT/'memory'/('gpu_'+a['attempt']+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            if not r['device_used_bytes']: continue
            sec=int(float(r['elapsed_seconds'])); z=samples.setdefault(sec,{'gpu':0.,'rss':None})
            z['gpu']=max(z['gpu'],int(r['device_used_bytes'])/2**30)
    with (ROOT/'memory'/('rss_'+a['attempt']+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            if int(r['pid'])!=pid: continue
            sec=int(float(r['elapsed_seconds'])); z=samples.setdefault(sec,{'gpu':None,'rss':0.})
            z['rss']=max(z['rss'] or 0.,int(r['rss_bytes'])/2**30)
    end=max(samples); ymax=math.ceil(max(v for z in samples.values() for v in z.values() if v is not None))
    left=85; top=65; width=900; height=360
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="535" viewBox="0 0 1050 535">',
        '<rect width="1050" height="535" fill="white"/>',
        '<style>text{font:14px sans-serif;fill:#25313d}</style>',
        '<text x="85" y="30" font-size="20">v61 complete M3/LWC calibration: sampled memory</text>']
    for y in range(ymax+1):
        yy=top+height-height*y/ymax
        parts += [f'<line x1="{left}" y1="{yy}" x2="{left+width}" y2="{yy}" stroke="#dddddd"/>',f'<text x="45" y="{yy+5}">{y}</text>']
    for minute in range(0,math.ceil(end/60)+1,5):
        xx=left+width*minute*60/end
        if xx>left+width: break
        parts += [f'<text x="{xx-5}" y="450">{minute}</text>']
    parts += ['<text x="420" y="477">Elapsed time from pre-run baseline (minutes)</text>',
        '<text x="22" y="60">GiB</text>']
    for key,color in [('gpu','#176a9c'),('rss','#ad631a')]:
        pts=' '.join(f'{left+width*t/end:.2f},{top+height-height*z[key]/ymax:.2f}' for t,z in sorted(samples.items()) if z[key] is not None)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    parts += ['<text x="85" y="501" fill="#176a9c">Blue: GPU device total (desktop/others included)</text>',
        '<text x="580" y="501" fill="#ad631a">Brown: workload main process RSS</text>',
        '<text x="85" y="523">Source: raw ~200ms samples; plotted as maximum per 1-second bin. Peaks are sampled, not instantaneous.</text>','</svg>']
    (ROOT/'MEMORY_TIMELINE.svg').write_text('\n'.join(parts),encoding='utf-8')
    print(ROOT/'MEMORY_TIMELINE.svg')

if __name__=='__main__': main()
