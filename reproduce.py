#!/usr/bin/env python3
"""Offline publication audit and plotting. Never invokes a model or database."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, shutil, statistics, sys, tempfile
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
sys.path.insert(0,str(ROOT/'static'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def check(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-7),(a,b)

def unpack(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for f in source.iterdir():
        if f.suffix=='.gz':(destination/f.stem).write_bytes(gzip.decompress(f.read_bytes()))
        elif f.is_file():shutil.copyfile(f,destination/f.name)

def verify_manifest():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Release file changed: '+name
    print(f"Verified {len(manifest['sha256'])} release files",flush=True)

def static_audit(output):
    import plot_static as plot
    from wireless_exp.metrics import audit_state
    from wireless_exp.tkg.graph_store import normalized
    with tempfile.TemporaryDirectory(prefix='wireless_static_audit_') as temp:
        base=Path(temp)
        for src in sorted((ROOT/'evidence/static').iterdir()):unpack(src,base/src.name)
        series,rows,audit=plot.load_and_validate(base)
        for (scene,method,repeat),steps in series.items():
            users=json.loads((base/f'{scene}__{method}__r{repeat:03d}'/'selected_users.json').read_text())
            truth={str(u['user_id']):u['ground_truth'] for u in users}
            for step in steps:
                numeric=audit_state(step['state'],truth,method!='rule_upstream')
                for k in ['admitted_users','valid_correct_users','resource_feasible_throughput_mbps','bandwidth_used_mhz']:
                    check(numeric[k],step[k])
        for scene in plot.SCENARIOS:
            for rep in plot.REPEATS:
                a=series[scene,'tkg_dtn',rep];b=series[scene,'wireless_kb',rep]
                assert all(normalized(x['state'])==normalized(y['state']) for x,y in zip(a,b))
        saved=read_csv(ROOT/'results/paper/plot_data.csv')
        index={(r['scenario'],r['method'],int(r['repeat']),int(r['arrivals'])):r for r in saved}
        for row in rows:
            ref=index[row['scenario'],row['method'],row['repeat'],row['arrivals']]
            for k,v in row.items():
                if isinstance(v,(int,float)):check(v,ref[k])
        save_csv(output/'static_plot_data.csv',rows)
        plot.service_figure(series,output);plot.cost_figure(audit['aggregate_costs'],output)
    print('Static: 60 runs, 1800 arrivals, 450 identical paired agent resource states',flush=True)
    return {'runs':60,'arrivals':1800,'paired_identical_resource_states':450,'independent_resource_audit':True}

def dynamic_audit(output):
    import numpy as np
    from dynamic_audit import audit_run
    refs={(r['method'],r['scenario'],int(r['trial_index'])):r for r in read_csv(ROOT/'results/paper/dynamic/trial_metrics.csv')}
    fairrefs={(r['method'],r['scenario'],int(r['trial_index']),r['cohort'],int(r['window_s'])):r for r in read_csv(ROOT/'results/paper/dynamic/window_fairness_by_trial_scene.csv')}
    all_rows=[];fairrows=[];initials={}
    for source in sorted((ROOT/'evidence/dynamic').iterdir()):
        method,scene,repeat=source.name.split('__');repeat=int(repeat[1:])
        with tempfile.TemporaryDirectory(prefix='wireless_dynamic_audit_') as temp:
            directory=Path(temp);unpack(source,directory)
            result=audit_run(directory)
            row=dict(method=method,scenario=scene,trial_index=repeat,**{k:v for k,v in result['summary'].items() if k not in ['method','scenario']})
            ref=refs[method,scene,repeat]
            for key,value in row.items():
                if isinstance(value,(int,float)) and key in ref:check(value,ref[key])
            initial=result['initial_state']
            identity=(initial['allocations'],initial['contracts'])
            if scene in initials:assert identity==initials[scene]
            initials[scene]=identity
            intervals=defaultdict(list)
            for interval in read_csv(directory/'traffic_intervals.csv'):intervals[interval['user_id']].append(interval)
            ids=sorted(initial['allocations'],key=int);cumulative=[]
            for uid in ids:
                items=sorted(intervals[uid],key=lambda r:float(r['start_time_s']))
                ends=[float(r['end_time_s']) for r in items]
                traffic=[float(r['traffic_mbit']) for r in items]
                cumulative.append(np.interp(np.arange(301),np.r_[0.,ends],np.r_[0.,np.cumsum(traffic)]))
            cumulative=np.array(cumulative).T
            demands=np.array([initial['contracts'][uid]['min_rate_mbps'] for uid in ids],dtype=float)
            for cohort in ['URLLC','eMBB']:
                mask=np.array([initial['contracts'][uid]['slice_type']==cohort for uid in ids])
                norm=cumulative[:,mask]/demands[mask]
                for width in range(1,301):
                    delta=norm[width:]-norm[:-width]
                    fairness=(delta.sum(axis=1)**2/(delta.shape[1]*(delta*delta).sum(axis=1))).mean()
                    check(fairness,fairrefs[method,scene,repeat,cohort,width]['mean_window_jain'])
                    fairrows.append(dict(method=method,scenario=scene,trial_index=repeat,cohort=cohort,window_s=width,mean_window_jain=float(fairness)))
            all_rows.append(row)
            print('Dynamic verified:',source.name,flush=True)
    assert len(all_rows)==33
    # First average the three scenarios within each repeat; then compute sample SD over repeats.
    aggregate=[]
    columns=['effective_user_time_pct_all30','mean_effective_throughput_mbps','mean_raw_throughput_mbps','llm_calls','mean_tokens_per_decision','mean_decision_seconds']
    for method in ['frozen','wireless_agent','tkg_dtn']:
        row={'method':method}
        for metric in columns:
            vals=[statistics.mean(r[metric] for r in all_rows if r['method']==method and r['trial_index']==rep) for rep in range(1,2 if method=='frozen' else 6)]
            row[metric]=statistics.mean(vals)
            row[metric+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else ''
        ref=next(r for r in read_csv(ROOT/'results/paper/dynamic/summary_aggregate.csv') if r['method']==method)
        for k,v in row.items():
            if k!='method' and v!='':check(v,ref[k])
        aggregate.append(row)
    save_csv(output/'dynamic_table_i.csv',aggregate)
    save_csv(output/'dynamic_trial_metrics.csv',all_rows)
    save_csv(output/'dynamic_window_fairness.csv',fairrows)
    return {'runs':33,'all_numeric_trial_metrics_match':True,'table_i_match':True,'slice_window_fairness_checks':len(fairrows),'identical_static_handoff':True}

def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    names={'tkg_dtn':'TKG-DTN','wireless_agent':'WirelessAgent','frozen':'Frozen static'}
    colors={'tkg_dtn':'#0072B2','wireless_agent':'#D55E00','frozen':'#666666'}
    service=pd.read_csv(ROOT/'results/paper/dynamic_service_data.csv')
    fair=pd.read_csv(ROOT/'results/paper/dynamic_slice_window_fairness_aggregate.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for method in names:
        data=service[service.method==method].sort_values('time_s')
        for ax,key in zip(axes[0],['effective_users','effective_throughput_mbps']):
            x=data.time_s.to_numpy();y=data[key].to_numpy();sd=data[key+'_sample_std'].fillna(0).to_numpy()
            ax.step(x,y,where='post',color=colors[method],label=names[method])
            ax.fill_between(x,y-sd,y+sd,step='post',color=colors[method],alpha=.15)
        for ax,cohort in zip(axes[1],['URLLC','eMBB']):
            data=fair[(fair.method==method)&(fair.cohort==cohort)].sort_values('window_s')
            x=data.window_s.to_numpy();y=data.equal_scene_mean_jain.to_numpy();sd=data.sample_std.fillna(0).to_numpy()
            ax.plot(x,y,color=colors[method],label=names[method]);ax.fill_between(x,y-sd,y+sd,color=colors[method],alpha=.15)
    for ax,title in zip(axes.flat,['Effective users','Effective throughput (Mbps)','URLLC Jain fairness','eMBB Jain fairness']):
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in axes[0]:ax.set_xlabel('Time (s)')
    for ax in axes[1]:ax.set_xlabel('Window length (s)')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(output/('figure_5.'+suffix),dpi=200)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['verify','static','dynamic','all'],default='all')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction')
    args=p.parse_args();verify_manifest()
    if args.mode=='verify':return
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'passed','api_calls':0,'mode':args.mode}
    if args.mode in ['static','all']:report['static']=static_audit(args.output)
    if args.mode in ['dynamic','all']:report['dynamic']=dynamic_audit(args.output);plots(args.output)
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
