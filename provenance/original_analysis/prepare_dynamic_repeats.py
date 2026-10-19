"""Audit 33 immutable runs, integrate exact intervals, and publish repeat statistics."""
from collections import defaultdict
from pathlib import Path
import json
import statistics
import numpy as np
from dynamic_repeat_auditor import audit_run
import paper_dynamic as d


def jain(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x, axis=-1)**2 / (x.shape[-1]*np.sum(x*x, axis=-1))


def stats(records, keys, group):
    result = dict(group, n=len(records))
    for key in keys:
        values = [float(r[key]) for r in records]
        result[key] = statistics.mean(values)
        result[key+'_sample_std'] = statistics.stdev(values) if len(values)>1 else ''
        result[key+'_sample_variance'] = statistics.variance(values) if len(values)>1 else ''
    return result


def main():
    d.REPORT.mkdir(parents=True, exist_ok=True)
    trials, audits, users, continuity, cohorts, windows = [], [], [], [], [], []
    source_hashes, results = {}, {}
    def bind(path, expected=None):
        value = d.sha256(path)
        if expected is not None:
            assert value == expected, str(path)
        source_hashes[d.relative(path)] = value
    batches = {d.METHOD:d.read_json(d.TKG_BATCH/'batch_manifest.json'),
               d.WIRELESS_METHOD:d.read_json(d.WIRELESS_BATCH/'batch_manifest.json')}
    for path in [d.TKG_BATCH/'batch_manifest.json',d.WIRELESS_BATCH/'batch_manifest.json',
                 d.HERE/'prepare_dynamic_repeats.py',d.HERE/'dynamic_repeat_auditor.py']:
        bind(path)
    identities = {}
    for method in d.METHODS:
        for trial in range(1, 2 if method==d.BASELINE_METHOD else 6):
            for scene in d.SCENARIOS:
                directory = d.run_directory(method,scene,trial)
                run = d.read_json(directory/'run.json')
                if method != d.BASELINE_METHOD:
                    batch = batches[method]
                    assert batch['status']=='complete'
                    if trial==1:
                        expected = batch['reference_manifests_sha256'][scene]
                    elif method==d.METHOD:
                        expected = next(w['run_manifest_sha256'] for w in batch['workers'] if w['five_sample_index']==trial and w['scene']==scene)
                    else:
                        expected = next(w['run_sha256'] for w in batch['workers'] if w['trial_index']==trial and w['scenario']==scene)
                    bind(directory/'run.json',expected)
                    ref = d.read_json(d.run_directory(method,scene)/'run.json')
                    for key in ('source_sha256','input_sha256','paper_source','model_settings','timing_mode','static_terminal_provenance'):
                        assert run[key]==ref[key],(method,trial,scene,key)
                result = audit_run(directory)
                results[method,trial,scene] = result
                row = dict(method=method,trial_index=trial,scenario=scene,
                           **{k:v for k,v in result['summary'].items() if k not in ['method','scenario']})
                trials.append(row)
                audits.append(dict(trial_index=trial,**result['audit']))
                initial = result['initial_state']
                identity = (initial['allocations'],initial['contracts'],run['input_sha256'])
                if scene in identities:
                    assert identity==identities[scene],(method,trial,scene,'different initial state/input')
                identities[scene] = identity
                for uid,total in result['user_totals'].items():
                    allocation = initial['allocations'].get(uid)
                    users.append(dict(method=method,trial_index=trial,scenario=scene,user_id=uid,
                        admitted=allocation is not None,assigned_slice=allocation['slice_type'] if allocation else '',
                        required_slice=initial['contracts'][uid]['slice_type'],**total))
                    continuity.append(dict(method=method,trial_index=trial,scenario=scene,user_id=uid,
                        all_equal=True,reference_run=d.relative(d.run_directory(d.METHOD,scene)/'run.json')))
                # Bind every raw artifact and the complete method/runtime/handoff chain.
                for base, hashes in [(directory,run['output_sha256']),
                    (d.MOBILITY,run['input_sha256']),
                    (Path(run['runtime_directory']).parents[1],run['source_sha256']),
                    (Path(run['static_terminal_dir']),run['static_terminal_provenance']['output_sha256'])]:
                    for name,expected in hashes.items():bind(base/name,expected)
                bind(directory/'run.json')
                bind(Path(run['static_terminal_dir'])/'manifest.json',run['static_terminal_provenance']['manifest_sha256'])
                original=Path(run['paper_source']['paper_run'])
                bind(original/'run.json',run['paper_source']['run_sha256'])
                for name,metadata in run['paper_source']['files'].items():
                    bind(Path(run['runtime_directory'])/name,metadata['sha256'])
                    if metadata['origin']=='paper_run_source_snapshot':bind(original/'source_snapshot'/name,metadata['sha256'])
                # Exact per-user cumulative integrals at integer endpoints.
                intervals=defaultdict(list)
                for item in d.read_csv(directory/'traffic_intervals.csv'):
                    intervals[item['user_id']].append(item)
                ids=sorted(initial['allocations'],key=int)
                cumulative=[]
                for uid in ids:
                    values=sorted(intervals[uid],key=lambda r:float(r['start_time_s']))
                    ends=np.array([float(r['end_time_s']) for r in values])
                    traffic=np.array([float(r['traffic_mbit']) for r in values])
                    cumulative.append(np.interp(np.arange(301),np.r_[0.,ends],np.r_[0.,np.cumsum(traffic)]))
                cumulative=np.array(cumulative).T
                demands=np.array([float(initial['contracts'][uid]['min_rate_mbps']) for uid in ids])
                for cohort in ('URLLC','eMBB'):
                    mask=np.array([initial['contracts'][uid]['slice_type']==cohort for uid in ids])
                    normalized=cumulative[:,mask]/demands[mask]
                    cohorts.append(dict(method=method,trial_index=trial,scenario=scene,cohort=cohort,
                        users=int(mask.sum()),cumulative_normalized_jain=float(jain(normalized[-1])),
                        mean_user_rate_mbps=float(cumulative[-1,mask].mean()/300)))
                    for width in range(1,301):
                        values=jain(normalized[width:]-normalized[:-width])
                        windows.append(dict(method=method,trial_index=trial,scenario=scene,cohort=cohort,
                            window_s=width,window_count=len(values),mean_window_jain=float(values.mean())))
                print(f'AUDITED {method} trial={trial} scene={scene}',flush=True)
    numeric=[k for k,v in trials[0].items() if isinstance(v,(int,float)) and k not in ('trial_index','duration_s','observed_users','admitted_users')]
    scene_rows=[]
    for method in d.METHODS:
        for scene in d.SCENARIOS:
            selected=[r for r in trials if r['method']==method and r['scenario']==scene]
            scene_rows.append(dict(method=method,scenario=scene,duration_s=300,observed_users=30,
                admitted_users=d.ADMITTED[scene],**stats(selected,numeric,{})))
    macro=[]
    for method in d.METHODS:
        for trial in range(1,2 if method==d.BASELINE_METHOD else 6):
            selected=[r for r in trials if r['method']==method and r['trial_index']==trial]
            assert len(selected)==3
            macro.append(dict(method=method,trial_index=trial,
                **{key:statistics.mean(float(r[key]) for r in selected) for key in numeric}))
    aggregate=[stats([r for r in macro if r['method']==method],numeric,{'method':method}) for method in d.METHODS]
    variance=[]
    for row in scene_rows+aggregate:
        for key in numeric:
            variance.append(dict(method=row['method'],scenario=row.get('scenario','three_scene_mean'),metric=key,n=row['n'],
                mean=row[key],sample_std=row[key+'_sample_std'],sample_variance=row[key+'_sample_variance']))
    # Right-continuous signal alignment preserves all fractional completion events.
    curves=[]
    for method in d.METHODS:
        selected={k:v for k,v in results.items() if k[0]==method}
        times=np.unique(np.concatenate([[float(r['time_s']) for r in v['curve_metrics']] for v in selected.values()]))
        fields=('effective_users','effective_throughput_mbps')
        values={field:[] for field in fields}
        for trial in range(1,2 if method==d.BASELINE_METHOD else 6):
            signals={field:[] for field in fields}
            for scene in d.SCENARIOS:
                data=selected[method,trial,scene]['curve_metrics']
                ts=np.array([float(r['time_s']) for r in data])
                ix=np.searchsorted(ts,times,side='right')-1
                for field in fields:signals[field].append(np.array([float(r[field]) for r in data])[ix])
            for field in fields:values[field].append(np.mean(signals[field],axis=0))
        for field in fields:
            integral=float(np.dot(np.diff(times),np.mean(values[field],axis=0)[:-1])/300)
            key='mean_'+field
            target=next(r[key] for r in aggregate if r['method']==method)
            assert np.isclose(integral,target,atol=1e-7),(method,field,integral,target)
        means={f:np.mean(v,axis=0) for f,v in values.items()}
        sds={f:np.std(v,axis=0,ddof=1) if len(v)>1 else np.zeros(len(times)) for f,v in values.items()}
        for i,time in enumerate(times):
            curves.append(dict(method=method,time_s=float(time),n=len(values[fields[0]]),
                **{key:float(means[key][i]) for key in fields},
                **{key+'_sample_std':float(sds[key][i]) for key in fields}))
    win_groups=defaultdict(list)
    for row in windows:win_groups[row['method'],row['trial_index'],row['cohort'],row['window_s']].append(row['mean_window_jain'])
    win_trial=[dict(method=m,trial_index=t,cohort=c,window_s=w,equal_scene_mean_jain=statistics.mean(v)) for (m,t,c,w),v in win_groups.items()]
    grouped=defaultdict(list)
    for row in win_trial:grouped[row['method'],row['cohort'],row['window_s']].append(row['equal_scene_mean_jain'])
    win_aggregate=[dict(method=m,cohort=c,window_s=w,n=len(v),equal_scene_mean_jain=statistics.mean(v),
        sample_std=statistics.stdev(v) if len(v)>1 else 0.,sample_variance=statistics.variance(v) if len(v)>1 else 0.) for (m,c,w),v in grouped.items()]
    tables={'trial_metrics.csv':trials,'trial_macro_metrics.csv':macro,'summary_by_scene.csv':scene_rows,
        'summary_aggregate.csv':aggregate,'repeat_statistics.csv':variance,'user_totals.csv':users,
        'initial_state_continuity.csv':continuity,'slice_metrics_by_trial_scene.csv':cohorts,
        'window_fairness_by_trial_scene.csv':windows,'window_fairness_by_trial.csv':win_trial,
        'window_fairness_aggregate.csv':win_aggregate,'dynamic_event_metrics_plot_data.csv':curves}
    for name,records in tables.items():d.write_csv(d.REPORT/name,records)
    (d.REPORT/'audit_runs.json').write_text(json.dumps({'status':'passed','expected_runs':33,'completed_runs':len(audits),'runs':audits},ensure_ascii=False,indent=2)+'\n')
    record={'status':'passed','completed_runs':33,'repeats_by_method':{d.METHOD:5,d.WIRELESS_METHOD:5,d.BASELINE_METHOD:1},
        'selection':{'tkg_dtn':'paper-selected second direct-execution run + additional_repeat_01..04',
                     'wireless_agent':'paper-selected original completion-driven run + repeat_02..05',
                     'frozen':'original deterministic baseline, not replicated'},
        'aggregate_semantics':'Equal-weight mean of three scenes within each trial; arithmetic mean, sample standard deviation and sample variance (ddof=1) of five trial means. Frozen baseline has one deterministic run per scene; no repeat uncertainty assigned.',
        'metric_semantics':'Exact modeled-rate integration over [0,300), including fractional event boundaries; all30 denominator for effective-user-time; fixed admitted cohort for Jain; costs include failures and terminal-drop workflows.',
        'latency_environment_note':'Each batch serializes its own scene workers; the TKG and Wireless batches overlapped in wall-clock time. Original WirelessAgent run also used multiple method clients. Retain measured delays; efficiency differences are not a controlled latency-isolation ablation.',
        'models_requested':sorted({r['summary']['models_requested'] for r in results.values() if r['summary']['models_requested']}),
        'models_returned':sorted({r['summary']['models_returned'] for r in results.values() if r['summary']['models_returned']}),
        'api_calls_by_publisher':0,'source_sha256':source_hashes,
        'output_sha256':{name:d.sha256(d.REPORT/name) for name in [*tables,'audit_runs.json']}}
    (d.REPORT/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('PUBLISHED 33 audited runs',flush=True)
    for row in aggregate:
        print(row['method'],{key:row[key] for key in ['n','effective_user_time_pct_all30','mean_effective_throughput_mbps','mean_decision_seconds']},flush=True)

if __name__=='__main__':main()
