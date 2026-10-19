"""Slice fairness prose from all five repeats; Table 3/P10 is removed."""
import statistics
from paper_dynamic import HERE, REPORT, METHOD, WIRELESS_METHOD, BASELINE_METHOD, read_csv, read_json, sha256


def verify():
    record=read_json(REPORT/'summary.json')
    assert record['status']=='passed' and record['completed_runs']==33
    for name in ('slice_metrics_by_trial_scene.csv','window_fairness_by_trial_scene.csv','window_fairness_by_trial.csv','window_fairness_aggregate.csv'):
        assert sha256(REPORT/name)==record['output_sha256'][name]
    return {'status':'passed','source_sha256':{},'output_sha256':{},'repeats_per_agent_scene':5}


def fairness(method,cohort,width=300):
    rows=[r for r in read_csv(REPORT/'window_fairness_aggregate.csv') if r['method']==method and r['cohort']==cohort and int(r['window_s'])==width]
    assert len(rows)==1
    return float(rows[0]['equal_scene_mean_jain']),float(rows[0]['sample_std'])


def scene_fairness(method,scene,cohort,width):
    values=[float(r['mean_window_jain']) for r in read_csv(REPORT/'window_fairness_by_trial_scene.csv')
        if r['method']==method and r['scenario']==scene and r['cohort']==cohort and int(r['window_s'])==width]
    assert len(values)==5
    return statistics.mean(values)


def supplement_blocks():
    w,t=WIRELESS_METHOD,METHOD
    jw,jt=[fairness(m,'URLLC') for m in (w,t)]
    ew,et=[fairness(m,'eMBB') for m in (w,t)]
    sw,st=[fairness(m,'eMBB',1)[0] for m in (w,t)]
    return [
        ('p','时序图谱对历史服务量的持续追踪为跨时段补偿提供了依据，长期公平性优势随评价窗口拉长而显现。'
         '图 5（c）（d）展示了统计窗口 W 从 1 s 至 300 s 变化时的切片内需求归一化 Jain 指数。'
         f'在 eMBB 切片中，W=1 s 时 WirelessAgent 的平均指数为 {sw:.4f}，高于 TKG-DTN 的 {st:.4f}；'
         f'随着窗口延长，TKG-DTN 逐步反超，至 W=300 s 时达到 {et[0]:.4f}±{et[1]:.4f}，高于 WirelessAgent 的 {ew[0]:.4f}±{ew[1]:.4f}。'
         '这一变化体现了瞬时速率均衡与跨时段服务量均衡的差异，与融合历史累计流量、补偿欠服务用户的规划机制相吻合。'
         f'URLLC 切片的长期收益更为突出：300 s 窗口下，平均 Jain 指数由 {jw[0]:.4f}±{jw[1]:.4f} 提升至 {jt[0]:.4f}±{jt[1]:.4f}，相对提高 {100*(jt[0]/jw[0]-1):.2f}%。'
         'URLLC 单用户带宽范围为 1–5 MHz，采用 1 MHz 整数粒度，资源池仅为 30 MHz；离散分配对单用户资源份额的相对影响更大，跨时段调整因而具有更大的补偿空间。'
         '两类切片的结果表明，历史服务量参与规划有助于将短时资源分配转化为更均衡的长期服务。'),
    ]
