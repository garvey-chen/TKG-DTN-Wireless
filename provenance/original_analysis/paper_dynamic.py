"""Verified five-repeat dynamic evidence used by the manuscript (no model calls)."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
BASELINE_DIR = HERE / '动态实验_1秒采样_完成即执行_20261016'
TKG_DIR = HERE / '动态实验_1秒采样_直接执行_20261016_第二次运行'
TKG_BATCH = HERE / '动态实验_TKGDTN_补跑4次_20261017'
WIRELESS_BATCH = HERE / '动态实验_WirelessAgent_补跑4次_20261017'
EVENT_DIR = HERE / 'dynamic_publication_five_repeats_20261017'
REPORT = EVENT_DIR / 'summary'
MOBILITY = HERE / '静态衔接动态场景_20261014/1秒粒度移动数据_20261015'
METHOD, WIRELESS_METHOD, BASELINE_METHOD = 'tkg_dtn', 'wireless_agent', 'frozen'
METHODS = (METHOD, WIRELESS_METHOD, BASELINE_METHOD)
SOURCE_DIRS = {METHOD: TKG_DIR, WIRELESS_METHOD: BASELINE_DIR, BASELINE_METHOD: BASELINE_DIR}
ADMITTED = {'center': 27, 'north': 30, 'south': 29}
SCENARIOS = tuple(ADMITTED)
EXPERIMENT = PROJECT / 'DTN-KG-transfer-package-20260909/DTN-KG-new-transfer-20260909/DTN-KG-new/experiments_Wireless_v2.1'
_HASH_CACHE = {}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    path = Path(os.path.abspath(path))
    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = identity(path.stat())
    if path in _HASH_CACHE and _HASH_CACHE[path][0] == before:
        return _HASH_CACHE[path][1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        assert identity(os.fstat(stream.fileno())) == before
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        assert identity(os.fstat(stream.fileno())) == before
    assert identity(path.stat()) == before, str(path)
    value = digest.hexdigest()
    _HASH_CACHE[path] = before, value
    return value


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return Path(os.path.abspath(path)).relative_to(PROJECT).as_posix()


def run_directory(method, scene, trial=1):
    assert method in METHODS and scene in ADMITTED
    assert 1 <= trial <= (1 if method == BASELINE_METHOD else 5)
    source = SOURCE_DIRS[method]
    if trial > 1:
        source = (TKG_BATCH / f'additional_repeat_{trial-1:02d}' if method == METHOD
                  else WIRELESS_BATCH / f'repeat_{trial:02d}')
    return source / 'runs' / f'{method}_{scene}'


def _verify_completed():
    record = read_json(REPORT / 'summary.json')
    assert record['status'] == 'passed' and record['completed_runs'] == 33
    assert record['repeats_by_method'] == {METHOD: 5, WIRELESS_METHOD: 5, BASELINE_METHOD: 1}
    checked = {}
    for group, base in [('source_sha256', PROJECT), ('output_sha256', REPORT)]:
        for name, expected in record[group].items():
            path = base / name
            value = checked[path] = sha256(path)
            assert value == expected, f'Dynamic evidence changed: {path}'
    checked[REPORT/'summary.json'] = sha256(REPORT/'summary.json')
    rows = read_csv(REPORT/'summary_by_scene.csv')
    assert len(rows) == 9
    return rows, checked


def load_completed():
    """Nine method/scene means; agent scene means each contain five repeats."""
    return _verify_completed()[0]


def aggregates():
    _verify_completed()
    return read_csv(REPORT/'summary_aggregate.csv')


def evidence_paths():
    return sorted(_verify_completed()[1], key=str)


def publish_dynamic_data():
    rows, checked = _verify_completed()
    report = read_json(REPORT/'summary.json')
    published = {}
    aliases = {
        'dynamic_scene_statistics.csv':'summary_by_scene.csv',
        'dynamic_summary_statistics.csv':'summary_aggregate.csv',
        'dynamic_trial_statistics.csv':'trial_metrics.csv',
        'dynamic_repeat_variance.csv':'repeat_statistics.csv',
        'dynamic_user_totals.csv':'user_totals.csv',
        'dynamic_initial_state_continuity.csv':'initial_state_continuity.csv',
        'wireless_comparison_scene_statistics.csv':'summary_by_scene.csv',
        'dynamic_run_cost_statistics.csv':'trial_metrics.csv',
        'wireless_comparison_run_costs.csv':'trial_metrics.csv',
        'dynamic_slice_window_fairness_aggregate.csv':'window_fairness_aggregate.csv',
    }
    for name, source in aliases.items():
        shutil.copyfile(REPORT/source, HERE/name)
        published[name] = {'source':relative(REPORT/source),'sha256':sha256(HERE/name),'transformation':'byte-identical copy'}
    manifest = {
        'active_experiment':'five independent online repeats per agent/scene; one deterministic frozen baseline per scene',
        'updated_date_asia_shanghai':'2026-10-17','agent_mode':'adaptive',
        'event_directory':relative(EVENT_DIR),'report_directory':relative(REPORT),
        'input_directory':relative(MOBILITY),'displayed_methods':list(METHODS),
        'completed_runs':33,'repeats_by_method':report['repeats_by_method'],
        'observed_users_per_scene':30,'initial_admitted_users':ADMITTED,
        'duration_s':300,'sample_interval_s':1,'frames_per_scene':301,
        'integration_interval':'[0,300), split at exact fractional-second completion events',
        'aggregate_semantics':report['aggregate_semantics'],
        'metric_semantics':report['metric_semantics'],
        'source_selection':report['selection'],
        'primary_fairness_population':'Fixed admitted cohorts grouped by contract slice, including wrong-slice and ineffective users',
        'cost_boundary':'Full measured policy duration; failed and terminal-drop workflows included; external guard cost separate',
        'model_requested':report['models_requested'],'model_returned':report['models_returned'],
        'source_sha256':{relative(path):value for path,value in checked.items()},
        'published_files':published,'publisher_sha256':sha256(__file__),
        'prior_paper_backup':'build/before_dynamic_five_repeats_remove_table3_20261017',
    }
    figure_path = HERE/'dynamic_figure_validation.json'
    if figure_path.exists():
        figure = read_json(figure_path)
        if figure.get('publication_directory') == relative(EVENT_DIR) and figure.get('script_sha256') == sha256(HERE/'plot_dynamic_figures.py'):
            for name, expected in figure['output_sha256'].items():
                assert sha256(HERE/name) == expected
                manifest['published_files'][name] = {'sha256':expected,'source':'plot_dynamic_figures.py','transformation':'five-repeat mean and sample standard deviation'}
            manifest['figure_validation_sha256'] = sha256(figure_path)
    (HERE/'dynamic_experiment_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest
