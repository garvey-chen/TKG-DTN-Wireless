#!/usr/bin/env python3
"""Publish figures from the five-repeat 1 s completion-driven experiment.

No model calls; all original runs and the audited summary remain read-only.
Figure 4 retains the map style and hides user IDs. Figure 5 combines the
three scenes in four service/fairness panels. Timing remains an auxiliary figure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np

from paper_dynamic import (HERE, PROJECT, EVENT_DIR, REPORT, MOBILITY, EXPERIMENT,
    METHOD, WIRELESS_METHOD, BASELINE_METHOD, METHODS, ADMITTED, SCENARIOS,
    read_csv, read_json, sha256, write_csv, run_directory, relative, publish_dynamic_data)

COLORS = {METHOD: "#2468A2", WIRELESS_METHOD: "#D78035", BASELINE_METHOD: "#7F8992"}
CAPTIONS = {
    "dynamic_mobility": "静态终态衔接的动态用户移动轨迹。中部、北部和南部各含30名用户，静止、持续移动和间歇移动各10名；仿真时长300 s，按1 s采样，由原10 s轨迹锚点线性细分。实线与虚线分别表示持续移动与间歇移动轨迹，空心圆与实心圆分别标记起点和终点，灰点为静止用户，星号为基站。新增采样点为派生仿真输入。",
    "dynamic_service": '动态服务性能与切片内公平性的三场景加权平均结果。（a）平均有效服务人数；（b）平均有效吞吐量；（c）URLLC 切片内公平性；（d）eMBB 切片内公平性。（a）（b）按各场景总用户数加权，每场景 30 人，权重均为 1/3，以直线连接相邻数据点；（c）（d）为不同窗口长度 W 下的需求归一化 Jain 指数，先对各场景的完整滑动窗口求均值，再对三场景等权平均。（c）纵轴采用断轴展示，（d）纵轴范围局部放大。蓝色实线、橙色虚线和灰色点线分别表示 TKG-DTN、WirelessAgent（瞬时公平适配）和冻结静态终态。',
    "dynamic_timing": "完整策略推理与执行时间轴。每个条带从当前轮读取最新已到达整数秒快照开始，延伸至策略完成；完成时间小于300 s时立即重校验候选，并在同一时刻启动下一轮。实心圆、空心圆、叉号和三角形分别表示实际调整、保持、拒绝与策略失败；斜纹表示越过300 s终点的轮次，其候选丢弃而完整调用开销仍计入。外部执行守卫、遥测回放与输出记账在模拟时序中视为瞬时，守卫实际耗时另行记录。",
    "dynamic_costs": "辅助图：完成驱动实验的实际模型调用、总Token及完整策略耗时。蓝色和橙色分别表示TKG-DTN与WirelessAgent；前两项按场景累计，第三项柱形为单轮均值、菱形为最大值。所有越过300 s终点的已启动轮次均计入开销；冻结静态分配无模型调用，外部执行守卫耗时单独列于成本表。"
}


def current_captions():
    import re
    from manuscript_source import plain_tex
    text = (HERE / 'wireless_v2_revised_section_cn.tex').read_text(encoding='utf-8')
    return {m[1]: plain_tex(m[2]) for m in re.finditer(
        r'\\includegraphics\[[^\]]+\]\{([^}]+)\.pdf\}\n\\caption\{([^\n]*)\}', text)}


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-11)


def load_mobility(experiment):
    generation = read_json(MOBILITY / "generation.json")
    tracks, cohorts, sources = {}, {}, {}
    for scene in SCENARIOS:
        path = MOBILITY / f"{scene}.csv"
        rows = read_csv(path)
        assert len(rows) == 9030
        tracks[scene] = defaultdict(list)
        for row in rows:
            assert row["Scenario"] == scene
            tracks[scene][row["RX_ID"]].append(row)
        assert set(tracks[scene]) == {str(i) for i in range(1, 31)}
        for user_rows in tracks[scene].values():
            user_rows.sort(key=lambda r: int(r["Time_s"]))
            assert [int(r["Time_s"]) for r in user_rows] == list(range(301))
        assert Counter(r[0]["Mobility_Profile"] for r in tracks[scene].values()) == {"static": 10, "continuous": 10, "intermittent": 10}
        run = read_json(run_directory(METHOD, scene) / "run.json")
        cohorts[scene] = set(run["initial_admitted_user_ids"])
        # Match the original static XYZ used by the exact handoff, for all 30 users.
        static = {r["user_id"]: r for r in read_csv(Path(run["static_terminal_dir"]) / scene / "static_terminal_users.csv")}
        for uid, trajectory in tracks[scene].items():
            assert all(close(trajectory[0][axis], static[uid]["original_" + axis.lower()]) for axis in ("X", "Y", "Z"))
        map_path = experiment.parent / "experiments_Wireless_v2" / generation["scenarios"][scene]["map_file"]
        assert sha256(map_path) == generation["scenarios"][scene]["map_sha256"]
        sources[relative(map_path)] = sha256(map_path)
    for path in (experiment / "generate_scenarios.py", experiment / "plot_mobility.py"):
        sources[relative(path)] = sha256(path)
    return tracks, cohorts, generation, sources


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8C8C8C")
    ax.tick_params(length=3, color="#8C8C8C")

def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    fig.savefig(output / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)

def place_map_labels(ax, labels):
    """Adapt plot_mobility._place_labels to readable two-column print sizes.

    The original nearest-position scoring and thin leaders are retained.
    Larger text and more candidate offsets accommodate 30 labels per panel.
    """
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    # Reserve the already drawn BS label as well as subsequent user labels.
    occupied = [text.get_window_extent(renderer).expanded(1.08, 1.12) for text in ax.texts]
    anchors = [ax.transData.transform((x, y)) for _, x, y, _ in labels]
    candidates = [(5, 5), (5, -10), (-5, 5), (-5, -10), (9, 0), (-9, 0),
                  (0, 11), (0, -14), (12, 9), (-12, 9), (12, -12), (-12, -12),
                  (20, 0), (-20, 0), (0, 21), (0, -23), (21, 15), (-21, 15),
                  (21, -19), (-21, -19), (30, 8), (-30, 8), (8, 29), (-8, -31),
                  (30, 20), (-30, 20), (30, -20), (-30, -20), (0, 34), (0, -34)]
    for label_index, (uid, x, y, color) in enumerate(labels):
        annotation = ax.annotate(
            uid, (x, y), xytext=(5, 5), textcoords="offset points",
            fontsize=10.5, fontweight="normal", color=color, zorder=12,
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.4,
                        "alpha": 0.65, "shrinkA": 2, "shrinkB": 3},
            annotation_clip=False,
        )
        best, best_score = None, float("inf")
        for dx, dy in candidates:
            annotation.set_position((dx, dy))
            annotation.set_horizontalalignment("left" if dx > 0 else "right" if dx < 0 else "center")
            annotation.set_verticalalignment("bottom" if dy > 0 else "top" if dy < 0 else "center")
            annotation.update_positions(renderer)
            annotation.update_bbox_position_size(renderer)
            box = annotation.get_bbox_patch().get_window_extent(renderer).expanded(1.08, 1.12)
            overlaps = sum(box.overlaps(other) for other in occupied)
            covered_points = sum(box.contains(*anchor) for j, anchor in enumerate(anchors) if j != label_index)
            outside = not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1)
            score = 1000 * overlaps + 100 * covered_points + 2000 * outside + math.hypot(dx, dy)
            if score < best_score:
                best, best_score = (dx, dy, Bbox.from_extents(*box.extents)), score
        dx, dy, box = best
        annotation.set_position((dx, dy))
        annotation.set_horizontalalignment("left" if dx > 0 else "right" if dx < 0 else "center")
        annotation.set_verticalalignment("bottom" if dy > 0 else "top" if dy < 0 else "center")
        occupied.append(box)

def refine_crowded_labels(ax):
    """Extend Figure 3's nearby candidates only when labels still overlap."""
    renderer = ax.figure.canvas.get_renderer()
    annotations = list(ax.texts)

    def box(text):
        text.update_positions(renderer)
        text.update_bbox_position_size(renderer)
        return text.get_bbox_patch().get_window_extent(renderer).expanded(1.06, 1.12)

    candidates = [(radius * math.cos(angle), radius * math.sin(angle))
                  for radius in (10, 15, 20, 25, 30, 36, 43, 51, 60, 70)
                  for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    for _ in range(2):
        for annotation in annotations:
            if not annotation.get_text().isdigit():
                continue
            others = [box(text) for text in annotations if text is not annotation]
            current = box(annotation)
            if not any(current.overlaps(other) for other in others):
                continue
            best, best_score = None, float("inf")
            for dx, dy in [annotation.get_position(), *candidates]:
                annotation.set_position((dx, dy))
                annotation.set_horizontalalignment("left" if dx > 1 else "right" if dx < -1 else "center")
                annotation.set_verticalalignment("bottom" if dy > 1 else "top" if dy < -1 else "center")
                candidate = box(annotation)
                overlaps = sum(candidate.overlaps(other) for other in others)
                outside = not ax.bbox.contains(candidate.x0, candidate.y0) or not ax.bbox.contains(candidate.x1, candidate.y1)
                score = 10000 * overlaps + 20000 * outside + math.hypot(dx, dy)
                if score < best_score:
                    best, best_score = (dx, dy), score
            dx, dy = best
            annotation.set_position(best)
            annotation.set_horizontalalignment("left" if dx > 1 else "right" if dx < -1 else "center")
            annotation.set_verticalalignment("bottom" if dy > 1 else "top" if dy < -1 else "center")
    boxes = [box(text) for text in annotations]
    return int(sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))))

def mobility_figure(experiment, tracks, cohorts, generation, output):
    import matplotlib.patheffects as effects

    spec = importlib.util.spec_from_file_location("dynamic_figure_geometry", experiment / "generate_scenarios.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    spec = importlib.util.spec_from_file_location("dynamic_mobility_style", experiment / "plot_mobility.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = source.load_model()
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 4.5))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.24, top=0.92, wspace=0.26)
    map_rows = []
    coordinate_frames = {}
    for index, (ax, scene) in enumerate(zip(axes, SCENARIOS)):
        buildings, tx, origin, original_bounds = source.scene_geometry(model, scene)
        bounds = generation["scenarios"][scene]["sampling_bounds_xy_m"]
        assert all(close(a, b) for a, b in zip(tx, generation["scenarios"][scene]["tx_position_m"]))
        # Translate the complete scene into a BS-centered display frame.
        # Raw trajectory coordinates and the experiment's physical model stay intact.
        display_origin = np.asarray(tx[:2], dtype=float)
        display_tx = np.asarray(tx[:2], dtype=float) - display_origin
        coordinate_frames[scene] = {"source_bs_xy_m": display_origin.tolist(),
                                    "display_bs_xy_m": display_tx.tolist()}
        for building in buildings:
            nodes = np.asarray(building["nodes"], dtype=float) - display_origin
            ax.add_patch(Polygon(nodes, facecolor="#E5E8EC", edgecolor="#B4BBC4", linewidth=0.5, zorder=1))
        for uid, rows in sorted(tracks[scene].items(), key=lambda pair: int(pair[0])):
            xs = [float(r["X"]) - display_origin[0] for r in rows]
            ys = [float(r["Y"]) - display_origin[1] for r in rows]
            profile = rows[0]["Mobility_Profile"]
            color = reference.STATIC_COLOR if profile == "static" else reference._color(uid)
            if profile == "static":
                ax.scatter(xs[0], ys[0], s=19, marker="o", color=color,
                           linewidths=0.5, edgecolors="white", zorder=8)
            else:
                line, = ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.91,
                                linestyle=reference._style(profile), zorder=4, solid_capstyle="round")
                line.set_path_effects([effects.Stroke(linewidth=1.9, foreground="white", alpha=0.7), effects.Normal()])
                ax.scatter(xs[0], ys[0], s=26, facecolors="white", edgecolors=color,
                           linewidths=0.9, zorder=9)
                ax.scatter(xs[-1], ys[-1], s=16, color=color, linewidths=0.4,
                           edgecolors="white", zorder=10)
            admitted = uid in cohorts[scene]
            for r, x, y in zip(rows, xs, ys):
                map_rows.append({"scenario": scene, "user_id": uid, "time_s": int(r["Time_s"]), "x_m": r["X"], "y_m": r["Y"], "x_bs_m": x, "y_bs_m": y, "cqi": int(r["CQI"]), "mobility_profile": profile, "admitted": admitted, "original_10s_anchor": int(r["Time_s"]) % 10 == 0})
        ax.scatter(*display_tx, marker="*", s=110, color="#BE3434", linewidth=0.65, edgecolor="white", zorder=11)
        ax.annotate("BS", display_tx, xytext=(6, -7), textcoords="offset points", fontsize=10.5,
                    color="#A12D2D", fontweight="bold", zorder=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5})
        # Equal-sized square panels without stretching the physical geometry.
        # Translate the view window by the same offset and pad it to a square.
        # Shared axis titles leave the gutters exclusively for tick labels.
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1.06
        cx = (bounds[0] + bounds[1]) / 2 - display_origin[0]
        cy = (bounds[2] + bounds[3]) / 2 - display_origin[1]
        display_limits = (cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2)
        if scene == "north":
            display_limits = (-230.0, 90.0, -150.0, 170.0)
        xmin, xmax, ymin, ymax = display_limits
        assert all(xmin < float(r["X"]) - display_origin[0] < xmax
                   and ymin < float(r["Y"]) - display_origin[1] < ymax
                   for user_rows in tracks[scene].values() for r in user_rows)
        coordinate_frames[scene].update({"display_limits_xy_m": list(display_limits),
                                         "visible_trajectory_samples": sum(map(len, tracks[scene].values()))})
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xticks([t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
        ax.set_yticks([t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]])
        ax.grid(color="#D9DEE5", linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#ADB5BD")
            spine.set_linewidth(0.7)
        ax.tick_params(colors="#59636D", labelsize=10.5, length=3)
        ax.xaxis.label.set_color("#44515E")
        ax.yaxis.label.set_color("#44515E")
        ax.set_title(f"({chr(97+index)}) {scene.title()}  |  30 users", loc="left", fontsize=12,
                     fontweight="bold", color="#24313F", pad=11)
    fig.supylabel("Local Y (m)", x=0.018, y=0.60, fontsize=11.5, color="#44515E")
    fig.supxlabel("Local X (m)", x=0.535, y=0.19, fontsize=11.5, color="#44515E")
    handles = [
        Line2D([], [], color="#356CA0", lw=1.2, label="Continuous movement"),
        Line2D([], [], color="#B25680", lw=1.2, ls=reference._style("intermittent"), label="Stop-and-go movement"),
        Line2D([], [], marker="o", color=reference.STATIC_COLOR, ls="none", markersize=4, label="Static user"),
        Line2D([], [], marker="o", markeredgecolor="#356CA0", markerfacecolor="white", markeredgewidth=0.9, ls="none", markersize=5, label="Start (0 s)"),
        Line2D([], [], marker="o", color="#356CA0", ls="none", markersize=4, label="End (300 s)"),
        Line2D([], [], marker="*", color="#BE3434", ls="none", markersize=10, label="Base station"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.005), ncol=3,
               frameon=False, fontsize=11, columnspacing=2.1, handlelength=2.0, labelspacing=0.7)
    fig.canvas.draw()
    # Keep only the base-station annotation; user IDs remain in the source CSV.
    assert all([text.get_text() for text in ax.texts] == ["BS"] for ax in axes)
    boxes = [list(ax.get_window_extent().bounds) for ax in axes]
    assert all(close(box[2], boxes[0][2]) and close(box[3], boxes[0][3])
               and close(box[1], boxes[0][1]) and close(box[2], box[3]) for box in boxes)
    renderer = fig.canvas.get_renderer()
    shared_titles = [t.get_window_extent(renderer) for t in fig.texts]
    for ax in axes:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax._left_title]:
            if label.get_visible() and label.get_text():
                assert not any(label.get_window_extent(renderer).overlaps(title)
                               for title in shared_titles), "Shared axis title overlaps a tick or panel title"
    map_layout = {"equal_square_panels": True, "aligned_panel_bottoms": True,
                  "axis_boxes_pixels_at_figure_dpi": boxes, "figure_dpi": fig.dpi,
                  "shared_axis_titles": [t.get_text() for t in fig.texts],
                  "shared_axis_titles_clear_of_ticks_and_titles": True,
                  "coordinate_transform": "display_xy = source_xy - source_bs_xy; each base station is at (0, 0)",
                  "coordinate_frames": coordinate_frames,
                  "all_90_initial_xyz_match_static": True, "user_id_label_collisions": 0,
                  "user_id_labels_shown": False, "user_id_label_count": 0,
                  "legend_labels": [handle.get_label() for handle in handles],
                  "window_rule": "BS-centered square panels; North uses a 320 m crop retaining all 9030 samples, other scenes use map bounds plus 6% margin"}
    save_figure(fig, output, "dynamic_mobility")
    data_path = output / "dynamic_mobility_data.csv"
    write_csv(data_path, map_rows)
    return map_layout

def service_figure(output):
    captions = current_captions()
    source=REPORT/'dynamic_event_metrics_plot_data.csv'
    rows=read_csv(source)
    fairness_path=REPORT/'window_fairness_aggregate.csv'
    fairness_rows=read_csv(fairness_path)
    from plot_figures import COLORS as static_colors
    colors={METHOD:static_colors['tkg_dtn'],WIRELESS_METHOD:static_colors['wireless_kb'],BASELINE_METHOD:static_colors['rule_upstream']}
    styles={METHOD:'-',WIRELESS_METHOD:(0,(4.,4.)),BASELINE_METHOD:':'}
    labels={METHOD:'TKG-DTN',WIRELESS_METHOD:'WirelessAgent',BASELINE_METHOD:'Frozen static'}
    fig,grid=plt.subplots(2,2,figsize=(10.7,7.6))
    def draw(ax,method,x,mean,sd):
        ax.plot(x,mean,color=colors[method],ls=styles[method],lw=1.8,label=labels[method],zorder=4 if method!=BASELINE_METHOD else 2)
        if method!=BASELINE_METHOD:
            ax.fill_between(x,np.asarray(mean)-sd,np.asarray(mean)+sd,color=colors[method],alpha=.17,linewidth=0,zorder=1)
    checks={}
    aggregates={r['method']:r for r in read_csv(REPORT/'summary_aggregate.csv')}
    for ax,field,title,ylabel in zip(grid[0],('effective_users','effective_throughput_mbps'),
            ('(a) Mean effective users','(b) Mean effective throughput'),('Mean number of users','Mean throughput (Mbps)')):
        for method in METHODS:
            selected=[r for r in rows if r['method']==method]
            x=np.array([float(r['time_s']) for r in selected])
            mean=np.array([float(r[field]) for r in selected])
            sd=np.array([float(r[field+'_sample_std']) for r in selected])
            integral=float(np.dot(np.diff(x),mean[:-1])/300)
            expected=float(aggregates[method]['mean_'+field])
            assert close(integral,expected),(method,field,integral,expected)
            checks[method+'_'+field]={'curve_integral':integral,'summary_mean':expected}
            draw(ax,method,x,mean,sd)
        style_axis(ax);ax.set(xlim=(0,300),xlabel='Time (s)',ylabel=ylabel)
        ax.set_title(title,pad=10,fontsize=12);ax.set_xticks(range(0,301,60));ax.margins(y=.12)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    cell=grid[1,0].get_subplotspec();grid[1,0].remove()
    split=cell.subgridspec(2,1,height_ratios=[3.2,1],hspace=.12)
    upper=fig.add_subplot(split[0]);lower=fig.add_subplot(split[1],sharex=upper)
    grid[1,0]=upper
    axis_layout={}
    for ax,cohort,title in zip(grid[1],('URLLC','eMBB'),('(c) URLLC fairness','(d) eMBB fairness')):
        bounds={'active':[],'frozen':[]}
        for method in METHODS:
            selected=sorted([r for r in fairness_rows if r['method']==method and r['cohort']==cohort],key=lambda r:int(r['window_s']))
            x=[int(r['window_s']) for r in selected];assert x==list(range(1,301))
            mean=np.array([float(r['equal_scene_mean_jain']) for r in selected])
            sd=np.array([float(r['sample_std']) for r in selected])
            assert np.all(mean-sd>0) and np.all(mean+sd<=1)
            target=lower if cohort=='URLLC' and method==BASELINE_METHOD else ax
            draw(target,method,x,mean,sd)
            bounds['frozen' if method==BASELINE_METHOD else 'active'].extend(np.r_[mean-sd,mean+sd])
        style_axis(ax);ax.set(xlim=(0,300),xlabel='Window length W (s)',ylabel='Mean Jain index')
        ax.set_title(title,pad=10,fontsize=12);ax.set_xticks(range(0,301,60))
        if cohort=='URLLC':
            active=bounds['active'];frozen=bounds['frozen']
            pad=(max(active)-min(active))*.10
            active_limits=(min(active)-pad,min(1.,max(active)+pad))
            # Keep the baseline visible and visibly label the omitted y range.
            pad_f=max((max(frozen)-min(frozen))*.3,.001)
            frozen_limits=(min(frozen)-pad_f,max(frozen)+pad_f)
            ax.set_ylim(active_limits);ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel('');ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
            style_axis(lower);lower.set(xlim=(0,300),ylim=frozen_limits,xlabel='Window length W (s)')
            lower.set_xticks(range(0,301,60));lower.yaxis.set_major_locator(MaxNLocator(2))
            ax.plot([-.012,.012],[-.018,.018],transform=ax.transAxes,color='#666666',lw=1,clip_on=False)
            lower.plot([-.012,.012],[1-.058,1+.058],transform=lower.transAxes,color='#666666',lw=1,clip_on=False)
            axis_layout[cohort]={'scale':'linear with marked y-axis break','active_limits':active_limits,'frozen_limits':frozen_limits}
        else:
            values=bounds['active']+bounds['frozen'];pad=(max(values)-min(values))*.12
            ax.set_ylim(min(values)-pad,min(1.,max(values)+pad));ax.yaxis.set_major_locator(MaxNLocator(5))
            axis_layout[cohort]={'scale':'linear with local y range','limits':ax.get_ylim()}
    handles=[Line2D([],[],color=colors[m],ls=styles[m],lw=1.8,label=labels[m]) for m in METHODS]
    handles.append(Patch(facecolor='#777777',alpha=.17,label='±1 sample SD (5 repeats)'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=4,frameon=False,fontsize=10,handlelength=2.5,columnspacing=1.4)
    fig.subplots_adjust(left=.085,right=.985,bottom=.10,top=.90,wspace=.34,hspace=.43)
    save_figure(fig,output,'dynamic_service')
    for target in ['dynamic_service_data.csv','dynamic_service_weighted_data.csv']:
        shutil.copyfile(source,output/target)
    shutil.copyfile(fairness_path,output/'dynamic_service_fairness_data.csv')
    return {'panels':['effective_users','effective_throughput_mbps','URLLC_window_jain','eMBB_window_jain'],
        'layout':'2 rows by 2 columns','aggregation':'scene mean within each trial, then mean and sample SD across five trials',
        'frozen_uncertainty':'one deterministic trajectory, no repeat band','source_rows':len(rows),
        'alignment':'union of exact completion and telemetry times across all scenes and repeats; last observation carried forward',
        'statistical_integration':'right-continuous signals; display joins adjacent points with straight lines',
        'time_integral_checks':checks,'fairness_axis_layout':axis_layout,'captions':captions}


def cost_figure(output):
    data={r['method']:r for r in read_csv(REPORT/'summary_aggregate.csv')}
    fields=['llm_calls','total_tokens','mean_decision_seconds']
    labels=['Model calls per scene','Total tokens per scene (thousands)','Policy duration per decision (s)']
    fig,axes=plt.subplots(1,3,figsize=(10.7,3.6))
    out=[]
    for ax,key,label in zip(axes,fields,labels):
        scale=1000 if key=='total_tokens' else 1
        for i,m in enumerate((METHOD,WIRELESS_METHOD)):
            mean=float(data[m][key])/scale;sd=float(data[m][key+'_sample_std'])/scale
            ax.bar(i,mean,yerr=sd,color=COLORS[m],width=.55,capsize=3)
            out.append(dict(method=m,metric=key,n=5,mean=mean,sample_std=sd))
        ax.set_xticks([0,1],['TKG-DTN','WirelessAgent']);ax.set_ylabel(label);style_axis(ax);ax.margins(y=.2)
    fig.tight_layout();save_figure(fig,output,'dynamic_costs');write_csv(output/'dynamic_costs_data.csv',out)
    return {'role':'auxiliary','uncertainty':'five-repeat sample SD after equal-scene averaging'}


def main():
    manifest=publish_dynamic_data()
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    # Regenerate Figure 4 presentation while preserving the audited trajectories.
    old=read_json(HERE/'build/before_dynamic_five_repeats_remove_table3_20261017/dynamic_figure_validation.json')
    assert sha256(HERE/'dynamic_mobility_data.csv') == old['output_sha256']['dynamic_mobility_data.csv']
    tracks,cohorts,generation,geometry_sources=load_mobility(EXPERIMENT)
    map_layout=mobility_figure(EXPERIMENT,tracks,cohorts,generation,HERE)
    assert map_layout['coordinate_frames'] == old['map_layout']['coordinate_frames']
    assert sha256(HERE/'dynamic_mobility_data.csv') == old['output_sha256']['dynamic_mobility_data.csv']
    service=service_figure(HERE)
    cost=cost_figure(HERE)
    outputs=[stem+suffix for stem in ['dynamic_mobility','dynamic_service','dynamic_costs'] for suffix in ['.png','.pdf','_data.csv']]
    outputs+=['dynamic_service_weighted_data.csv','dynamic_service_fairness_data.csv']
    captions = current_captions()
    audit={'verification_status':'passed','completed_runs':33,'repeats_by_method':{METHOD:5,WIRELESS_METHOD:5,BASELINE_METHOD:1},
        'api_calls_by_publisher':0,'publication_directory':relative(EVENT_DIR),'main_paper_figures':{'4':'dynamic_mobility','5':'dynamic_service'},
        'source_sha256':{**manifest['source_sha256'],**geometry_sources},
        'output_sha256':{name:sha256(HERE/name) for name in outputs},
        'script_sha256':sha256(__file__),'configuration_sha256':sha256(HERE/'paper_dynamic.py'),
        'map_layout':map_layout,'service_layout':service,'cost_layout':cost,'captions':captions,
        'checks':{'all_33_runs_independently_audited':True,'map_geometry_and_mobility_data_unchanged':True, 'redundant_map_heading_removed':True,
            'service_curves_integrate_to_repeat_means':True,'fairness_bands_use_repeat_sd':True}}
    (HERE/'dynamic_figure_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    # Figure 1 still uses the identical geometry and initial points. Refresh its
    # dependency on this presentation manifest without redrawing the image.
    static_path = HERE/'static_distribution_validation.json'
    if static_path.exists():
        static = read_json(static_path)
        for name, expected in static['output_sha256'].items():
            assert sha256(HERE/name) == expected, name
        dependency = relative(HERE/'dynamic_figure_validation.json')
        for name, expected in static['source_sha256'].items():
            if name != dependency:
                assert sha256(PROJECT/name) == expected, name
        static['source_sha256'][dependency] = sha256(HERE/'dynamic_figure_validation.json')
        static['binding_refresh_note'] = 'Figure 4 heading removed; map geometry and trajectory data unchanged. Figure 1 image assets retained.'
        static_path.write_text(json.dumps(static,ensure_ascii=False,indent=2)+'\n')
    publish_dynamic_data()
    print('Updated dynamic figures; map geometry and all trajectory values preserved.')


if __name__=='__main__':main()
