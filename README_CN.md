# TKG-DTN：无线网络实验

[English README](README.md)

本仓库提供 **TKG-DTN: Temporal Knowledge Graph-Driven Digital Twin Network for 6G Networking** 无线实验的源码、固定场景输入和运行说明。

本发布包以 **2026 年 10 月 19 日**的英文实验稿 *Framework Transfer and Dynamic Service Assurance in Wireless Networks*（无线网络中的框架迁移与动态服务保障）为依据，收录该稿最终使用的静态与动态实验。

TKG-DTN 通过图表示无线用户、网络切片、资源分配及带有时间信息的服务历史。LLM 智能体检索网络状态证据并调用资源分配工具，完成顺序接入控制或持续带宽调整。实验采用从 WirelessAgent 适配而来的香港科技大学（HKUST）Center、North 和 South 三个场景。

![Center、North 和 South 动态场景中的用户轨迹](figures/dynamic_mobility.png)

*三个 HKUST 场景中用户在 300 s 内的位置与移动轨迹。每个场景均包含静止用户、持续移动用户和间歇移动用户。*

## 实验内容

**静态顺序接入实验**比较 TKG-DTN、WirelessAgent、Prompt-based 和 Rule-based 在每场景 30 名用户依次到达时的表现。每种方法在三个场景中分别重复运行五次。

**动态服务保障实验**比较 TKG-DTN、WirelessAgent 和冻结静态分配在 300 s 用户移动与信道变化过程中的表现。每种智能体在每个场景中重复运行五次，确定性的冻结基线每场景运行一次。

共同设置：

- 总带宽为 120 MHz，划分为相互隔离的 90 MHz eMBB 切片和 30 MHz URLLC 切片。
- 论文模型为 **deepseek-v4-flash**，温度为 0.2。实验记录中的 API 请求别名为 `deepseek-chat`，响应中的模型标识为 `deepseek-flash`；二者是实验中同一模型的 API 标识。
- 沿用的工程速率映射为 `R = 10 B log10(1 + 10^(CQI/10))`，其中带宽单位为 MHz，速率单位为 Mbps。
- eMBB 每用户约束为 6–20 MHz、100–400 Mbps；URLLC 每用户约束为 1–5 MHz、1–100 Mbps。
- 五次重复使用相同输入与到达顺序。离散程度采用**重复运行之间的样本标准差**，而不是单次请求之间或重新采样网络之间的标准差。

静态实验中的两种智能体共享 `exact_resources_v2` 分配工具。Prompt-based 每个请求调用一次模型，不重新分配已有用户的带宽。Rule-based 使用真实服务类别标签，保留上游连续带宽的比例公平分配及回退机制；该方法不施加其他方法采用的整数带宽约束。

动态实验在 Center、North 和 South 中分别继承 27、30 和 29 名已接入用户，接入用户集合及其切片归属保持不变。每个场景观测全部 30 名用户，包括 10 名静止用户、10 名持续移动用户和 10 名间歇移动用户。输入包含从 0 到 300 s 的 301 个整秒采样时刻。1 秒粒度的位置由原始 10 秒轨迹段细分得到；信道根据几何模型重新计算，并以静态信道值为锚点校准。

两种智能体均采用完成驱动控制：每轮决策从最新可用状态开始，实际 API 推理期间继续使用上一轮资源分配。推理完成后，系统依据最新硬资源约束检查原始候选方案，通过检查后执行，并立即启动下一轮。超过 300 s 才完成的决策仍计入模型开销，但不会改变评估区间内的服务。本实验是使用真实 API 时延的离散事件仿真，并非物理无线测试床。

## 主要结果

静态实验中，TKG-DTN 与 WirelessAgent 在切片选择、带宽分配和服务性能上保持一致，同时减少了模型调用和 Token 消耗，并显著缩短决策时延。这表明该框架能够迁移到无线网络切片场景。

动态实验中，持续资源调整比冻结静态分配更能维持服务质量。相比 WirelessAgent，TKG-DTN 提高了有效服务水平和长期公平性，其中 URLLC 切片的公平性改善尤为明显。更短的控制轮次支持更频繁的资源调整，但也增加了持续运行期间的模型调用总量。

## 仓库结构

```text
static/                 静态实验入口、四种策略、资源工具及 Neo4j 适配器
dynamic/tkg/            TKG-DTN 动态实验入口及必需的运行模块
dynamic/wireless/       WirelessAgent/冻结基线入口及必需的运行模块
dynamic/data/           用于初始化动态实验的静态终态输入
data/mobility/          三个场景的一秒间隔信道与移动输入
preparation/            移动轨迹和信道生成代码及原始输入
figures/                动态场景示意图
requirements.txt        Python 依赖
.env.example            环境变量示例
```

静态 TKG-DTN 使用 **Neo4j**。动态 TKG-DTN 使用本地 **SQLite 时序图**，不需要 Neo4j 服务器。各动态方法保留各自的运行模块，以保持实验所用实现一致。`runtime_config.json` 提供实验采用的模型配置，以及运行加载器需要的源码校验值。

## 安装

使用 Python **3.11**，克隆仓库后安装依赖：

```bash
git clone https://github.com/garvey-chen/TKG-DTN-Wireless.git
cd TKG-DTN-Wireless
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。静态实验记录的依赖版本见 `static/requirements-lock.txt`。

## 重新运行静态实验

规则基线无需 API 密钥：

```bash
python static/run.py run --methods rule_upstream --scenarios center north south \
  --users 30 --repeats 5 --arrival-order source --output outputs/static_rule
```

运行完整的四方法实验时，先配置本地 Neo4j 数据库，再通过环境变量提供凭据：

```bash
cp static/configs/neo4j.example.json static/configs/local.neo4j.json
# 按需修改本地配置文件中的 URI、用户名和数据库名。
export WIRELESS_NEO4J_PASSWORD='YOUR_NEO4J_PASSWORD'
export WIRELESS_LLM_API_KEY='YOUR_API_KEY'

python static/run.py run --config static/configs/paper.json \
  --methods tkg_dtn wireless_kb rule_upstream prompt \
  --scenarios center north south --users 30 --repeats 5 --arrival-order source \
  --neo4j-config static/configs/local.neo4j.json --graph-reset none \
  --workers 3 --output outputs/static_new
```

PowerShell 对应的环境变量设置为 `$env:WIRELESS_LLM_API_KEY = 'YOUR_API_KEY'` 和 `$env:WIRELESS_NEO4J_PASSWORD = 'YOUR_NEO4J_PASSWORD'`。示例命令显式保留已有图数据（`--graph-reset none`），与选定实验批次一致。请使用专门的实验数据库。`.env.example` 列出了变量名，脚本不会自动加载该文件。

## 重新运行动态实验

根据方法选择对应的最终运行入口。以下命令均使用包内发布的静态衔接状态和相同的移动输入：

```bash
# 确定性基线：无需 API 凭据或 Neo4j。
python dynamic/wireless/run_completion_driven.py --method frozen --scene center \
  --output outputs/frozen_center

# 智能体：先设置 WIRELESS_LLM_API_KEY。
python dynamic/tkg/run_completion_driven.py --method tkg_dtn --scene center \
  --preflight-only
python dynamic/tkg/run_completion_driven.py --method tkg_dtn --scene center \
  --output outputs/tkg_center_r001
python dynamic/wireless/run_completion_driven.py --method wireless_agent --scene center \
  --output outputs/wireless_center_r001
```

对 `north`、`south` 同样执行智能体命令，每个场景重复五次并使用五个不同的输出目录。冻结基线每场景运行一次。动态程序拒绝写入已存在的输出目录。每次智能体运行都会进行真实、计费的模型调用。新运行的 API 时延和生成结果可能变化，因此不预期逐项精确复现所有历史数值。动态论文运行入口有意保留归档的 API 模型设置；使用不同后端或模型时，需要显式修改归档设置并记录变更。

公开复跑使用选定静态运行所提供的衔接状态。如果用新的静态结果替换该状态，就构成新的实验，需要重新生成并验证衔接数据。

## 重新生成移动输入

```bash
python preparation/generate_mobility_1s.py --output outputs/mobility_regenerated
```

该命令从包内原始 10 秒轨迹重建最终的 1 秒位置与锚定信道，保留用户身份、初始位置与请求，以及移动和暂停时序。生成的场景 CSV 可与 `data/mobility/{center,north,south}.csv` 对照；生成元数据会随输出路径及软件环境变化。该命令不会重新随机采样轨迹。

## 指标定义与适用范围

- **静态合格用户：**切片符合服务类别，且满足每用户和切片资源池约束。静态资源可行吞吐量包含资源可行但切片选择错误的用户。
- **动态有效用户时间比例：**每场景所有用户的有效服务时长之和除以 `30 × 300 s`。未接入用户仍计入分母。
- **动态吞吐量：**在精确的遥测与决策完成事件区间上积分模型速率，再除以 300 s。有效吞吐量还要求服务类别和资源约束均满足。
- **公平性：**在按服务类别划分的固定已接入用户群体内，将滑动窗口中的已交付流量除以最低保障流量，再计算 Jain 指数。先在每组运行内对窗口取平均，再在每次重复内对场景取平均，最后对重复运行取平均。切片选择错误的用户仍保留在固定群体中。
- **控制时延：**实测策略、模型和工具的墙钟时间，与无线分组时延不同。仿真将主机遥测回放和物理提交记账视为瞬时操作。
- **Token：**API 报告的输入与输出 Token 之和，包含缓存输入。该指标是 Token 数量，而不是货币价格。

这些实验评估框架适配和仿真服务保障，不测量分组级时延、无线硬件、切换或干扰。共享资源工具解释了观察到的静态决策一致性；效率差异比较的是完整的适配后智能体实现，不能单独归因于图表示。

## 来源与署名

WirelessAgent 代码和数据集改编自 [WirelessAgent](https://github.com/jwentong/WirelessAgent_R1)。地图保留 [OpenStreetMap 贡献者署名](https://www.openstreetmap.org/copyright)。来源和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

稿件使用的参考文献：

> J. Tong, W. Guo, J. Shao, Q. Wu, Z. Li, Z. Lin, and J. Zhang, “WirelessAgent: Large language model agents for intelligent wireless networks,” *China Communications*, vol. 23, no. 3, pp. 265–285, 2026. DOI: 10.23919/JCC.fa.2025-0163.202603.

使用这些实验时，请引用 TKG-DTN 稿件及上述 WirelessAgent 文献。
