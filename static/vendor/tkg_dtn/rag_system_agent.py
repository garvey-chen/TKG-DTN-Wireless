"""网络知识图谱 RAG 问答代理。

该模块基于 Neo4j 图谱与语言模型，实现网络运维领域的智能问答代理。
核心特性：
- ReAct 推理循环：LLM 按“思考→行动→观察”模式迭代调用工具；
- 图谱工具：支持实体检索、Cypher 查询、子图可视化；
- 流式输出：与 Web 前端对接时可采用 Server-Sent Events (SSE) 式按步骤推送。"""

from __future__ import annotations

import json
import logging
import argparse
import sys
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

from neo4j import GraphDatabase, Query, basic_auth, graph
from openai import OpenAI
from utils.config import config as global_config
from utils.topology_summarizer import TopologySummarizer


logger = logging.getLogger(__name__)


EXTRA_STRIP_CHARS = "`\"' “”‘’《》「」«»"


def _strip_markdown_quotes(text: str) -> str:
    """清理 LLM 输出中的多余引号或 Markdown 粗体符号。"""

    if not isinstance(text, str):
        return ""
    cleaned = text.strip().strip(EXTRA_STRIP_CHARS)
    cleaned = cleaned.replace("**", "")
    return cleaned


def _safe_to_string(value: object) -> str:
    """在 JSON 序列化前，对 Neo4j 属性值做安全格式化。"""

    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_safe_to_string(v) for v in value)
    if isinstance(value, (float, int)):
        return str(value)
    return str(value)


class NetworkRAGAgent:
    """动态网络知识图谱问答代理。"""

    def __init__(
        self,
        max_iterations: int = 5,
        memory_handler=None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.chat_history: List[Dict[str, str]] = []
        self.max_iterations = max_iterations
        self.memory_handler = memory_handler
        
        # Store provided overrides
        self.provided_model_name = model_name
        self.provided_api_key = api_key
        self.provided_base_url = base_url

        self.driver = None
        self.llm_client = None
        self.graph_schema: str = ""
        self.config = global_config

        # -----------------------------------------------------------
        # 配置加载逻辑：优先读取 llm_experimental 用于消融实验
        # -----------------------------------------------------------
        self.llm_config = self.config.get("llm", {}).copy() or {}
        experimental_config = self.config.get("llm_experimental", {})
        if experimental_config:
            logger.info(f"Using experimental LLM config: {experimental_config.get('model', 'unknown')}")
            self.llm_config.update(experimental_config)

        neo4j_cfg = self.config.get("neo4j", {}) or {}
        self.neo4j_uri = os.getenv("NEO4J_URI") or neo4j_cfg.get("uri", "neo4j://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER") or neo4j_cfg.get("username", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD") or neo4j_cfg.get("password", "[REMOVED_DEFAULT]")
        self.neo4j_database = os.getenv("NEO4J_DATABASE") or neo4j_cfg.get("database", "neo4j")

        self.llm_model = (
            self.provided_model_name
            or os.getenv("NETWORK_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or self.llm_config.get("model", "deepseek-chat")
        )
        cypher_timeout_env = os.getenv("NETWORK_CYPHER_TIMEOUT_SECONDS")
        if cypher_timeout_env and cypher_timeout_env.strip():
            self.cypher_timeout_seconds = float(cypher_timeout_env)
        else:
            self.cypher_timeout_seconds = float(
                self.config.get("neo4j", {}).get("query_timeout_seconds", 8.0)
            )
        temperature_env = os.getenv("NETWORK_LLM_TEMPERATURE")
        if temperature_env and temperature_env.strip():
            self.llm_temperature = float(temperature_env)
        else:
            self.llm_temperature = float(self.llm_config.get("temperature", 0.2))
        self._initialise()

    # ------------------------------------------------------------------
    # 初始化相关
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        """完成数据库与 LLM 初始化。"""

        try:
            self._init_llm()
            self._init_graph_driver()
            self._load_schema()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"初始化网络问答代理失败: {exc}") from exc

    def _init_llm(self) -> None:
        # 优先使用显式传入的参数或者已加载的配置，
        # 其次才是通用的环境变量，避免旧的 ENV 污染新的配置。
        api_key = (
            self.provided_api_key
            or self.llm_config.get("api_key")
            or os.getenv("NETWORK_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "请在环境变量或 config/config.yaml 的 llm.api_key 中配置大模型 API Key。"
            )

        base_url = (
            self.provided_base_url
            or self.llm_config.get("base_url")
            or os.getenv("NETWORK_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
        )

        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_base_url = base_url

    def _init_graph_driver(self) -> None:
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=basic_auth(self.neo4j_user, self.neo4j_password),
        )
        self.driver.verify_connectivity()
        logger.info("Neo4j 连接成功: %s", self.neo4j_uri)

    def _load_schema(self) -> None:
        """从 Neo4j 查询并缓存节点、关系 Schema 摘要。"""

        query_node = (
            "CALL db.schema.nodeTypeProperties() "
            "YIELD nodeLabels, propertyName "
            "RETURN nodeLabels[0] AS label, collect(DISTINCT propertyName) AS props"
        )
        query_rel = (
            "MATCH (s)-[r]->(t) "
            "RETURN DISTINCT labels(s) AS source, type(r) AS rel, labels(t) AS target "
            "LIMIT 200"
        )
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                node_records = session.run(query_node).data()
                rel_records = session.run(query_rel).data()

            node_segments = [
                f"{rec['label']}({', '.join(sorted(p for p in rec['props'] if p != 'embedding'))})"
                for rec in node_records
                if rec.get("label")
            ]
            relation_segments = [
                f"(:{rec['source']})-[:{rec['rel']}]->(:{rec['target']})"
                for rec in rel_records
                if rec.get("source") and rec.get("target")
            ]
            self.graph_schema = (
                "节点标签与属性: " + "; ".join(node_segments)
                + "\n关系结构: "
                + ", ".join(relation_segments)
            )
        except Exception as exc:
            logger.warning("获取图谱 Schema 失败: %s", exc)
            self.graph_schema = "(未能获取图谱 Schema，可继续进行问答)"

    

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
        self.driver = None

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------
    def _format_subgraph_for_vis(self, query_result_records: Iterable[graph.Record]) -> Dict[str, object]:
        nodes: Dict[str, Dict[str, object]] = {}
        edges: set = set()

        def add_node(node: graph.Node) -> None:
            node_id = node.element_id
            if node_id in nodes:
                return
            label = next(iter(node.labels)) if node.labels else "Node"
            properties = dict(node)
            title_lines = [
                f"{k}: {_safe_to_string(v)}"
                for k, v in properties.items()
                if k != "embedding" and v is not None
            ]
            display_name = (
                properties.get("name")
                or properties.get("id")
                or properties.get("mac")
                or properties.get("mac_address")
                or properties.get("ip")
                or properties.get("ip_address")
                or properties.get("title")
                or "…"
            )
            nodes[node_id] = {
                "id": node_id,
                "label": str(display_name)[:36],
                "group": label,
                "title": "\n".join(title_lines),
            }

        def add_edge(edge: graph.Relationship) -> None:
            add_node(edge.start_node)
            add_node(edge.end_node)
            edges.add((edge.start_node.element_id, edge.end_node.element_id, edge.type))

        for record in query_result_records:
            for value in record.values():
                if isinstance(value, graph.Path):
                    for node in value.nodes:
                        add_node(node)
                    for rel in value.relationships:
                        add_edge(rel)
                elif isinstance(value, graph.Node):
                    add_node(value)
                elif isinstance(value, graph.Relationship):
                    add_edge(value)

        return {
            "nodes": list(nodes.values()),
            "edges": [
                {"from": start, "to": end, "label": rel, "arrows": "to"}
                for start, end, rel in edges
            ],
        }

    ##新版本节点搜索工具:
    def _search_entities(self, keyword: Union[str, List[str]], limit: int = 15) -> Dict[str, object]:
        # 标准化输入为列表
        if isinstance(keyword, str):
            keywords = [keyword]
        elif isinstance(keyword, list):
            keywords = [str(k) for k in keyword]
        else:
            return {"error": "keyword 参数必须是字符串或字符串列表"}

        # 1. 修正 Cypher 语法错误：移除 toString(n)，改用 elementId(n) 或 id(n)
        # 2. 优化搜索逻辑：增加对 list 类型的容错处理
        # 兼容性修改：因用户 Neo4j 版本较低，使用 toString(id(n)) 替代 elementId(n)
        # 支持多关键词批量匹配
        cypher = (
            "MATCH (n) "
            "WHERE ANY(kw IN $kws WHERE "
            "  ANY(k IN keys(n) WHERE toString(n[k]) CONTAINS kw)"
            ") "
            "RETURN DISTINCT labels(n)[0] AS label, "
            "coalesce(n.name, n.id, n.mac, n.mac_address, n.ip, n.ip_address, n.title, toString(id(n))) AS identifier, "
            "properties(n) AS props "
            "LIMIT $limit"
        )
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                records = session.run(cypher, kws=keywords, limit=limit).data()
        except Exception as exc:
            return {"error": f"实体检索失败: {exc}"}

        results = []
        for record in records:
            props = record.get("props", {}) or {}
            
            # --- 核心修改：移除激进的属性过滤逻辑 ---
            # 仅仅移除 embedding 这种长向量，其他属性全部保留。
            # 让 LLM 自己决定哪些属性有用，而不是我们在代码里硬编码。
            clean_props = {
                k: self._safe_to_string(v) 
                for k, v in props.items() 
                if k != "embedding" and v is not None
            }
            
            results.append(
                {
                    "label": record.get("label"),
                    "identifier": record.get("identifier"),
                    "properties": clean_props,  # 返回完整属性
                }
            )

        # 3. 增强返回结果的可读性
        # 如果没有结果，明确告诉模型，防止它产生幻觉
        if not results:
            msg_kw = str(keywords) if len(keywords) > 1 else keywords[0]
            return {"results": [], "count": 0, "message": f"未找到包含关键字 '{msg_kw}' 的实体"}

        return {"results": results, "count": len(results)}

    # 辅助函数（如果尚未定义）
    def _safe_to_string(self, value):
        try:
            return str(value)
        except:
            return ""

    # def _search_entities(self, keyword: str, limit: int = 8) -> Dict[str, object]:
    #     cypher = (
    #         "MATCH (n) "
    #         "WHERE ANY(k IN keys(n) WHERE toString(n[k]) CONTAINS $kw) "
    #         "RETURN DISTINCT labels(n)[0] AS label, "
    #         "coalesce(n.name, n.id, n.mac, n.mac_address, n.ip, n.ip_address, n.title, toString(n)) AS identifier, "
    #         "properties(n) AS props "
    #         "LIMIT $limit"
    #     )
    #     try:
    #         with self.driver.session(database=self.neo4j_database) as session:
    #             records = session.run(cypher, kw=keyword, limit=limit).data()
    #     except Exception as exc:
    #         return {"error": f"实体检索失败: {exc}"}

    #     results = []
    #     for record in records:
    #         props = record.get("props", {}) or {}
    #         top_keys = [
    #             key
    #             for key in (
    #                 "state",
    #                 "management_ip",
    #                 "ip",
    #                 "ip_address",
    #                 "mac",
    #                 "mac_address",
    #                 "datapath_id",
    #                 "event_type",
    #                 "severity",
    #                 "location",
    #             )
    #             if key in props and props[key] not in (None, "")
    #         ]
    #         if not top_keys:
    #             top_keys = [key for key in props.keys() if key != "embedding"][:4]
    #         summary = {key: _safe_to_string(props[key]) for key in top_keys}
    #         results.append(
    #             {
    #                 "label": record.get("label"),
    #                 "identifier": record.get("identifier"),
    #                 "summary": summary,
    #             }
    #         )
    #     return {"results": results, "count": len(results)}



    ##新版本 Cypher 查询工具:
    def _execute_cypher(self, query: str) -> Dict[str, object]:
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result_cursor = session.run(
                    Query(query, timeout=self.cypher_timeout_seconds)
                )
                records = list(result_cursor)
        except Exception as exc:
            error_text = str(exc)
            if "timed out" in error_text.lower() or "timeout" in error_text.lower():
                error_text = (
                    f"Cypher 查询超时（>{self.cypher_timeout_seconds:.1f}s）: {exc}"
                )
            else:
                error_text = f"Cypher 查询执行失败: {exc}"
            return {"error": error_text, "query": query}

        # 假设这个函数你是有的且没问题
        subgraph = self._format_subgraph_for_vis(records)
        
        simplified_records: List[Dict[str, object]] = []

        # ❌ 移除 allowed_props 白名单，改用黑名单逻辑（排除 embedding）

        for record in records[:50]:
            simplified_record: Dict[str, object] = {}
            for key, value in record.items():
                # 1. 处理路径 (Path)
                if isinstance(value, graph.Path):
                    simplified_record[key] = {
                        "nodes": [
                            {
                                "labels": list(node.labels),
                                "properties": {
                                    k: _safe_to_string(v)
                                    for k, v in dict(node).items()
                                    if k != "embedding" and v is not None # 仅过滤向量
                                },
                            }
                            for node in value.nodes
                        ],
                        "relationships": [
                            {
                                "type": rel.type,
                                "start": rel.start_node.element_id,
                                "end": rel.end_node.element_id,
                                "properties": { # ✅ 增加关系属性提取
                                    k: _safe_to_string(v)
                                    for k, v in dict(rel).items()
                                    if k != "embedding" and v is not None
                                }
                            }
                            for rel in value.relationships
                        ],
                    }
                # 2. 处理节点 (Node)
                elif isinstance(value, graph.Node):
                    simplified_record[key] = {
                        "labels": list(value.labels),
                        "properties": {
                            k: _safe_to_string(v)
                            for k, v in dict(value).items()
                            if k != "embedding" and v is not None # ✅ 允许返回 manufacturer, hardware 等
                        },
                    }
                # 3. 处理关系 (Relationship)
                elif isinstance(value, graph.Relationship):
                    simplified_record[key] = {
                        "type": value.type,
                        "start": value.start_node.element_id,
                        "end": value.end_node.element_id,
                        "properties": { # ✅ 增加关系属性提取
                            k: _safe_to_string(v)
                            for k, v in dict(value).items()
                            if k != "embedding" and v is not None
                        }
                    }
                # 4. 处理基础类型 (String, Int, List...)
                else:
                    simplified_record[key] = _safe_to_string(value)
                    
            simplified_records.append(simplified_record)

        return {
            "count": len(records),
            "results": simplified_records,
            "subgraph": subgraph,
        }
    # def _execute_cypher(self, query: str) -> Dict[str, object]:
    #     try:
    #         with self.driver.session(database=self.neo4j_database) as session:
    #             result_cursor = session.run(query)
    #             records = list(result_cursor)
    #     except Exception as exc:
    #         return {"error": f"Cypher 查询执行失败: {exc}", "query": query}

    #     subgraph = self._format_subgraph_for_vis(records)
    #     simplified_records: List[Dict[str, object]] = []
    #     allowed_props = {
    #         "id",
    #         "name",
    #         "title",
    #         "state",
    #         "ip",
    #         "ip_address",
    #         "mac",
    #         "mac_address",
    #         "datapath_id",
    #         "priority",
    #         "event_type",
    #         "severity",
    #         "byte_count",
    #         "packet_count",
    #         "bandwidth",
    #         "latency",
    #         "loss_rate",
    #         "utilization",
    #     }

    #     for record in records[:50]:
    #         simplified_record: Dict[str, object] = {}
    #         for key, value in record.items():
    #             if isinstance(value, graph.Path):
    #                 simplified_record[key] = {
    #                     "nodes": [
    #                         {
    #                             "labels": list(node.labels),
    #                             "properties": {
    #                                 prop: _safe_to_string(val)
    #                                 for prop, val in dict(node).items()
    #                                 if prop in allowed_props and val is not None
    #                             },
    #                         }
    #                         for node in value.nodes
    #                     ],
    #                     "relationships": [
    #                         {
    #                             "type": rel.type,
    #                             "start": rel.start_node.element_id,
    #                             "end": rel.end_node.element_id,
    #                         }
    #                         for rel in value.relationships
    #                     ],
    #                 }
    #             elif isinstance(value, graph.Node):
    #                 simplified_record[key] = {
    #                     "labels": list(value.labels),
    #                     "properties": {
    #                         prop: _safe_to_string(val)
    #                         for prop, val in dict(value).items()
    #                         if prop in allowed_props and val is not None
    #                     },
    #                 }
    #             elif isinstance(value, graph.Relationship):
    #                 simplified_record[key] = {
    #                     "type": value.type,
    #                     "start": value.start_node.element_id,
    #                     "end": value.end_node.element_id,
    #                 }
    #             else:
    #                 simplified_record[key] = _safe_to_string(value)
    #         simplified_records.append(simplified_record)

    #     return {
    #         "count": len(records),
    #         "results": simplified_records,
    #         "subgraph": subgraph,
    #     }
    
    def _Network_topology_query(self) -> Dict[str, object]:
        # 复用通用执行逻辑，且返回路径 p 以便构建可视化子图
        query = (
            "MATCH (s:交换机)\n"
            "OPTIONAL MATCH (c:控制器)-[:控制]->(s)\n"
            "OPTIONAL MATCH (p:端口)-[:归属于]->(s)\n"
            "OPTIONAL MATCH (p)-[:连接到]->(remote_p:端口)\n"
            "OPTIONAL MATCH (h:主机)-[:连接到]->(p)\n"
            "WITH s, c, p, remote_p, h\n"
            "ORDER BY toInteger(p.number)\n"
            "WITH s, c,\n"
            "collect(\n"
            "    CASE\n"
            "        WHEN remote_p IS NOT NULL THEN {\n"
            "            local_port: p.number,\n"
            "            connects_to: \"Switch\",\n"
            "            target_id: split(remote_p.id, '/')[0],\n"
            "            target_port: remote_p.number\n"
            "        }\n"
            "        WHEN h IS NOT NULL THEN {\n"
            "            local_port: p.number,\n"
            "            connects_to: \"Host\",\n"
            "            ip: h.ip_address,\n"
            "            mac: h.mac_address,\n"
            "            location: h.location\n"
            "        }\n"
            "        ELSE null\n"
            "    END\n"
            ") AS switch_connections\n"
            "RETURN {\n"
            "    device: \"Switch\",\n"
            "    id: s.id,\n"
            "    controller: c.ip_address,\n"
            "    connections: switch_connections\n"
            "} AS TopologyContext"
        )
        # query = (
        #     "MATCH (n)-[r]-(m)\n"
        #     "WHERE (n:交换机 OR n:主机 OR n:控制器) \n"
        #     "  AND (m:交换机 OR m:主机 OR m:控制器)\n"
        #     "  AND n.id IS NOT NULL AND m.id IS NOT NULL\n"
        #     "WITH n, r, m\n"
        #     "WHERE id(n) < id(m) OR type(r) = '控制'\n"
        #     "RETURN \n"
        #     "  n.id as source_id,\n"
        #     "  labels(n) as source_labels,\n"
        #     "  type(r) as relationship_type, \n"
        #     "  m.id as target_id,\n"
        #     "  labels(m) as target_labels\n"
        #     "LIMIT 1000"
        # )
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                topology = session.run(query).data()
            
            # 提取 TopologyContext 列表
            raw_devices = [item['TopologyContext'] for item in topology if 'TopologyContext' in item]
            
            # 使用 TopologySummarizer 生成摘要
            summarizer = TopologySummarizer(raw_devices)
            summary_text = summarizer.summarize()
            
        except Exception as exc:
            return {"error": f"网络拓扑查询执行失败: {exc}", "query": query}
            
        # 返回摘要而非原始数据，以节省 LLM 上下文
        return {"summary": summary_text}

    def _find_shortest_path(self, source_id: str, target_id: str) -> Dict[str, object]:
        """查询两点间基于物理连接(连接到/归属于)的最短路径。"""
        query = (
            "MATCH (s) WHERE s.id = $source_id "
            "WITH collect(s) as sources "
            "MATCH (t) WHERE t.id = $target_id "
            "WITH sources, collect(t) as targets "
            "UNWIND sources as s "
            "UNWIND targets as t "
            "MATCH p = shortestPath((s)-[:连接到|归属于*..15]-(t)) "
            "WHERE id(s) <> id(t) "
            "RETURN p"
        )
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # 显式转换 query result 为 list，确保数据被拉取到内存
                # 如果 run 之后没有 .data() 或 list()，可能返回 lazy iterator
                result = session.run(query, source_id=source_id, target_id=target_id)
                records = list(result)
        except Exception as exc:
            logger.error(f"Path finding error: {exc}")
            return {"error": f"路径查询失败: {exc}", "query": query}

        # 复用通用的子图格式化逻辑
        subgraph = self._format_subgraph_for_vis(records)
        
        # 提取简化的路径信息
        paths = []
        for record in records:
            p = record["p"]
            path_info = {
                "length": len(p.relationships),
                "nodes": [n.get("id", n.element_id) for n in p.nodes],
                "relationships": [r.type for r in p.relationships]
            }
            paths.append(path_info)
            
        return {
            "count": len(paths),
            "paths": paths,
            "subgraph": subgraph
        }

    def _get_dynamic_examples(self) -> str:
        """
        获取具有代表性的网络图谱关系示例，优先展示拓扑和控制面关系，
        及数据观测对象关联，帮助 LLM 理解图谱结构。
        """
        examples = []
        # 定义关键路径查询模板，确保覆盖：主机接入、链路连接、设备归属、控制关系、监控关联
        queries = [
            # 1. 主机接入: 展示主机 IP/MAC 如何关联到交换机端口
            (
                "MATCH (h:`主机`)-[r:`连接到`]->(p:`端口`) "
                "RETURN labels(h)[0] as s_label, coalesce(h.ip_address, h.mac_address, 'UnknownHost') as s_name, "
                "type(r) as rel, labels(p)[0] as t_label, p.name as t_name LIMIT 1"
            ),
            # 2. 端口链路连接: 展示核心网络路径
            (
                "MATCH (p1:`端口`)-[r:`连接到`]->(p2:`端口`) "
                "RETURN labels(p1)[0] as s_label, p1.name as s_name, type(r) as rel, "
                "labels(p2)[0] as t_label, p2.name as t_name LIMIT 1"
            ),
            # 3. 端口归属交换机: 展示端口与设备的从属关系
            (
                "MATCH (p:`端口`)-[r:`归属于`]->(s:`交换机`) "
                "RETURN labels(p)[0] as s_label, p.name as s_name, type(r) as rel, "
                "labels(s)[0] as t_label, s.id as t_name LIMIT 1"
            ),
            # 4. 控制器控制交换机: 展示控制平面视角
            (
                "MATCH (c:`控制器`)-[r:`控制`]->(s:`交换机`) "
                "RETURN labels(c)[0] as s_label, coalesce(c.ip_address, 'Controller') as s_name, type(r) as rel, "
                "labels(s)[0] as t_label, s.id as t_name LIMIT 1"
            ),
             # 5. 观测数据关联: 展示 Metric/Event 如何关联到物理设施
            (
                "MATCH (o:Observation)-[r:ABOUT]->(t) "
                "RETURN labels(o)[0] as s_label, 'Obs-Metric' as s_name, type(r) as rel, "
                "labels(t)[0] as t_label, coalesce(t.name, t.id, labels(t)[0]) as t_name LIMIT 1"
            )
        ]

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                for q in queries:
                    try:
                        result = session.run(q).data()
                        if result:
                            row = result[0]
                            # 过滤并格式化，确保 s_name/t_name 不为空
                            s_name = row.get('s_name') or 'Unknown'
                            t_name = row.get('t_name') or 'Unknown'
                            examples.append(f"{row['s_label']} {s_name} --[{row['rel']}]--> {row['t_label']} {t_name}")
                    except Exception:
                        # 忽略单个查询失败（例如该类型关系暂不存在）
                        continue
        except Exception as exc:
            logger.warning("获取动态图谱示例失败: %s", exc)
            return "未能获取示例。"

        # 如果没有获取到特定关系（可能是空库），回退到通用查询
        if not examples:
            fallback_query = (
                "MATCH (s)-[r]->(t) "
                "WHERE NOT s:Time AND NOT t:Time "  # 排除时间节点干扰
                "RETURN labels(s)[0] AS s_label, coalesce(s.name, s.id) AS s_name, "
                "type(r) AS rel, labels(t)[0] AS t_label, coalesce(t.name, t.id) AS t_name "
                "LIMIT 2"
            )
            try:
                with self.driver.session(database=self.neo4j_database) as session:
                    rows = session.run(fallback_query).data()
                    examples = [
                        f"{row['s_label']} {row.get('s_name','?')} --[{row['rel']}]--> {row['t_label']} {row.get('t_name','?')}"
                        for row in rows
                    ]
            except Exception:
                pass

        if not examples:
            return "暂无图谱数据示例。"

        return "\n".join([f"示例{i+1}: {ex}" for i, ex in enumerate(examples)])


    # ------------------------------------------------------------------
    # LLM 循环
    # ------------------------------------------------------------------
    def _decide_next_action(self, question: str, scratchpad: str) -> str:
        dynamic_examples = self._get_dynamic_examples()
        
        # 获取记忆上下文（如果配置了 Memory Handler）
        memory_context = ""
        if self.memory_handler and hasattr(self.memory_handler, "retrieve"):
            try:
                memory_context = self.memory_handler.retrieve(question)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")

        prompt = f"""
You are an intelligent QA agent for SDN/Network Operations, interacting with a Knowledge Graph via tools.
Please follow the ReAct pattern: "Thought -> Action -> Observation".

【Tool List】
1. search_entities(keywords: Union[str, List[str]])
   - Purpose: Fuzzy search for nodes to view identifiers and key properties. Supports single keyword or list.
   - Example (Single): {{"tool": "search_entities", "args": {{"keyword": "of:0000000000000001"}}}}
   - Example (Batch): {{"tool": "search_entities", "args": {{"keyword": ["h1", "h2", "of:0001"]}}}}
2. execute_cypher(query: str)
   - 用途: 执行 Cypher 查询，获取网络结构或状态数据。
   - 示例: {{"tool": "execute_cypher", "args": {{"query": "MATCH p=(d:交换机 {{id:'of:0000000000000001'}})-[*1..2]-(m) RETURN p LIMIT 50"}}}}
3. Network_topology_query()
   - 用途: 执行预设好的cypher查询，获取网络拓扑相关信息。
   - 示例: {{"tool": "Network_topology_query", "args": {{}}}}
4. find_shortest_path(source_id: str, target_id: str)
   - 用途: 查询两个设备/主机间基于物理连接的最短路径(避开控制链路)。
   - 示例: {{"tool": "find_shortest_path", "args": {{"source_id": "of:0000000000000001", "target_id": "of:0000000000000002"}}}}
5. finish()
   - 用途: 已获取足够信息，生成最终回答。

【操作原则】
- 若涉及到网络物理结构，可以优先使用 Network_topology_query 获取最新拓扑，可能可以辅助回答问题。
- 若用户问题包含明确的节点 ID/名称，可以使用 search_entities 或直接构造查询验证该实体的上下文。
- execute_cypher 时应给出 LIMIT，根据问题的复杂性，推荐 50 以内，避免全量扫描。
- **关键：Cypher 关系查询必须注意方向性**。网络图谱中的关系是有向的（例如 `Host->Port`, `Controller->Switch`）。如果你不确定方向，**必须使用无方向查询** `(template (a)-[r]-(b))`，这样能同时匹配 `(a)-[r]->(b)` 和 `(b)-[r]->(a)`，否则极易导致查不到数据！
- 避免重复查询或在缺乏上下文时盲目 finish。
- 特别注意严格按照以下输出格式进行，输出格式固定为：
  思考: <你的推理>
  行动: {{"tool": "...", "args": {{...}}}}

【图谱 Schema 摘要】
{self.graph_schema}

【记忆与上下文】
{memory_context if memory_context else '（无）'}

【随机关系示例】
{dynamic_examples}

【历史轨迹】
{scratchpad if scratchpad else '（无）'}

【当前问题】{question}

按上述格式继续下一步。"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "你是网络知识图谱的专业分析助手，严格遵循工具使用规范。",
                },
                {"role": "user", "content": prompt},
            ],
            stop=["观察:"],
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # 回答组装
    # ------------------------------------------------------------------
    def _generate_final_answer(self, question: str, thought_trace: str) -> str:
        observations: List[str] = []
        for line in thought_trace.splitlines():
            if line.startswith("观察: "):
                observations.append(line.replace("观察: ", "", 1))

        # prompt = (
        #     "你是网络运维知识图谱助手，请整合以下工具观察结果，构造清晰、可验证的回答。"
        #     "\n要求:"
        #     "\n1. 用中文解释，不超过 1000 字。"
        #     "\n2. 用分点形式突出关键发现 (使用 \"-\")。"
        #     "\n3. 若存在事件/状态，需要包含时间或状态字段。"
        #     "\n4. 若数据不足，请明确说明。"
        #     f"\n\n【问题】{question}\n【观察数据】{json.dumps(observations, ensure_ascii=False)}"
        # )

        prompt = (
            "作为网络运维助手，请基于提供的观察数据回答问题。"
            "\n回答准则:"
            "\n- 【结论先行】：在回答开头用一句话直接给出最终结论或数值。"
            "\n- 【数据溯源】：引用数据时，必须注明来自哪个设备、端口或观测时间点。"
            "\n- 【逻辑严密】：如果观察结果中存在报错（SyntaxError）或查询为空，请诚实解释该情况如何影响了结论。"
            "\n- 【逻辑推理】：如果具体数据获取失败，可以针对当前已有知识结合网络运维常识进行合理推断，但必须明确标注“推断”二字。"
            f"\n\n【用户问题】：{question}"
            f"\n【原始观察数据】：{json.dumps(observations, ensure_ascii=False)}"
            "\n\n请输出结构化答案，字数不超过1000字："
        )
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0.2,
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": "你是SDN网络知识图谱问答助手，仅根据提供的数据作答，禁止猜测。",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # 对外主流程 (SSE 风格生成器)
    # ------------------------------------------------------------------
    def ask_stream(self, question: str):
        scratchpad = ""
        full_trace = ""
        latest_subgraph: Dict[str, object] = {}

        for iteration in range(self.max_iterations):
            yield self._format_event("status", f"--- 第 {iteration + 1} 轮思考 ---")
            llm_output = self._decide_next_action(question, scratchpad)

            try:
                thought, action = self._parse_llm_action(llm_output)
            except ValueError as exc:
                error_msg = f"解析 LLM 响应失败: {exc}. 原始响应: {llm_output}"
                yield self._format_event("error", error_msg)
                full_trace += f"\n思考: {llm_output}\n观察: {error_msg}"
                break

            yield self._format_event("status", f"思考: {thought}")

            if action.get("tool") == "finish":
                full_trace += f"\n思考: {thought}\n行动: {json.dumps(action, ensure_ascii=False)}"
                break

            tool_result, observation_summary = self._run_tool(action)
            full_trace += (
                f"\n思考: {thought}\n行动: {json.dumps(action, ensure_ascii=False)}"
                f"\n观察: {json.dumps(tool_result, ensure_ascii=False)}"
            )
            scratchpad += (
                f"\n思考: {thought}\n行动: {json.dumps(action, ensure_ascii=False)}"
                f"\n观察: {observation_summary}"
            )
            yield self._format_event("status", f"观察: {observation_summary}")

            if (
                action.get("tool") in ("execute_cypher", "Network_topology_query")
                and isinstance(tool_result, dict)
                and tool_result.get("subgraph", {}).get("nodes")
            ):
                latest_subgraph = tool_result.get("subgraph", latest_subgraph)

        yield self._format_event("status", "✅ 思考完成，正在生成最终答案...")
        final_answer = self._generate_final_answer(question, full_trace)
        self.chat_history.append({"question": question, "answer": final_answer})
        yield self._format_event("final_answer", final_answer)

        # 保存记忆（如果配置了 Memory Handler）
        if self.memory_handler and hasattr(self.memory_handler, "save"):
            try:
                self.memory_handler.save(question=question, answer=final_answer, trace=full_trace)
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

        # if latest_subgraph:
        #     yield self._format_event("subgraph", latest_subgraph)

        yield self._format_event("end", None)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _format_event(self, event_type: str, data: Optional[object]) -> str:
        return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"

    def _parse_llm_action(self, llm_output: str) -> Tuple[str, Dict[str, object]]:
        # 兼容中文冒号，避免解析错误
        llm_output = llm_output.replace("行动：", "行动:").replace("思考：", "思考:")
        
        if "行动:" not in llm_output:
            raise ValueError("缺少 '行动:' 关键字")
        thought = llm_output.split("行动:", 1)[0].replace("思考:", "").strip()
        action_text = llm_output.split("行动:", 1)[1].strip()
        match = re.search(r"\{.*\}", action_text, re.DOTALL)
        if not match:
            raise ValueError("行动部分未解析出合法 JSON")
        action = json.loads(match.group())
        return thought, action

    def _run_tool(self, action: Dict[str, object]) -> Tuple[Dict[str, object], str]:
        tool = action.get("tool")
        args = action.get("args", {}) or {}

        if tool == "search_entities":
            # 支持 keyword 为 string 或 list
            raw_kw = args.get("keyword")
            if isinstance(raw_kw, list):
                kw_arg = [_strip_markdown_quotes(str(k)) for k in raw_kw]
            else:
                kw_arg = _strip_markdown_quotes(str(raw_kw)) if raw_kw else ""
            
            result = self._search_entities(kw_arg)
            summary = (
                f"找到 {result.get('count', 0)} 个候选实体"
                if "error" not in result
                else result["error"]
            )
            return result, summary

        if tool == "execute_cypher":
            query = args.get("query", "")
            result = self._execute_cypher(query)
            if "error" in result:
                summary = f"查询失败: {result['error']}"
            else:
                summary = f"查询成功，返回 {result.get('count', 0)} 条记录"
            return result, summary

        if tool == "Network_topology_query":
            result = self._Network_topology_query()
            if "error" in result:
                summary = f"查询失败: {result['error']}"
            else:
                # 优先使用生成的摘要
                summary = result.get("summary", f"查询成功，返回 {result.get('count', 0)} 条记录")
            return result, summary

        if tool == "get_network_info":
            result = self._get_network_info()
            summary = result["info"]
            return result, summary

        if tool == "find_shortest_path":
            s_id = args.get("source_id")
            t_id = args.get("target_id")
            if not s_id or not t_id:
                return {"error": "缺参数"}, "调用 find_shortest_path 需要 source_id 和 target_id"
            result = self._find_shortest_path(s_id, t_id)
            if "error" in result:
                summary = result["error"]
            elif result["count"] == 0:
                summary = f"未找到从 {s_id} 到 {t_id} 的物理路径"
            else:
                steps = result["paths"][0]["length"]
                summary = f"找到路径: {steps} 跳, 详见 subgraph"
            return result, summary

        result = {"error": f"未知的工具: {tool}"}
        return result, result["error"]


__all__ = ["NetworkRAGAgent"]
# ------------------------------------------------------------------
# CLI 运行逻辑 (内置测试支持)
# ------------------------------------------------------------------

EXAMPLE_QUESTIONS: Iterable[str] = (
    "当前链路 of:0000000000000001/3 的状态是什么？",
    "交换机 of:0000000000000002 下挂主机的 IP 和 MAC 信息？",
    "最近是否有严重级别的网络告警？",
    "控制器中带宽利用率最高的链路是哪一条？",
)

def print_example_questions() -> None:
    print("\n💡 示例问题（可直接复制）：")
    for idx, item in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {idx}. {item}")

def run_agent_interactive(agent: NetworkRAGAgent, question: str) -> Dict[str, object]:
    final_payload: Dict[str, object] = {
        "answer": None,
        "subgraph": None,
        "trace": [],
    }

    for event in agent.ask_stream(question):
        payload = event.removeprefix("data: ").strip()
        if not payload:
            continue
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            continue
            
        event_type = message.get("type")
        data = message.get("data")

        if event_type == "status":
            print(f"[状态] {data}")
            final_payload["trace"].append({"status": data})
        elif event_type == "error":
            print(f"[错误] {data}", file=sys.stderr)
            final_payload["trace"].append({"error": data})
        elif event_type == "final_answer":
            print("\n=== 最终回答 ===")
            print(data)
            final_payload["answer"] = data
        elif event_type == "subgraph":
            final_payload["subgraph"] = data
        elif event_type == "end":
            break

    return final_payload

def main() -> None:
    parser = argparse.ArgumentParser(description="网络知识图谱 RAG 问答 (内置测试模式)")
    parser.add_argument("--question", help="直接查询的问题，设置后执行一次问答后退出。")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="仅展示示例问题，不发起真实查询。",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="启用 Mem0 记忆功能 (需要 network_rag_demo.py 提供接口)",
    )
    args = parser.parse_args()

    if args.examples:
        print_example_questions()
        return

    # 记忆模块集成
    memory_handler = None
    if args.memory:
        try:
            from Mem0Handler import get_mem0_handler
            memory_handler = get_mem0_handler()
            print("✅ 已启用 Mem0 记忆模块")
        except ImportError:
            print("❌ 无法导入 network_rag_demo 模块，确保该文件存在。", file=sys.stderr)
        except Exception as e:
            print(f"❌ 初始化记忆模块失败: {e}", file=sys.stderr)

    # 实例化 Agent
    agent = NetworkRAGAgent(memory_handler=memory_handler)
    
    try:
        if args.question:
            run_agent_interactive(agent, args.question.strip())
            return

        print_example_questions()
        print("\n请输入网络问题（直接回车退出，Ctrl+C 亦可，输入 'memories' 查看记忆）：")
        while True:
            try:
                question = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break

            if not question:
                print("已退出。")
                break

            if question.lower() == "memories":
                if memory_handler and hasattr(memory_handler, "get_all_memories"):
                    memory_handler.get_all_memories()
                else:
                    print("记忆模块未启用。")
                continue

            try:
                run_agent_interactive(agent, question)
            except Exception as exc: 
                print(f"[异常] {exc}", file=sys.stderr)
    finally:
        if memory_handler and hasattr(memory_handler, "cleanup_session"):
            memory_handler.cleanup_session()
        agent.close()

if __name__ == "__main__":
    main()
