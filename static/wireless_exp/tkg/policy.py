"""Policy interface joining wireless simulation, temporal mapping and copied agent."""

import json
from copy import deepcopy

from .connection import GraphConnection
from .graph_store import WirelessGraph
from .rag_system_agent import NetworkRAGAgent
from .tools import WirelessTools


class TKGPolicy:
    execution_optimizations = ["combined_context_and_revision_cache",
                              "compact_model_evidence", "prefetch_initial_context"]
    intent_knowledge_context = "complete_every_model_round"

    def __init__(self, llm, neo4j_config, output, max_iterations=8):
        self.llm = llm
        self.output = output
        self.connection = GraphConnection(neo4j_config)
        self.execution_optimizations = list(self.execution_optimizations)
        if self.connection.transport == "http":
            self.execution_optimizations.insert(0, "persistent_direct_http")
        self.connection.log_path = output / "cypher_calls.jsonl"
        try:
            if not self.connection.verify():
                raise RuntimeError("Neo4j verification failed")
            self.graph = WirelessGraph(self.connection, output / "graph")
            self.graph.initialize()
            self.tools = WirelessTools(self.graph, llm, output)
            self.agent = NetworkRAGAgent(llm, self.tools, max_iterations=max_iterations)
            self.arrivals = 0
        except Exception:
            self.connection.close()
            raise

    def step(self, observation):
        self.arrivals += 1
        self.llm.user_id = observation["user_id"]
        self.graph.arrive(observation, self.arrivals)
        self.tools.start(observation["user_id"])
        question = f"Handle the newly arrived user {observation['user_id']} using the prefetched graph context. Decide the service slice, and allocate or reject."
        with (self.output / "agent_events.jsonl").open("a", encoding="utf-8") as handle:
            for message in self.agent.ask_stream(question):
                event = json.loads(message.removeprefix("data: ").strip())
                handle.write(json.dumps({"user_id": observation["user_id"], **event}, ensure_ascii=False) + "\n")
        if self.tools.result is None:
            raise RuntimeError("No terminal network decision")
        return deepcopy(self.tools.result)

    def snapshot(self):
        return deepcopy(self.graph.state)

    def close(self):
        try:
            self.graph.export()
        finally:
            self.connection.close()
