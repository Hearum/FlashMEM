"""Capture short draft queries while retaining the same selector on long inputs."""

from dataclasses import dataclass

import torch

from long_selector import LongSelector


@dataclass
class QueryGraph:
    graph: torch.cuda.CUDAGraph
    host_ids: torch.Tensor
    device_ids: torch.Tensor
    queries: torch.Tensor


class QueryGraphSelector(LongSelector):
    def __init__(self, bank):
        super().__init__(bank, device="cuda")
        self.query_graphs = {}

    @torch.inference_mode()
    def capture_queries(self, sizes=(8, 16, 32)):
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(device=self.device)
        stream.wait_stream(current)
        for size in sizes:
            host_ids = torch.zeros((1, size), dtype=torch.long, pin_memory=True)
            device_ids = torch.zeros((1, size), dtype=torch.long, device=self.device)

            def project_static_input():
                self.mode = "query"
                self.captured = []
                self.model.model(input_ids=device_ids, use_cache=False)
                output = torch.stack(self.captured)
                self.mode = None
                self.captured = []
                return output

            with torch.cuda.stream(stream):
                for _ in range(3):
                    project_static_input()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                queries = project_static_input()
            self.query_graphs[size] = QueryGraph(graph, host_ids, device_ids, queries)
        current.wait_stream(stream)

    def project(self, ids, mode):
        if mode == "query":
            for capacity, work in self.query_graphs.items():
                if 0 < len(ids) <= capacity:
                    work.host_ids[0, :len(ids)].copy_(torch.as_tensor(ids, dtype=torch.long))
                    work.device_ids.copy_(work.host_ids, non_blocking=True)
                    work.graph.replay()
                    # Attention is causal, so right padding does not affect
                    # any subsequent score. The scoring path rotates in place.
                    return work.queries[:, :, :len(ids)].clone()
        return super().project(ids, mode)
