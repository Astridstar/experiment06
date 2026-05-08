from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple


@dataclass
class Dag:
    adjacency: Dict[str, Set[str]]      # parent -> children
    dependencies: Dict[str, Set[str]]   # child -> parents
    nodes: Set[str]


def parse_dependency_line(line: str) -> List[Tuple[str, str]]:
    line = line.strip()

    if not line:
        raise ValueError("Empty dependency line found")

    parts = [p.strip() for p in line.split(">>")]

    if len(parts) < 2 or any(p == "" for p in parts):
        raise ValueError(f"Invalid dependency format: {line}")

    pairs = []
    for i in range(len(parts) - 1):
        parent, child = parts[i], parts[i + 1]
        if parent == child:
            raise ValueError(f"Self-dependency is not allowed: {line}")
        pairs.append((parent, child))

    return pairs


def _parse_edges(lines: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Set[str]]:
    """Parse dependency lines into raw adjacency structures without cycle validation."""
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    dependencies: Dict[str, Set[str]] = defaultdict(set)
    nodes: Set[str] = set()

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        for parent, child in parse_dependency_line(line):
            adjacency[parent].add(child)
            dependencies[child].add(parent)
            nodes.add(parent)
            nodes.add(child)
            adjacency.setdefault(child, set())
            dependencies.setdefault(parent, set())

    return dict(adjacency), dict(dependencies), nodes


def build_dag(lines: List[str]) -> Dag:
    adjacency, dependencies, nodes = _parse_edges(lines)
    dag = Dag(adjacency=adjacency, dependencies=dependencies, nodes=nodes)
    validate_no_cycles(dag)
    return dag


def find_cycles(adjacency: Dict[str, Set[str]], max_cycles: int = 20) -> List[List[str]]:
    """
    Find cycles using iterative DFS with gray/black coloring.
    Returns up to max_cycles distinct cycle paths, each expressed as
    [n1, n2, ..., nk, n1] so the repeated node makes the loop explicit.
    Iterative to avoid hitting Python's recursion limit on large graphs.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = dict.fromkeys(adjacency, WHITE)
    cycles: List[List[str]] = []
    path: List[str] = []
    path_index: Dict[str, int] = {}

    def enter(node: str, stack: list) -> None:
        color[node] = GRAY
        path.append(node)
        path_index[node] = len(path) - 1
        stack[-1] = (node, iter(sorted(adjacency.get(node, set()))), False)

    def leave(node: str, stack: list) -> None:
        stack.pop()
        path.pop()
        path_index.pop(node, None)
        color[node] = BLACK

    def advance(child: str, stack: list) -> None:
        if color[child] == GRAY:
            cycles.append(path[path_index[child]:] + [child])
        elif color[child] == WHITE:
            stack.append((child, None, True))

    for start in sorted(adjacency):
        if color[start] != WHITE or len(cycles) >= max_cycles:
            continue

        stack: List[tuple] = [(start, None, True)]
        while stack and len(cycles) < max_cycles:
            node, children_iter, entering = stack[-1]
            if entering:
                if color[node] != WHITE:
                    stack.pop()
                else:
                    enter(node, stack)
            else:
                try:
                    advance(next(children_iter), stack)
                except StopIteration:
                    leave(node, stack)

    return cycles


def topological_sort(dag: Dag) -> List[str]:
    in_degree = {node: len(dag.dependencies.get(node, set())) for node in dag.nodes}

    queue = deque(sorted([node for node, deg in in_degree.items() if deg == 0]))
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for child in sorted(dag.adjacency.get(node, set())):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(dag.nodes):
        unresolved = sorted(node for node, deg in in_degree.items() if deg > 0)
        raise ValueError(f"Cycle detected. Unresolved nodes: {unresolved}")

    return result


def validate_no_cycles(dag: Dag) -> None:
    try:
        topological_sort(dag)
    except ValueError:
        cycles = find_cycles(dag.adjacency)
        if cycles:
            formatted = "\n".join(
                f"  Cycle {i + 1}: {' -> '.join(c)}" for i, c in enumerate(cycles)
            )
            raise ValueError(f"Cycle(s) detected:\n{formatted}") from None
        raise


def execution_levels(dag: Dag) -> List[List[str]]:
    """
    Returns nodes grouped by level.
    Nodes in the same level can run in parallel.
    """
    in_degree = {node: len(dag.dependencies.get(node, set())) for node in dag.nodes}
    current_level = sorted([node for node, deg in in_degree.items() if deg == 0])

    levels = []

    while current_level:
        levels.append(current_level)
        next_level = []

        for node in current_level:
            for child in sorted(dag.adjacency.get(node, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_level.append(child)

        current_level = sorted(next_level)

    processed = sum(len(level) for level in levels)
    if processed != len(dag.nodes):
        unresolved = [node for node, deg in in_degree.items() if deg > 0]
        raise ValueError(f"Cycle detected. Unresolved nodes: {unresolved}")

    return levels


def cluster_dag(dag: Dag, depth: int = 2) -> Tuple[Dag, Dict[str, Set[str]]]:
    """
    Collapse nodes into clusters by their first `depth` underscore-separated segments.
    Returns the condensed Dag and a mapping of cluster_name -> member nodes.
    """
    def cluster_key(node: str) -> str:
        return "_".join(node.split("_")[:depth])

    clusters: Dict[str, Set[str]] = defaultdict(set)
    for node in dag.nodes:
        clusters[cluster_key(node)].add(node)

    node_to_cluster = {node: cluster_key(node) for node in dag.nodes}

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    dependencies: Dict[str, Set[str]] = defaultdict(set)
    cluster_nodes: Set[str] = set(clusters.keys())

    for parent, children in dag.adjacency.items():
        pc = node_to_cluster[parent]
        for child in children:
            cc = node_to_cluster[child]
            if pc != cc:
                adjacency[pc].add(cc)
                dependencies[cc].add(pc)

    for node in cluster_nodes:
        adjacency.setdefault(node, set())
        dependencies.setdefault(node, set())

    condensed = Dag(
        adjacency=dict(adjacency),
        dependencies=dict(dependencies),
        nodes=cluster_nodes,
    )
    return condensed, dict(clusters)


def analyze_dag(dag: Dag) -> None:
    """Print cluster size distributions and collapsible node candidates."""
    print(f"Total nodes: {len(dag.nodes)}")
    print(f"Total edges: {sum(len(v) for v in dag.adjacency.values())}")

    levels = execution_levels(dag)
    print(f"Execution levels: {len(levels)}")
    print(f"Max parallelism: {max(len(l) for l in levels)} nodes at level {max(range(len(levels)), key=lambda i: len(levels[i]))}")
    print()

    for depth in range(1, 5):
        clusters: Dict[str, Set[str]] = defaultdict(set)
        for node in dag.nodes:
            key = "_".join(node.split("_")[:depth])
            clusters[key].add(node)
        multi = {k: v for k, v in clusters.items() if len(v) > 1}
        print(f"Depth {depth}: {len(clusters)} clusters, {len(multi)} with >1 member, "
              f"largest={max(len(v) for v in clusters.values())}")

    print()
    print("Collapsible candidates (nodes sharing all but the last underscore segment):")
    candidates: Dict[str, List[str]] = defaultdict(list)
    for node in sorted(dag.nodes):
        parts = node.split("_")
        if len(parts) > 1:
            base = "_".join(parts[:-1])
            candidates[base].append(node)
    found = 0
    for base, members in sorted(candidates.items()):
        if len(members) > 1:
            print(f"  {base}: {members}")
            found += 1
    if not found:
        print("  (none found — all nodes have a unique base name)")


def render_dag_mermaid(dag: Dag, output_file: str = "dag", left_to_right: bool = True) -> str:
    direction = "LR" if left_to_right else "TD"
    lines = [f"flowchart {direction}"]

    for parent, children in sorted(dag.adjacency.items()):
        for child in sorted(children):
            lines.append(f"    {parent} --> {child}")

    path = f"{output_file}.mmd"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def render_dag_matplotlib(dag: Dag, output_file: str = "dag", left_to_right: bool = True,
                          labels: Dict[str, str] | None = None) -> str:
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("networkx and matplotlib not installed. Run: pip install networkx matplotlib")

    G = nx.DiGraph()
    G.add_nodes_from(sorted(dag.nodes))
    for parent, children in sorted(dag.adjacency.items()):
        for child in sorted(children):
            G.add_edge(parent, child)

    levels = execution_levels(dag)
    for level_idx, level_nodes in enumerate(levels):
        for node in level_nodes:
            G.nodes[node]["subset"] = level_idx

    # align="vertical" spreads subsets left-to-right; "horizontal" spreads them top-to-bottom
    align = "vertical" if left_to_right else "horizontal"
    pos = nx.multipartite_layout(G, subset_key="subset", align=align)

    node_labels = labels or {n: n for n in dag.nodes}
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, ax=ax, labels=node_labels,
            node_color="lightblue", node_size=2000,
            font_size=10, arrowsize=20)

    path = f"{output_file}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_dag_pyvis(dag: Dag, output_file: str = "dag", left_to_right: bool = True) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis package not installed. Run: pip install pyvis")

    import json

    direction = "LR" if left_to_right else "UD"
    net = Network(directed=True, notebook=False)
    net.set_options(json.dumps({
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": direction,
                "sortMethod": "directed",
            }
        },
        "physics": {"enabled": False},
    }))

    for node in sorted(dag.nodes):
        net.add_node(node, label=node)

    for parent, children in sorted(dag.adjacency.items()):
        for child in sorted(children):
            net.add_edge(parent, child)

    path = f"{output_file}.html"
    net.save_graph(path)
    return path


def render_dag_graphviz(dag: Dag, output_file: str = "dag", left_to_right: bool = True) -> str:
    try:
        import graphviz
    except ImportError:
        raise ImportError("graphviz package not installed. Run: pip install graphviz")

    direction = "LR" if left_to_right else "TB"
    dot = graphviz.Digraph(
        engine="dot",
        graph_attr={"rankdir": direction},
    )

    for node in sorted(dag.nodes):
        dot.node(node)

    for parent, children in sorted(dag.adjacency.items()):
        for child in sorted(children):
            dot.edge(parent, child)

    path = dot.render(output_file, format="png", cleanup=True)
    return path





def _report_cycles(lines: List[str], max_cycles: int) -> None:
    adjacency, _, _ = _parse_edges(lines)
    cycles = find_cycles(adjacency, max_cycles=max_cycles)
    if not cycles:
        print("No cycles found.")
    else:
        print(f"Found {len(cycles)} cycle(s) (showing up to {max_cycles}):\n")
        for i, cycle in enumerate(cycles, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Render a dependency file as a graphical DAG.")
    parser.add_argument("file", help="Path to the dependency file (e.g. deps.txt)")
    parser.add_argument("-o", "--output", default="dag", help="Output file name without extension (default: dag)")
    parser.add_argument("--top-bottom", action="store_true", help="Layout top-to-bottom instead of left-to-right")
    parser.add_argument("--renderer", choices=["matplotlib", "pyvis", "graphviz", "mermaid"], default="matplotlib", help="Renderer to use (default: matplotlib)")
    parser.add_argument("--cluster-depth", type=int, default=0, metavar="N",
                        help="Collapse nodes into clusters by first N underscore-separated name segments before rendering")
    parser.add_argument("--analyze", action="store_true",
                        help="Print node/cluster statistics and collapsible candidates, then exit")
    parser.add_argument("--detect-cycles", action="store_true",
                        help="Find and print all cycles without aborting, then exit")
    parser.add_argument("--max-cycles", type=int, default=20, metavar="N",
                        help="Maximum number of cycles to report with --detect-cycles (default: 20)")
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.detect_cycles:
        _report_cycles(lines, max_cycles=args.max_cycles)
        return

    try:
        dag = build_dag(lines)
    except ValueError as e:
        print(f"Error building DAG: {e}", file=sys.stderr)
        sys.exit(1)

    if args.analyze:
        analyze_dag(dag)
        return

    if args.cluster_depth > 0:
        dag, clusters = cluster_dag(dag, depth=args.cluster_depth)
        labels = {name: f"{name}\n({len(members)})" for name, members in clusters.items()}
    else:
        labels = None

    left_to_right = not args.top_bottom
    if args.renderer == "matplotlib":
        rendered = render_dag_matplotlib(dag, output_file=args.output, left_to_right=left_to_right, labels=labels)
    elif args.renderer == "pyvis":
        rendered = render_dag_pyvis(dag, output_file=args.output, left_to_right=left_to_right)
    elif args.renderer == "graphviz":
        rendered = render_dag_graphviz(dag, output_file=args.output, left_to_right=left_to_right)
    else:
        rendered = render_dag_mermaid(dag, output_file=args.output, left_to_right=left_to_right)
    print(f"DAG rendered to: {rendered}")


if __name__ == "__main__":
    main()
