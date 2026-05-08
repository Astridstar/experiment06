from collections import Counter, defaultdict, deque
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
    in_degrees  = {n: len(dag.dependencies.get(n, set())) for n in dag.nodes}
    out_degrees = {n: len(dag.adjacency.get(n, set()))    for n in dag.nodes}
    sources = [n for n, d in in_degrees.items()  if d == 0]
    sinks   = [n for n, d in out_degrees.items() if d == 0]

    levels = execution_levels(dag)
    widest_level = max(range(len(levels)), key=lambda i: len(levels[i]))

    total_edges = sum(len(v) for v in dag.adjacency.values())
    avg_in  = total_edges / len(dag.nodes)
    avg_out = avg_in

    print(f"Nodes          : {len(dag.nodes)}")
    print(f"Edges          : {total_edges}")
    print(f"Execution levels: {len(levels)}")
    print(f"Max parallelism: {len(levels[widest_level])} nodes at level {widest_level}")
    print(f"Sources (no parents): {len(sources)}")
    print(f"Sinks   (no children): {len(sinks)}")
    print(f"In-degree  — max: {max(in_degrees.values())}  avg: {avg_in:.1f}")
    print(f"Out-degree — max: {max(out_degrees.values())}  avg: {avg_out:.1f}")
    print()

    print("Top 10 bottleneck nodes (highest combined degree):")
    top = sorted(dag.nodes, key=lambda n: in_degrees[n] + out_degrees[n], reverse=True)[:10]
    for node in top:
        print(f"  {node}  (in={in_degrees[node]}, out={out_degrees[node]})")
    print()

    longest_path = find_longest_path(dag)
    print(f"Longest path   : {len(longest_path) - 1} hops ({len(longest_path)} nodes)")
    for i, node in enumerate(longest_path):
        prefix = "    " if i == 0 else "    -> "
        print(f"{prefix}{node}")
    print()

    chains = find_linear_chains(dag)
    print(f"Linear chains (unbranched runs): {len(chains)}")
    if chains:
        longest = chains[0]
        print(f"  Longest chain: {len(longest)} nodes")
        for i, node in enumerate(longest):
            prefix = "    " if i == 0 else "    -> "
            print(f"{prefix}{node}")


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

    try:
        levels = execution_levels(dag)
        for level_idx, level_nodes in enumerate(levels):
            for node in level_nodes:
                G.nodes[node]["subset"] = level_idx
        align = "vertical" if left_to_right else "horizontal"
        pos = nx.multipartite_layout(G, subset_key="subset", align=align)
    except ValueError:
        # Clustering can introduce apparent cycles between clusters even when the
        # original DAG is acyclic, so fall back to a layout that tolerates cycles.
        print("Warning: cluster graph contains cycles; using spring layout instead of multipartite.")
        pos = nx.spring_layout(G, seed=42)

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


def _bfs_expand(frontier_map: Dict[str, Set[str]], start: str, hops: int, included: Set[str]) -> None:
    frontier = {start}
    for _ in range(hops):
        next_frontier = {
            neighbour
            for n in frontier
            for neighbour in frontier_map.get(n, set())
            if neighbour not in included
        }
        included.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break


def node_neighborhood(dag: Dag, node: str, hops: int) -> Dag:
    if node not in dag.nodes:
        raise ValueError(f"Node not found: {node!r}")

    included = {node}
    _bfs_expand(dag.adjacency, node, hops, included)
    _bfs_expand(dag.dependencies, node, hops, included)

    adjacency = {n: dag.adjacency.get(n, set()) & included for n in included}
    dependencies = {n: dag.dependencies.get(n, set()) & included for n in included}
    return Dag(adjacency=adjacency, dependencies=dependencies, nodes=included)


def _ascii_push_children(
    stack: List[Tuple[str, str, bool]],
    adjacency: Dict[str, Set[str]],
    node: str,
    prefix: str,
    is_last: bool,
) -> None:
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = sorted(adjacency.get(node, set()))
    items = [(c, child_prefix, i == len(children) - 1) for i, c in enumerate(children)]
    stack.extend(reversed(items))


def render_dag_ascii(dag: Dag, output_file: str = "dag") -> str:
    roots = sorted(n for n in dag.nodes if not dag.dependencies.get(n))
    visited: Set[str] = set()
    lines: List[str] = []

    for root in roots:
        lines.append(root)
        visited.add(root)
        children = sorted(dag.adjacency.get(root, set()))
        stack: List[Tuple[str, str, bool]] = [
            (child, "", i == len(children) - 1) for i, child in enumerate(children)
        ]
        stack.reverse()

        while stack:
            node, prefix, is_last = stack.pop()
            connector = "└── " if is_last else "├── "
            if node in visited:
                lines.append(prefix + connector + node + " (ref)")
                continue
            visited.add(node)
            lines.append(prefix + connector + node)
            _ascii_push_children(stack, dag.adjacency, node, prefix, is_last)

        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    result = "\n".join(lines)
    try:
        print(result)
    except UnicodeEncodeError:
        print(result.encode("ascii", errors="replace").decode("ascii"))
    path = f"{output_file}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(result + "\n")
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





def find_longest_path(dag: Dag) -> List[str]:
    """Return the node sequence forming the longest path in the DAG."""
    order = topological_sort(dag)
    dist: Dict[str, int] = dict.fromkeys(dag.nodes, 0)
    prev: Dict[str, str | None] = dict.fromkeys(dag.nodes, None)

    for node in order:
        for child in dag.adjacency.get(node, set()):
            if dist[node] + 1 > dist[child]:
                dist[child] = dist[node] + 1
                prev[child] = node

    end = max(dag.nodes, key=lambda n: dist[n])
    path: List[str] = []
    current: str | None = end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path


def find_linear_chains(dag: Dag, min_length: int = 3) -> List[List[str]]:
    """
    Find maximal unbranched paths: every interior node has exactly one parent
    and one child. Returns chains sorted longest-first.
    """
    visited: Set[str] = set()
    chains: List[List[str]] = []

    for node in topological_sort(dag):
        if node in visited:
            continue
        visited.add(node)
        chain = [node]
        current = node
        while True:
            children = dag.adjacency.get(current, set())
            if len(children) != 1:
                break
            (child,) = children
            if len(dag.dependencies.get(child, set())) != 1:
                break
            chain.append(child)
            visited.add(child)
            current = child
        if len(chain) >= min_length:
            chains.append(chain)

    return sorted(chains, key=lambda c: -len(c))


def find_repeated_sequences(
    dag: Dag,
    max_length: int = 4,
    min_count: int = 2,
    max_paths_per_node: int = 500,
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Find normalized node-name sequences (last underscore segment) that appear
    as consecutive paths at least min_count times across the DAG.
    max_paths_per_node caps memory on high-fanin nodes; a warning is printed
    if the cap is hit.
    """
    def normalize(n: str) -> str:
        return n.split("_")[-1]

    seq_counts: Counter = Counter()
    ending: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    capped = False

    for node in topological_sort(dag):
        norm = normalize(node)
        here: Set[Tuple[str, ...]] = {(norm,)}
        for parent in dag.dependencies.get(node, set()):
            for path in ending.get(parent, set()):
                if len(path) < max_length:
                    here.add(path + (norm,))

        if len(here) > max_paths_per_node:
            here = set(list(here)[:max_paths_per_node])
            capped = True

        ending[node] = here
        for path in here:
            if len(path) >= 2:
                seq_counts[path] += 1

    if capped:
        print(
            f"Warning: some nodes exceeded {max_paths_per_node} path variants; "
            "sequence counts may be approximate.",
            flush=True,
        )

    return sorted(
        [(seq, count) for seq, count in seq_counts.items() if count >= min_count],
        key=lambda x: (-x[1], x[0]),
    )


def _normalize_node(node: str, depth: int) -> str:
    """Return the last `depth` underscore-separated segments, or the full name if no underscores."""
    parts = node.split("_")
    return "_".join(parts[-depth:]) if len(parts) >= depth else node


def _print_chain_groups(
    groups: Dict[Tuple[str, ...], List[List[str]]],
    min_count: int,
) -> None:
    repeated = {s: g for s, g in groups.items() if len(g) >= min_count}
    unique   = {s: g for s, g in groups.items() if len(g) <  min_count}
    total    = sum(len(g) for g in groups.values())
    print(f"=== Linear chains: {total} total, "
          f"{len(repeated)} repeated patterns, {len(unique)} unique ===")
    if not repeated:
        print("  (no repeated chain patterns)")
    else:
        print("\n  Repeated chain patterns:")
        for sig, instances in sorted(repeated.items(), key=lambda x: -len(x[1])):
            print(f"    {len(instances):4d}x  [{len(sig)}]  {' -> '.join(sig)}")
            for inst in instances[:2]:
                print(f"           e.g. {' -> '.join(inst)}")
            if len(instances) > 2:
                print(f"           ... and {len(instances) - 2} more")
    if unique:
        print(f"\n  Unique chains (first 10 of {len(unique)}):")
        for _sig, (inst, *_) in list(unique.items())[:10]:
            print(f"    [{len(_sig)}]  {' -> '.join(inst)}")


def _print_path_patterns(seqs: List[Tuple[Tuple[str, ...], int]], max_length: int,
                         min_count: int, normalize_depth: int) -> None:
    print(f"=== Repeated name patterns (length 2-{max_length}, min {min_count}x, "
          f"normalized to last {normalize_depth} segment(s)) ===")
    if not seqs:
        print("  (none)")
        return
    for seq, count in seqs[:30]:
        print(f"  {count:5d}x  {' -> '.join(seq)}")
    if len(seqs) > 30:
        print(f"  ... and {len(seqs) - 30} more patterns")


def _report_sequences(dag: Dag, max_length: int, min_count: int, normalize_depth: int) -> None:
    normalize = lambda n: _normalize_node(n, normalize_depth)

    chains = find_linear_chains(dag)
    groups: Dict[Tuple[str, ...], List[List[str]]] = defaultdict(list)
    for chain in chains:
        groups[tuple(normalize(n) for n in chain)].append(chain)

    _print_chain_groups(groups, min_count)
    print()
    seqs = find_repeated_sequences(dag, max_length=max_length, min_count=min_count)
    _print_path_patterns(seqs, max_length, min_count, normalize_depth)


def _report_cycles(lines: List[str], max_cycles: int) -> None:
    adjacency, _, _ = _parse_edges(lines)
    cycles = find_cycles(adjacency, max_cycles=max_cycles)
    if not cycles:
        print("No cycles found.")
    else:
        print(f"Found {len(cycles)} cycle(s) (showing up to {max_cycles}):\n")
        for i, cycle in enumerate(cycles, 1):
            print(f"  Cycle {i}: {' -> '.join(cycle)}")


def _render(dag: Dag, renderer: str, output_file: str, left_to_right: bool,
            labels: Dict[str, str] | None) -> str:
    if renderer == "ascii":
        return render_dag_ascii(dag, output_file=output_file)
    if renderer == "matplotlib":
        return render_dag_matplotlib(dag, output_file=output_file, left_to_right=left_to_right, labels=labels)
    if renderer == "pyvis":
        return render_dag_pyvis(dag, output_file=output_file, left_to_right=left_to_right)
    if renderer == "graphviz":
        return render_dag_graphviz(dag, output_file=output_file, left_to_right=left_to_right)
    return render_dag_mermaid(dag, output_file=output_file, left_to_right=left_to_right)


def _print_longest_path(dag: Dag) -> None:
    path = find_longest_path(dag)
    print(f"Longest path: {len(path) - 1} hops ({len(path)} nodes)")
    for i, node in enumerate(path):
        prefix = "    " if i == 0 else "    -> "
        print(f"{prefix}{node}")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Render a dependency file as a graphical DAG.")
    parser.add_argument("file", help="Path to the dependency file (e.g. deps.txt)")
    parser.add_argument("-o", "--output", default="dag", help="Output file name without extension (default: dag)")
    parser.add_argument("--top-bottom", action="store_true", help="Layout top-to-bottom instead of left-to-right")
    parser.add_argument("--renderer", choices=["matplotlib", "pyvis", "graphviz", "mermaid", "ascii"], default="matplotlib", help="Renderer to use (default: matplotlib)")
    parser.add_argument("--node", metavar="NAME", help="Restrict rendering to the neighborhood of this node")
    parser.add_argument("--hops", type=int, default=2, metavar="N",
                        help="Hops upstream and downstream when using --node (default: 2)")
    parser.add_argument("--cluster-depth", type=int, default=0, metavar="N",
                        help="Collapse nodes into clusters by first N underscore-separated name segments before rendering")
    parser.add_argument("--analyze", action="store_true",
                        help="Print node/cluster statistics and collapsible candidates, then exit")
    parser.add_argument("--detect-cycles", action="store_true",
                        help="Find and print all cycles without aborting, then exit")
    parser.add_argument("--max-cycles", type=int, default=20, metavar="N",
                        help="Maximum number of cycles to report with --detect-cycles (default: 20)")
    parser.add_argument("--longest-path", action="store_true",
                        help="Print the longest path in the DAG, then exit")
    parser.add_argument("--find-sequences", action="store_true",
                        help="Detect repeated node sequences and linear chains, then exit")
    parser.add_argument("--min-sequence-count", type=int, default=2, metavar="N",
                        help="Minimum occurrences to report a repeated sequence (default: 2)")
    parser.add_argument("--max-sequence-length", type=int, default=4, metavar="N",
                        help="Maximum sequence length to search for (default: 4)")
    parser.add_argument("--sequence-depth", type=int, default=1, metavar="N",
                        help="Number of trailing underscore-segments used to normalize node names for sequence grouping (default: 1)")
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

    if args.longest_path:
        _print_longest_path(dag)
        return

    if args.find_sequences:
        _report_sequences(dag, max_length=args.max_sequence_length,
                          min_count=args.min_sequence_count, normalize_depth=args.sequence_depth)
        return

    if args.node:
        try:
            dag = node_neighborhood(dag, args.node, args.hops)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.cluster_depth > 0:
        dag, clusters = cluster_dag(dag, depth=args.cluster_depth)
        labels = {name: f"{name}\n({len(members)})" for name, members in clusters.items()}
    else:
        labels = None

    rendered = _render(dag, args.renderer, args.output, not args.top_bottom, labels)
    print(f"DAG rendered to: {rendered}")


if __name__ == "__main__":
    main()
