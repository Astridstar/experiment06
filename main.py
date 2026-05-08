from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple


@dataclass
class Dag:
    adjacency: Dict[str, Set[str]]      # parent -> children
    dependencies: Dict[str, Set[str]]   # child -> parents
    nodes: Set[str]


def parse_dependency_line(line: str) -> Tuple[str, str]:
    line = line.strip()

    if not line:
        raise ValueError("Empty dependency line found")

    parts = line.split(">>")
    if len(parts) != 2:
        raise ValueError(f"Invalid dependency format: {line}")

    parent = parts[0].strip()
    child = parts[1].strip()

    if not parent or not child:
        raise ValueError(f"Invalid dependency format: {line}")

    if parent == child:
        raise ValueError(f"Self-dependency is not allowed: {line}")

    return parent, child


def build_dag(lines: List[str]) -> Dag:
    adjacency = defaultdict(set)
    dependencies = defaultdict(set)
    nodes = set()

    for line in lines:
        if not line.strip():
            continue

        parent, child = parse_dependency_line(line)

        adjacency[parent].add(child)
        dependencies[child].add(parent)

        nodes.add(parent)
        nodes.add(child)

        adjacency.setdefault(child, set())
        dependencies.setdefault(parent, set())

    dag = Dag(
        adjacency=dict(adjacency),
        dependencies=dict(dependencies),
        nodes=nodes
    )

    validate_no_cycles(dag)
    return dag


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
        unresolved = [node for node, deg in in_degree.items() if deg > 0]
        raise ValueError(f"Cycle detected. Unresolved nodes: {unresolved}")

    return result


def validate_no_cycles(dag: Dag) -> None:
    topological_sort(dag)


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





def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Render a dependency file as a graphical DAG.")
    parser.add_argument("file", help="Path to the dependency file (e.g. deps.txt)")
    parser.add_argument("-o", "--output", default="dag", help="Output file name without extension (default: dag)")
    parser.add_argument("--top-bottom", action="store_true", help="Layout top-to-bottom instead of left-to-right")
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        dag = build_dag(lines)
    except ValueError as e:
        print(f"Error building DAG: {e}", file=sys.stderr)
        sys.exit(1)

    rendered = render_dag_mermaid(dag, output_file=args.output, left_to_right=not args.top_bottom)
    print(f"DAG rendered to: {rendered}")


if __name__ == "__main__":
    main()
