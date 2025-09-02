"""
Script to visualize and print computational graphs.

This module provides functions to print computational graphs in various formats:
- Text-based tree structure
- Graphviz DOT format (for generating visual graphs)
- Compact summary format
"""

from autodiff.graph.var import Variable, Constant
from autodiff.graph.node import Node
from typing import Set, Dict, List


def print_graph(node: Node, format: str = "tree", max_depth: int = 10):
    """
    Print a computational graph starting from a given node.

    Args:
        node: The root node to start printing from
        format: Format to use ('tree', 'dot', 'summary')
        max_depth: Maximum depth to traverse
    """
    if format == "tree":
        print_tree(node, max_depth)
    elif format == "dot":
        print_dot(node, max_depth)
    elif format == "summary":
        print_summary(node, max_depth)
    else:
        raise ValueError(f"Unknown format: {format}")


def print_tree(node: Node, max_depth: int = 10):
    """Print the computational graph as a tree structure."""
    print("Computational Graph (Tree View):")
    print("=" * 50)
    visited = set()
    _print_tree_recursive(node, 0, max_depth, visited, "")


def _print_tree_recursive(node: Node, depth: int, max_depth: int, visited: Set[id], prefix: str):
    """Recursive helper for tree printing."""
    if depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    node_id = id(node)
    is_revisit = node_id in visited
    visited.add(node_id)

    # Print current node
    node_info = _get_node_info(node)
    marker = "↻" if is_revisit else "•"
    print(f"{prefix}{marker} {node_info}")

    # Don't traverse children if we've already visited this node
    if is_revisit:
        return

    # Print operation and its inputs
    if hasattr(node, 'op') and node.op is not None:
        op_info = _get_op_info(node.op)
        print(f"{prefix}  └─ {op_info}")

        # Print inputs
        if hasattr(node.op, 'inputs'):
            for i, input_node in enumerate(node.op.inputs):
                is_last = i == len(node.op.inputs) - 1
                child_prefix = prefix + ("     " if is_last else "  │  ")
                connector = "└─" if is_last else "├─"
                print(f"{prefix}     {connector} input[{i}]:")
                _print_tree_recursive(input_node, depth + 1, max_depth, visited, child_prefix)


def print_dot(node: Node, max_depth: int = 10):
    """Print the computational graph in Graphviz DOT format."""
    print("digraph ComputationalGraph {")
    print("  rankdir=BT;")
    print("  node [shape=box];")

    visited = set()
    node_counter = [0]  # Use list to make it mutable
    node_to_id = {}  # Maps id(node) to string identifier
    edges = []

    _collect_dot_info_fixed(node, 0, max_depth, visited, node_to_id, edges, node_counter)

    # Print nodes - we need to collect all nodes first
    all_nodes = {}
    _collect_all_nodes(node, 0, max_depth, set(), all_nodes)

    # Print node definitions
    for node_id_val, node_id_str in node_to_id.items():
        if node_id_val in all_nodes:
            node_obj = all_nodes[node_id_val]
            node_info = _get_node_info(node_obj).replace('"', '\\"')
            print(f'  {node_id_str} [label="{node_info}"];')

    # Print edges
    for parent_id, child_id, label in edges:
        if label:
            print(f'  {parent_id} -> {child_id} [label="{label}"];')
        else:
            print(f'  {parent_id} -> {child_id};')

    print("}")
    # dot -Tpng graph.dot -o graph.png


def _collect_dot_info_fixed(node: Node, depth: int, max_depth: int, visited: Set[id],
                            node_to_id: Dict[id, str], edges: List[tuple], node_counter: List[int]):
    """Collect information for DOT format generation using node IDs."""
    if depth > max_depth:
        return

    node_id_val = id(node)
    if node_id_val in visited:
        return

    visited.add(node_id_val)

    # Assign ID if not already assigned
    if node_id_val not in node_to_id:
        node_to_id[node_id_val] = f"node_{node_counter[0]}"
        node_counter[0] += 1

    if hasattr(node, 'op') and node.op is not None:
        # Create ID for operation
        op_id_val = id(node.op)
        if op_id_val not in node_to_id:
            node_to_id[op_id_val] = f"op_{node_counter[0]}"
            node_counter[0] += 1

        # Edge from operation to result
        edges.append((node_to_id[op_id_val], node_to_id[node_id_val], ""))

        # Edges from inputs to operation
        if hasattr(node.op, 'inputs'):
            for i, input_node in enumerate(node.op.inputs):
                _collect_dot_info_fixed(input_node, depth + 1, max_depth, visited,
                                        node_to_id, edges, node_counter)
                input_id_val = id(input_node)
                if input_id_val in node_to_id:
                    edges.append((node_to_id[input_id_val], node_to_id[op_id_val], f"input[{i}]"))


def _collect_all_nodes(node: Node, depth: int, max_depth: int, visited: Set[id],
                       all_nodes: Dict[id, Node]):
    """Collect all nodes in the graph."""
    if depth > max_depth:
        return

    node_id_val = id(node)
    if node_id_val in visited:
        return

    visited.add(node_id_val)
    all_nodes[node_id_val] = node

    if hasattr(node, 'op') and node.op is not None:
        # Also store the operation
        op_id_val = id(node.op)
        if op_id_val not in all_nodes:
            all_nodes[op_id_val] = node.op

        if hasattr(node.op, 'inputs'):
            for input_node in node.op.inputs:
                _collect_all_nodes(input_node, depth + 1, max_depth, visited, all_nodes)


def _collect_dot_info(node: Node, depth: int, max_depth: int, visited: Set[id],
                      node_ids: Dict[Node, str], edges: List[tuple]):
    """Collect information for DOT format generation."""
    if depth > max_depth:
        return

    node_id_val = id(node)
    if node_id_val in visited:
        return

    visited.add(node_id_val)
    node_ids[node] = f"node_{len(node_ids)}"

    if hasattr(node, 'op') and node.op is not None:
        op_id = f"op_{len(node_ids)}"
        node_ids[node.op] = op_id

        # Edge from operation to result
        edges.append((op_id, node_ids[node], ""))

        # Edges from inputs to operation
        if hasattr(node.op, 'inputs'):
            for i, input_node in enumerate(node.op.inputs):
                _collect_dot_info(input_node, depth + 1, max_depth, visited, node_ids, edges)
                if input_node in node_ids:
                    edges.append((node_ids[input_node], op_id, f"input[{i}]"))


def print_summary(node: Node, max_depth: int = 10):
    """Print a compact summary of the computational graph."""
    print("Computational Graph Summary:")
    print("=" * 50)

    visited = set()
    stats = {"variables": 0, "constants": 0, "operations": {}}

    _collect_summary_stats(node, 0, max_depth, visited, stats)

    print(f"Total Variables: {stats['variables']}")
    print(f"Total Constants: {stats['constants']}")
    print("Operations:")
    for op_name, count in stats['operations'].items():
        print(f"  {op_name}: {count}")

    print(f"\nGraph traversal (depth {max_depth}):")
    visited.clear()
    _print_summary_path(node, 0, max_depth, visited)


def _collect_summary_stats(node: Node, depth: int, max_depth: int,
                           visited: Set[id], stats: Dict):
    """Collect statistics for summary."""
    if depth > max_depth:
        return

    node_id = id(node)
    if node_id in visited:
        return

    visited.add(node_id)

    if isinstance(node, Variable):
        stats["variables"] += 1
    elif isinstance(node, Constant):
        stats["constants"] += 1

    if hasattr(node, 'op') and node.op is not None:
        op_name = node.op.__class__.__name__
        stats["operations"][op_name] = stats["operations"].get(op_name, 0) + 1

        if hasattr(node.op, 'inputs'):
            for input_node in node.op.inputs:
                _collect_summary_stats(input_node, depth + 1, max_depth, visited, stats)


def _print_summary_path(node: Node, depth: int, max_depth: int, visited: Set[id]):
    """Print path summary."""
    if depth > max_depth:
        return

    node_id = id(node)
    if node_id in visited:
        return

    visited.add(node_id)

    indent = "  " * depth
    node_info = _get_node_info(node)
    print(f"{indent}{node_info}")

    if hasattr(node, 'op') and node.op is not None and hasattr(node.op, 'inputs'):
        for input_node in node.op.inputs:
            _print_summary_path(input_node, depth + 1, max_depth, visited)


def _get_node_info(node: Node) -> str:
    """Get formatted information about a node."""
    if isinstance(node, Variable):
        grad_info = f", grad={node.grad}" if node.grad is not None else ""
        return f"Variable(data={node.data:.4f}{grad_info})"
    elif isinstance(node, Constant):
        return f"Constant(data={node.data:.4f})"
    elif hasattr(node, '__class__') and hasattr(node, '__dict__'):
        # This is likely an operation
        return f"Op: {node.__class__.__name__}"
    else:
        return f"{node.__class__.__name__}({getattr(node, 'data', 'N/A')})"


def _get_op_info(op) -> str:
    """Get formatted information about an operation."""
    return f"Op: {op.__class__.__name__}"


def create_sample_graph():
    """Create a sample computational graph for testing."""
    # Create some variables
    x = Variable(2.0)
    y = Variable(3.0)
    z = Variable(1.0)

    # Create a more complex computation
    # result = (x + y) * z + x
    from autodiff.graph.operations.math import Add, Mul

    # x + y
    add_op = Add()
    add_op.inputs = [x, y]
    sum_xy = Variable(Add._forward(x, y))
    sum_xy.op = add_op

    # (x + y) * z
    mult_op = Mul()
    mult_op.inputs = [sum_xy, z]
    prod = Variable(Mul._forward(sum_xy, z))
    prod.op = mult_op

    # ((x + y) * z) + x
    add_op2 = Add()
    add_op2.inputs = [prod, x]
    result = Variable(Add._forward(prod, x))
    result.op = add_op2

    return result


if __name__ == "__main__":
    # Create a sample graph
    print("Creating sample computational graph...")
    print("Expression: ((x + y) * z) + x where x=2.0, y=3.0, z=1.0")
    print()

    try:
        graph = create_sample_graph()

        # Print in different formats
        print_graph(graph, format="tree")
        print()
        print_graph(graph, format="summary")
        print()
        print_graph(graph, format="dot")

    except Exception as e:
        print(f"Error creating sample graph: {e}")
        print("This might be due to the specific implementation details.")
        print("Try using this script with your own computational graph:")
        print()
        print("# Example usage:")
        print("from autodiff.graph.var import Variable")
        print("from print import print_graph")
        print("")
        print("# Create your computational graph")
        print("x = Variable(2.0)")
        print("# ... perform operations ...")
        print("# Print the graph")
        print("print_graph(result_variable)")
