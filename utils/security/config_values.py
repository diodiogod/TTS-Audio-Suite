"""Parse configuration literals and their list addition/repetition expressions."""

import ast


MAX_ITEMS = 100_000


def parse_config_value(value):
    if not isinstance(value, str):
        return value
    if len(value) > 1_000_000:
        raise ValueError("Configuration expression is too large.")
    tree = ast.parse(value, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 10_000:
        raise ValueError("Configuration expression has too many elements.")

    def visit(node, depth=0):
        if depth > 100:
            raise ValueError("Configuration expression is too deeply nested.")
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            result = ast.literal_eval(node)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [visit(item, depth + 1) for item in node.elts]
            result = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
        elif isinstance(node, ast.Dict):
            result = {visit(key, depth + 1): visit(item, depth + 1) for key, item in zip(node.keys, node.values)}
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add) and isinstance(left, (list, tuple)) and type(left) is type(right):
                if len(left) + len(right) > MAX_ITEMS:
                    raise ValueError("Configuration sequence is too large.")
                result = left + right
            elif isinstance(node.op, ast.Mult):
                if type(left) is int:
                    left, right = right, left
                if not isinstance(left, (list, tuple)) or type(right) is not int or len(left) * max(0, right) > MAX_ITEMS:
                    raise ValueError("Only bounded list/tuple repetition is supported.")
                result = left * right
            else:
                raise ValueError("Only list/tuple addition and repetition are supported.")
        else:
            raise ValueError("Configuration must contain data literals, not executable expressions.")
        if isinstance(result, (list, tuple, dict, set, str, bytes)) and len(result) > MAX_ITEMS:
            raise ValueError("Configuration value is too large.")
        return result

    return visit(tree.body)
