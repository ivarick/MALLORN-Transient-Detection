"""Enumerate all top-level defs, classes, and decorated functions in nuton.py."""
import ast
import sys

with open('nuton.py', 'r', encoding='utf-8', errors='replace') as f:
    src = f.read()

try:
    tree = ast.parse(src)
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    sys.exit(1)

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        parent = getattr(node, '_parent', None)
        print(f"L{node.lineno:5d}  {'class' if isinstance(node, ast.ClassDef) else 'def  '}  {node.name}")
