"""
Module for analyzing the semantic profile of code snippets using LJWP primitives.
"""

import ast
from typing import Dict, Any
from src.phi_geometric_engine import PhiCoordinate, universal_semantic_mix

class CodeSemanticAnalyzer:
    """
    Analyzes a code snippet and maps its constructs to LJWP semantic primitives.
    """

    def __init__(self):
        # Simplified mapping of AST node types to LJWP primary weights
        # This is a starting point and will be refined.
        self.node_to_ljwp_map = {
            # Power (agency/action)
            ast.Assign: {'power': 1.0}, # Assignment is an action
            ast.For: {'power': 1.0},    # Loops are repeated actions
            ast.While: {'power': 1.0},  # Loops are repeated actions
            ast.Call: {'power': 1.0},   # Function calls are actions
            ast.Return: {'power': 1.0}, # Returning a value is an action
            ast.Expr: {'power': 0.5},   # Expressions can be actions (e.g., function calls)

            # Justice (order/structure)
            ast.List: {'justice': 1.0}, # Lists are ordered structures
            ast.Dict: {'justice': 1.0}, # Dictionaries are structured mappings
            ast.ClassDef: {'justice': 1.0}, # Class definitions create structure
            ast.FunctionDef: {'justice': 0.5}, # Function definitions create structure
            ast.Module: {'justice': 1.0}, # Modules provide overall structure

            # Wisdom (knowledge/abstraction)
            ast.FunctionDef: {'wisdom': 1.0}, # Function definitions encapsulate logic/knowledge
            ast.If: {'wisdom': 1.0},     # Conditionals involve decision-making/logic
            ast.Compare: {'wisdom': 1.0}, # Comparisons involve logic
            ast.BoolOp: {'wisdom': 1.0},  # Boolean operations involve logic
            ast.comprehension: {'wisdom': 1.0}, # Comprehensions are concise logic

            # Love (connection/interface)
            ast.Import: {'love': 1.0},   # Imports connect modules
            ast.ImportFrom: {'love': 1.0}, # Imports connect modules
            # ast.arguments: {'love': 1.0}, # Function arguments are interfaces (more complex to map directly)
        }

    def analyze_code(self, code_snippet: str) -> PhiCoordinate:
        """
        Analyzes a Python code snippet and returns its LJWP semantic profile.
        """
        try:
            tree = ast.parse(code_snippet)
        except SyntaxError as e:
            print(f"Syntax Error in code snippet: {e}")
            return PhiCoordinate(0, 0, 0, 0) # Return neutral if syntax error

        ljwp_weights: Dict[str, float] = {'love': 0.0, 'justice': 0.0, 'power': 0.0, 'wisdom': 0.0}

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type in self.node_to_ljwp_map:
                weights = self.node_to_ljwp_map[node_type]
                for primary, weight in weights.items():
                    ljwp_weights[primary] += weight
            
            # Special handling for some nodes based on context or attributes
            if isinstance(node, ast.arguments): # Function arguments represent interfaces/connections
                ljwp_weights['love'] += 0.5
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store): # Variable assignment target
                ljwp_weights['justice'] += 0.2 # Naming/structuring data
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load): # Variable usage
                ljwp_weights['wisdom'] += 0.1 # Accessing knowledge

        # Normalize weights to get a PhiCoordinate
        # Use universal_semantic_mix for normalization
        return universal_semantic_mix(ljwp_weights)

# Example Usage (for testing during development)
if __name__ == "__main__":
    analyzer = CodeSemanticAnalyzer()

    code1 = """
def calculate_sum(a, b):
    x = a + b
    return x
"""
    profile1 = analyzer.analyze_code(code1)
    print(f"Code 1 Semantic Profile: {profile1}")

    code2 = """
class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")

import requests
response = requests.get("http://example.com")
"""
    profile2 = analyzer.analyze_code(code2)
    print(f"Code 2 Semantic Profile: {profile2}")

    code3 = """
for i in range(10):
    if i % 2 == 0:
        print(i)
"""
    profile3 = analyzer.analyze_code(code3)
    print(f"Code 3 Semantic Profile: {profile3}")

    code4 = """
# This is a comment
x = 10 # Another comment
"""
    profile4 = analyzer.analyze_code(code4)
    print(f"Code 4 Semantic Profile: {profile4}")

    code5 = """
import os
import sys

def process_files(files):
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            # Process content (placeholder for power/wisdom)
            if 'error' in content:
                print(f"Error in {file}")
            else:
                print(f"Processed {file}")
"""
    profile5 = analyzer.analyze_code(code5)
    print(f"Code 5 Semantic Profile: {profile5}")

    code6 = """
# Syntax error example
def my_func(:
    pass
"""
    profile6 = analyzer.analyze_code(code6)
    print(f"Code 6 Semantic Profile (Error): {profile6}")
