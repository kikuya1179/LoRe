"""Restricted DSL executor for safe LLM code execution."""

from __future__ import annotations

import ast
import json
import math
import operator
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ..utils.code_out import CodeOut


class DSLExecutor:
    """Safe DSL executor with restricted operations for Crafter environment."""
    
    # Allowed operators
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Lt: operator.lt,
        ast.Le: operator.le,
        ast.Gt: operator.gt,
        ast.Ge: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Allowed builtin functions
    ALLOWED_BUILTINS = {
        'abs': abs,
        'max': max,
        'min': min,
        'round': round,
        'sum': sum,
        'len': len,
        'int': int,
        'float': float,
        'bool': bool,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'sorted': sorted,
        'reversed': reversed,
    }
    
    # Math functions
    ALLOWED_MATH = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self, timeout_ms: int = 200):
        self.timeout_ms = timeout_ms
        self.execution_count = 0
        self.max_recursion = 100
    
    def execute(self, code: str, obs: np.ndarray, context: Dict[str, Any]) -> Optional[CodeOut]:
        """Execute DSL code safely and return CodeOut or None on failure."""
        try:
            start_time = time.time()
            
            # Parse AST
            tree = ast.parse(code, mode='eval')
            
            # Security check
            if not self._is_safe_ast(tree):
                return None
            
            # Create safe environment
            env = self._create_safe_environment(obs, context)
            
            # Execute with timeout
            result = self._execute_ast(tree.body, env, start_time)
            
            # Convert result to CodeOut
            return self._result_to_code_out(result)
            
        except Exception:
            return None
    
    def _is_safe_ast(self, tree: ast.AST) -> bool:
        """Check if AST only contains safe operations."""
        for node in ast.walk(tree):
            # Forbidden nodes
            if isinstance(node, (
                ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                ast.ClassDef, ast.Return, ast.Delete, ast.Assign, ast.AnnAssign,
                ast.AugAssign, ast.Raise, ast.Try, ast.With, ast.AsyncWith,
                ast.While, ast.For, ast.AsyncFor, ast.Global, ast.Nonlocal,
                ast.Exec, ast.Eval, ast.Lambda, ast.GeneratorExp, ast.ListComp,
                ast.SetComp, ast.DictComp, ast.comprehension, ast.ExceptHandler,
                ast.arguments, ast.arg, ast.keyword, ast.alias, ast.withitem,
            )):
                return False
            
            # Check function calls are allowed
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.ALLOWED_BUILTINS and node.func.id not in self.ALLOWED_MATH:
                        # Allow Crafter-specific functions
                        if not node.func.id.startswith(('count_', 'has_', 'distance_', 'goal_')):
                            return False
                elif isinstance(node.func, ast.Attribute):
                    # Only allow safe attribute access
                    if not self._is_safe_attribute(node.func.attr):
                        return False
                else:
                    return False
            
            # Check attribute access
            if isinstance(node, ast.Attribute):
                if not self._is_safe_attribute(node.attr):
                    return False
        
        return True
    
    def _is_safe_attribute(self, attr: str) -> bool:
        """Check if attribute access is safe."""
        # Allow basic array/dict operations
        safe_attrs = {
            'shape', 'size', 'ndim', 'dtype', 'mean', 'std', 'min', 'max', 'sum',
            'get', 'keys', 'values', 'items', 'append', 'extend', 'count', 'index',
        }
        return attr in safe_attrs
    
    def _create_safe_environment(self, obs: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create safe execution environment with Crafter-specific functions."""
        env = {
            # Builtins
            **self.ALLOWED_BUILTINS,
            **self.ALLOWED_MATH,
            
            # Observation data
            'obs': obs,
            'context': context,
            
            # Crafter-specific helper functions
            'count_pixels': lambda color: self._count_pixels(obs, color),
            'has_item': lambda item: context.get('inventory', {}).get(item, 0) > 0,
            'distance_to': lambda target: self._distance_to(obs, target),
            'goal_progress': lambda goal: context.get('goals', {}).get(goal, 0.0),
            'health_ratio': lambda: context.get('health', 100) / 100.0,
            'inventory_full': lambda: context.get('inventory_size', 0) >= context.get('max_inventory', 10),
            
            # Safe numpy-like operations
            'array': np.array,
            'zeros': np.zeros,
            'ones': np.ones,
            'mean': np.mean,
            'std': np.std,
            'clip': np.clip,
        }
        
        return env
    
    def _count_pixels(self, obs: np.ndarray, color: Union[int, List[int]]) -> int:
        """Count pixels of specific color in observation."""
        try:
            if isinstance(color, int):
                return int(np.sum(obs == color))
            elif isinstance(color, (list, tuple)):
                mask = np.ones(obs.shape[:2], dtype=bool)
                for i, c in enumerate(color):
                    if i < obs.shape[-1]:
                        mask &= (obs[..., i] == c)
                return int(np.sum(mask))
            return 0
        except Exception:
            return 0
    
    def _distance_to(self, obs: np.ndarray, target: str) -> float:
        """Compute distance to target (simplified heuristic)."""
        try:
            # This is a placeholder - in real Crafter, you'd have proper object detection
            h, w = obs.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Mock distance based on target type
            distance_map = {
                'wood': 0.3,
                'stone': 0.5,
                'water': 0.7,
                'iron': 0.8,
                'diamond': 0.9,
            }
            
            base_distance = distance_map.get(target, 0.5)
            # Add some noise based on observation
            noise = (np.mean(obs) / 255.0 - 0.5) * 0.1
            return max(0.0, min(1.0, base_distance + noise))
        except Exception:
            return 0.5
    
    def _execute_ast(self, node: ast.AST, env: Dict[str, Any], start_time: float) -> Any:
        """Execute AST node with timeout and recursion checks."""
        # Timeout check
        if (time.time() - start_time) * 1000 > self.timeout_ms:
            raise TimeoutError("Execution timeout")
        
        # Recursion check
        self.execution_count += 1
        if self.execution_count > self.max_recursion:
            raise RecursionError("Max recursion depth exceeded")
        
        try:
            return self._eval_node(node, env)
        finally:
            self.execution_count -= 1
    
    def _eval_node(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        """Evaluate individual AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        
        elif isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            else:
                raise NameError(f"Name '{node.id}' not defined")
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            op = self.ALLOWED_OPS.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            op = self.ALLOWED_OPS.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, env)
            result = True
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                right = self._eval_node(comparator, env)
                op_func = self.ALLOWED_OPS.get(type(op))
                if op_func:
                    result = result and op_func(left, right)
                    left = right
                else:
                    raise ValueError(f"Unsupported comparison: {type(op)}")
            return result
        
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(value, env) for value in node.values]
            op = self.ALLOWED_OPS.get(type(node.op))
            if op == operator.and_:
                return all(values)
            elif op == operator.or_:
                return any(values)
            else:
                raise ValueError(f"Unsupported boolean operator: {type(node.op)}")
        
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func, env)
            args = [self._eval_node(arg, env) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, env) for kw in node.keywords}
            return func(*args, **kwargs)
        
        elif isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value, env)
            return getattr(obj, node.attr)
        
        elif isinstance(node, ast.Subscript):
            obj = self._eval_node(node.value, env)
            slice_val = self._eval_node(node.slice, env)
            return obj[slice_val]
        
        elif isinstance(node, ast.List):
            return [self._eval_node(elt, env) for elt in node.elts]
        
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, env) for elt in node.elts)
        
        elif isinstance(node, ast.Dict):
            keys = [self._eval_node(k, env) for k in node.keys]
            values = [self._eval_node(v, env) for v in node.values]
            return dict(zip(keys, values))
        
        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, env)
            if test:
                return self._eval_node(node.body, env)
            else:
                return self._eval_node(node.orelse, env)
        
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    def _result_to_code_out(self, result: Any) -> Optional[CodeOut]:
        """Convert execution result to CodeOut structure."""
        try:
            if isinstance(result, dict):
                return CodeOut(
                    features=result.get('features', []),
                    subgoal=result.get('subgoal'),
                    r_shaped=float(result.get('r_shaped', 0.0)),
                    policy=result.get('policy'),
                    mask=result.get('mask'),
                    confidence=float(result.get('confidence', 0.5)),
                    notes=str(result.get('notes', '')),
                )
            elif isinstance(result, (int, float)):
                # Single numeric result treated as shaped reward
                return CodeOut(r_shaped=max(-0.2, min(0.2, float(result))))
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                # List treated as features
                features = [max(-3.0, min(3.0, float(x))) for x in result[:32]]
                return CodeOut(features=features)
            else:
                # Default empty result
                return CodeOut()
        
        except Exception:
            return None


# Example DSL expressions for Crafter:
EXAMPLE_DSL_CODE = '''
{
    "features": [
        count_pixels(64) / 100.0,  # Normalize wood pixel count
        distance_to("stone"),
        health_ratio() - 0.5,
        float(has_item("wood")),
    ],
    "subgoal": {"id": "collect_wood", "tau": 10} if not has_item("wood") else None,
    "r_shaped": 0.1 if has_item("wood") else -0.05,
    "policy": {"logits": [0.2, -0.1, 0.0, 0.1, -0.2] + [0.0] * 12},
    "mask": [1] * 17,  # All actions allowed
    "confidence": 0.8 if has_item("workbench") else 0.3,
    "notes": "Collecting resources for crafting"
}
'''