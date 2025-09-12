import os
import math
import logging
from typing import Dict, List, Optional
import clang.cindex as cindex
from clang.cindex import CursorKind, TypeKind
import ast   

log = logging.getLogger("ast_parser")
log.setLevel(logging.INFO)

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

class ASTParser:
    def __init__(self):
        self.index = cindex.Index.create()
        self.function_patterns = {
            'matmul': ['matmul', 'matrix_multiply', 'sgemm'],
            'fft': ['fft', 'dft', 'fftw', 'cufft'],
            'sort': ['sort', 'qsort', 'mergesort', 'quicksort'],
            'conv2d': ['conv2d', 'convolution'],
            'transpose': ['transpose'],
            'scan': ['scan', 'prefix_sum']
        }
        self.type_sizes = {'float': 4, 'double': 8, 'int': 4, 'char': 1, 'pointer': 8}

   def parse_file(self, path: str, include_paths: Optional[List[str]] = None) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            return self._parse_python(path)
        else:
            return self._parse_c_cpp(path, include_paths)

   def _parse_c_cpp(self, path: str, include_paths: Optional[List[str]]) -> Dict:
        args = ['-std=c11']
        if include_paths:
            for p in include_paths:
                args.extend(['-I', p])
        tu = self.index.parse(path, args=args)
        if not tu:
            raise RuntimeError("clang failed to parse")
        return self._extract_features(tu.cursor)

    def _extract_features(self, root) -> Dict:
        analysis = {
            'function_calls': [],
            'nested_loop_depth': 0,
            'loop_bounds': [],
            'has_reduction': False,
            'has_transpose_pattern': False,
            'has_elementwise': False,
            'declared_memory_bytes': 0,
            'array_dimensions': [],
            'has_main': False,
            'exported_functions': []
        }
        self._traverse(root, analysis, loop_depth=0)
        op_type = self._classify(analysis)
        total_bytes = self._estimate_total_bytes(analysis)
        log_total = math.log10(max(total_bytes, 1))

        primary_fn = analysis['exported_functions'][0][0] if analysis['exported_functions'] else None
        primary_params = analysis['exported_functions'][0][1] if analysis['exported_functions'] else []

        log.info(
            f"[C/C++] Detected op={op_type} log_size={round(log_total,4)} "
            f"(total_bytes={total_bytes}) has_main={analysis['has_main']} fn={primary_fn}"
        )

        return {
            'operation_type': op_type,
            'log_total_sizes': round(log_total, 4),
            'has_main': analysis['has_main'],
            'primary_function_name': primary_fn,
            'primary_function_params': primary_params,
            'total_memory_bytes': total_bytes,
            'raw_analysis': analysis
        }

    def _traverse(self, cursor, analysis, loop_depth=0):
        if cursor.kind == CursorKind.FUNCTION_DECL:
            name = cursor.spelling or cursor.displayname
            if name == 'main':
                analysis['has_main'] = True
            params = [(p.spelling, p.type.spelling) for p in cursor.get_arguments()]
            if cursor.is_definition() and name != 'main':
                analysis['exported_functions'].append((name, params))

        if cursor.kind == CursorKind.FOR_STMT:
            new_depth = loop_depth + 1
            analysis['nested_loop_depth'] = max(analysis['nested_loop_depth'], new_depth)
            bound = self._extract_loop_constant_bound(cursor)
            if bound:
                analysis['loop_bounds'].append(bound)
            if self._check_reduction_in_loop(cursor):
                analysis['has_reduction'] = True
            if self._check_elementwise_in_loop(cursor):
                analysis['has_elementwise'] = True

        if cursor.kind == CursorKind.CALL_EXPR:
            fname = cursor.spelling or cursor.displayname or "".join([t.spelling for t in cursor.get_tokens()][:1])
            analysis['function_calls'].append(fname.lower())

        if cursor.kind in (CursorKind.VAR_DECL, CursorKind.PARM_DECL):
            try:
                t = cursor.type
                if t.kind == TypeKind.CONSTANTARRAY:
                    bytes_size = self._type_constantarray_bytes(t)
                    analysis['declared_memory_bytes'] += bytes_size
                    dims = self._get_array_dims_from_type(t)
                    if dims:
                        analysis['array_dimensions'].append(dims)
            except Exception:
                pass

        if cursor.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            if self._is_transpose_access(cursor):
                analysis['has_transpose_pattern'] = True

        for ch in cursor.get_children():
            self._traverse(ch, analysis, loop_depth + 1 if cursor.kind == CursorKind.FOR_STMT else loop_depth)

    def _extract_loop_constant_bound(self, for_node):
        for child in for_node.get_children():
            if child.kind == CursorKind.BINARY_OPERATOR:
                tokens = list(child.get_tokens())
                for t in tokens:
                    if t.spelling.isdigit():
                        return int(t.spelling)
        return None

    def _type_constantarray_bytes(self, t):
        if t.kind == TypeKind.CONSTANTARRAY:
            arr_count = t.get_array_size()
            elem_type = t.get_array_element_type()
            return arr_count * self._type_constantarray_bytes(elem_type)
        return self._get_type_size(t)

    def _get_array_dims_from_type(self, t):
        dims = []
        while t.kind == TypeKind.CONSTANTARRAY:
            dims.append(t.get_array_size())
            t = t.get_array_element_type()
        return dims

    def _get_type_size(self, t):
        s = t.spelling.lower()
        for k, v in self.type_sizes.items():
            if k in s:
                return v
        if t.kind == TypeKind.POINTER:
            return self.type_sizes.get('pointer', 8)
        return 4

    def _is_transpose_access(self, cursor):
        idxs = []
        cur = cursor
        while cur and cur.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(cur.get_children())
            if len(children) >= 2:
                tokens = [t.spelling for t in children[1].get_tokens()]
                idxs.extend(tokens)
                cur = children[0]
            else:
                break
        return ('i' in idxs and 'j' in idxs) or ('j' in idxs and 'i' in idxs)

    def _check_reduction_in_loop(self, node):
        for c in node.get_children():
            if c.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
                return True
            if c.kind == CursorKind.BINARY_OPERATOR:
                toks = [t.spelling for t in c.get_tokens()]
                if len(toks) >= 3 and toks[1] == '=' and toks[0] == toks[2]:
                    return True
        return False

    def _has_simple_array_access(self, node):
        for c in node.get_children():
            if c.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
                return True
            if self._has_simple_array_access(c):
                return True
        return False

    def _check_elementwise_in_loop(self, node):
        for c in node.get_children():
            if c.kind == CursorKind.BINARY_OPERATOR and self._has_simple_array_access(c):
                return True
        return False

    def _estimate_total_bytes(self, analysis) -> int:
        declared = analysis['declared_memory_bytes']
        if analysis['loop_bounds']:
            est_elements = 1
            for b in analysis['loop_bounds']:
                est_elements *= max(1, b)
            est_bytes = est_elements * 4
        else:
            est_bytes = 0
        return max(declared, est_bytes)

    def _classify(self, analysis):
        for call in analysis['function_calls']:
            for op, pats in self.function_patterns.items():
                if any(p in call for p in pats):
                    return op
        if analysis['nested_loop_depth'] >= 3:
            return 'matmul'
        if analysis['nested_loop_depth'] == 1 and analysis['has_reduction']:
            return 'reduce'
        if analysis['has_transpose_pattern']:
            return 'transpose'
        if any(len(d) >= 2 for d in analysis['array_dimensions']) and analysis['nested_loop_depth'] >= 2:
            return 'conv2d'
        if analysis['nested_loop_depth'] == 1 and analysis['has_elementwise']:
            return 'elementwise'
        return 'unknown'

    def _parse_python(self, path: str) -> Dict:
        import ast, math

        with open(path, "r") as f:
            source = f.read()
        tree = ast.parse(source, filename=path)

        funcs, calls = [], []
        loop_bounds = []

        def extract_range_bound(node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                    return int(node.args[0].value)
                if len(node.args) == 2 and all(isinstance(a, ast.Constant) for a in node.args):
                    return int(node.args[1].value) - int(node.args[0].value)
            return 1

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs.append(node.name.lower())
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id.lower())
            elif isinstance(node, ast.For):
                bound = 1
                if isinstance(node.iter, ast.Call):
                    bound = extract_range_bound(node.iter)
                loop_bounds.append(bound)

        total_iters = 1
        for b in loop_bounds:
            total_iters *= max(1, b)

        op_type = "unknown"
        for op, pats in self.function_patterns.items():
            if any(pat in name for name in funcs + calls for pat in pats):
                op_type = op
                break
        if op_type == "unknown" and len(loop_bounds) >= 3:
            op_type = "matmul"

        log_size = round(math.log10(total_iters + 1), 4)

        log.info(f"[Python] Detected op={op_type}, loop_bounds={loop_bounds}, total_iters={total_iters}, log_size={log_size}")

        return {
            "operation_type": op_type,
            "log_total_sizes": log_size,
            "has_main": "main" in funcs,
            "primary_function_name": funcs[0] if funcs else None,
            "primary_function_params": [],
            "total_memory_bytes": total_iters * 4,
            "raw_analysis": {
                "functions": funcs,
                "function_calls": calls,
                "loop_bounds": loop_bounds,
                "nested_loop_depth": len(loop_bounds)
            }
        }

def parse_and_detect(path):
    return ASTParser().parse_file(path)
