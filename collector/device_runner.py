import subprocess
import tempfile
import os
import logging
from analyzer.ast_parser import parse_and_detect

log = logging.getLogger("device_runner")
log.setLevel(logging.INFO)


def compile_and_run(cmd_compile, exe_path="./a.out"):
    try:
        subprocess.run(cmd_compile, check=True)
        subprocess.run([exe_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Execution failed: {e}")
        return False


def run_on_device(source_file: str, device: str, extra_info: dict = None):
    """
    device: "cpu" or "gpu"
    extra_info: parsed metadata from AST (operation_type, primary_function_name, params, has_main)
    """
    metadata = extra_info or parse_and_detect(source_file)
    has_main = metadata.get("has_main", False)

    if source_file.endswith(".py"):
        log.info(f"Python file detected. Predicted device={device}. Running with python3.")
        cmd = ["python3", source_file]
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Python execution failed: {e}")
            return False

    if device == "gpu":
        cu_file = source_file.replace(".c", ".cu")
        if os.path.exists(cu_file):
            
            # cmd = ["nvcc", cu_file, "-o", "a.out"]
            cmd = ["nvcc", cu_file, "-O3","-use_fast_math","-o","a.out"]

            log.info(f"Compiling (GPU) with: {' '.join(cmd)}")
            return compile_and_run(cmd)
        else:
            log.warning(f"GPU predicted but no CUDA file found for {source_file}. Falling back to CPU.")
            device = "cpu"

    if has_main:
       # cmd = ["/usr/bin/gcc", source_file, "-O3", "-o", "a.out"]
        cmd = ["/usr/bin/gcc", source_file, "-O3","-march=native","-funroll-loops","-ffast-math","-o","a.out"]

        log.info(f"Compiling (CPU) with: {' '.join(cmd)}")
        ok = compile_and_run(cmd)
        if not ok:
            log.error("Failed to compile/run with main present.")
        return ok
    else:
        fn = metadata.get("primary_function_name")
        params = metadata.get("primary_function_params", [])
        log.info(f"No main found. Generating generic wrapper for fn={fn} params={params}")

        wrapper_code = generate_generic_wrapper(fn, params)
        if wrapper_code is None:
            log.error("Could not generate wrapper. Please provide a main in source.")
            return False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as wf:
            wf.write(wrapper_code)
            wrapper_path = wf.name

       # cmd = ["/usr/bin/gcc", source_file, wrapper_path, "-O3", "-o", "a.out"]
        cmd = ["/usr/bin/gcc", source_file, "-O3","-march=native","-funroll-loops","-ffast-math","-o","a.out"]
        log.info(f"Compiling wrapper (CPU) with: {' '.join(cmd)}")
        ok = compile_and_run(cmd)

        try:
            os.unlink(wrapper_path)
        except Exception:
            pass
        return ok

def generate_generic_wrapper(fn_name, params):
    """
    Generate a generic wrapper main() for any function with known params.
    - fn_name: function name (string)
    - params: list of (param_type, param_name, dimensions) extracted by AST
      Example: [("float", "A", [32,32]), ("float", "B", [32,32]), ("float", "C", [32,32])]
    """
    if not fn_name:
        return None

    decls = []
    inits = []
    args = []

    for ptype, pname, dims in params:
        if not dims:  
            decls.append(f"{ptype} {pname} = 1;")
            args.append(pname)
        else:  
            dim_str = "".join([f"[{d}]" for d in dims])
            decls.append(f"static {ptype} {pname}{dim_str};")
            idx = "][".join([f"i{d}" for d in range(len(dims))])
            loops = "".join([f"for(int i{d}=0;i{d}<{dims[d]};i{d}++){{" for d in range(len(dims))])
            close = "}" * len(dims)
            inits.append(f"""{loops} {pname}[{idx}] = 1; {close}""")
            args.append(pname)

    wrapper = f"""
#include <stdio.h>
#include <stdlib.h>
extern void {fn_name}({', '.join([f'{ptype} {pname}' + ''.join([f'[{d}]' for d in dims]) for ptype, pname, dims in params])});

int main() {{
    {"".join(decls)}
    {"".join(inits)}
    {fn_name}({", ".join(args)});
    printf("wrapper: {fn_name} called\\n");
    return 0;
}}
"""
    return wrapper
