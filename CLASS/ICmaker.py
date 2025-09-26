from pathlib import Path
import subprocess, shutil, textwrap, re

# ---- config ----
here = Path(__file__).resolve().parent
run_dir = here / "data"
run_dir.mkdir(parents=True, exist_ok=True)

root_base   = run_dir / "cambv2_"   # CLASS will add 00_, 01_, ... to avoid overwrite
z_out       = 99
CLASS_EXE   = Path("/home/jbb/Downloads/class_public/class")  # adjust if needed

# ---- write minimal INI for S-GenIC (CAMB layout, density transfer only) ----
ini = textwrap.dedent(f"""
h = 0.67
omega_b = 0.02237
omega_cdm = 0.1200
A_s = 2.1e-9
n_s = 0.965
tau_reio = 0.054

output = mPk,dTk
z_pk = {z_out}
P_k_max_h/Mpc = 100

format = camb           # CAMB text layout
headers = no
root = {root_base.as_posix()}

input_verbose = 1
output_verbose = 1
""").strip() + "\n"

ini_path = run_dir / f"write_camb_{z_out}.ini"
ini_path.write_text(ini)

# ---- run CLASS CLI and capture stdout so we can parse the effective root ----
res = subprocess.run([str(CLASS_EXE), str(ini_path)],
                     cwd=run_dir, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError("CLASS CLI failed")