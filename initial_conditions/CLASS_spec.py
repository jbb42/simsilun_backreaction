from pathlib import Path
import subprocess, shutil, textwrap, re

# ---- config ----
here = Path(__file__).resolve().parent
run_dir = here / "data"
run_dir.mkdir(parents=True, exist_ok=True)

root_base   = run_dir / "class"   # CLASS will add 00_, 01_, ... to avoid overwrite
z_out       = 90
CLASS_EXE   = Path("../class_public/class")  # adjust if needed

# ---- write minimal INI for S-GenIC (CAMB layout, density transfer only) ----
ini = textwrap.dedent(f"""
h = 0.70
Omega_m = 0.3
Omega_l = 0.7
Omega_k = 0.0
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


# ---- clean up old CLASS output so it won't append 00_, 01_, ...
for f in run_dir.glob(f"{root_base.name}*"):
    if f.is_file():
        f.unlink()
    else:
        shutil.rmtree(f)

# ---- run CLASS CLI and capture stdout so we can parse the effective root ----
res = subprocess.run([str(CLASS_EXE), str(ini_path)],
                     cwd=run_dir, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError("CLASS CLI failed")
else:
    subprocess.run(
        ["mpiexec", "-np", "1", "./N-GenIC", "ngenic.param"],
        cwd="/home/jbb/Documents/simsilun_backreaction/initial_conditions/S-GenIC"
    )
