import re

def update_params(filename, updates):
    """Update given parameters in a .param file."""
    with open(filename) as f:
        lines = f.readlines()

    for key, new_value in updates.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        found = False

        for i, line in enumerate(lines):
            if pattern.match(line):
                # Keep comment if there is one
                parts = line.split("#", 1)
                comment = ("#" + parts[1]) if len(parts) > 1 else ""
                lines[i] = f"{key} = {new_value} {comment}\n"
                found = True
                break

        if not found:
            # Append new key-value if not in file
            lines.append(f"{key} = {new_value}\n")

    with open(filename, "w") as f:
        f.writelines(lines)

    print(f"Updated {len(updates)} parameters in {filename}.")

def convert_value(val):
    """Convert a string to int or float if possible."""
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        else:
            return int(val)
    except ValueError:
        return val.strip()

def read_params(filename):
    params = {}
    with open(filename) as f:
        for line in f:
            # Remove inline comments
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            # Split key = value
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = convert_value(val.strip())
                params[key] = val
    print("\nParameters read from", filename)
    for key, val in params.items():
        print(f"  {key:20s} = {val}")
    return params

