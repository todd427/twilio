#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

import toml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml

def sha_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def detect_version(repo_root: Path) -> str:
    try:
        import subprocess
        v = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        return v
    except Exception:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return f"0.0.0+{sha_short(str(repo_root))}-{ts}"

def load_toml(path: Path) -> dict:
    return toml.load(path.open("r", encoding="utf-8"))

def load_ignore(ignore_file: Path) -> list[str]:
    if not ignore_file.exists(): return []
    lines = [ln.strip() for ln in ignore_file.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def should_ignore(rel_path: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch
    rel_path_norm = rel_path.replace("\\","/")
    return any(
        fnmatch(rel_path_norm, pat)
        or fnmatch(rel_path_norm.rstrip("/") + "/**", pat)  # directory patterns
        for pat in patterns
    )

def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def copy_tree(src: Path, dst: Path, ignore_patterns: list[str], dry: bool, verbose: bool):
    """
    Copy src -> dst while honoring ignore patterns.
    Also make sure we never walk into dst if dst is inside src.
    """
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        # hard stop: don't descend into dst if it's inside src
        if is_within(root_p, dst):
            # Should not happen because dst is under src; but be defensive
            if verbose:
                print(f"✗ skip subtree (output): {root_p.relative_to(src)}")
            dirs[:] = []
            continue

        rpath = root_p.relative_to(src)

        # prune dirs by ignore and by dst containment
        keep_dirs = []
        for d in dirs:
            abs_dir = root_p / d
            rel_dir = (rpath / d).as_posix()
            if is_within(abs_dir, dst) or should_ignore(rel_dir, ignore_patterns) or should_ignore(rel_dir + "/**", ignore_patterns):
                if verbose: print(f"✗ skip dir: {rel_dir}")
                continue
            keep_dirs.append(d)
        dirs[:] = keep_dirs

        # files
        for f in files:
            rel = (rpath / f).as_posix()
            if should_ignore(rel, ignore_patterns):
                if verbose: print(f"✗ skip file: {rel}")
                continue
            src_f = src / rpath / f
            dst_f = dst / rpath / f
            if verbose: print(f"→ copy {rel}")
            if not dry:
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst_f)

def render_templates(src_root: Path, out_root: Path, files: list[str], vars: dict, dry: bool, verbose: bool):
    env = Environment(loader=FileSystemLoader(str(src_root)), undefined=StrictUndefined, autoescape=False)
    for rel in files:
        tpl_path = src_root / rel
        if not tpl_path.exists():
            if verbose: print(f"⚠ missing template: {rel}")
            continue
        if verbose: print(f"✎ render {rel}")
        template = env.get_template(rel)
        out = template.render(**vars)
        out_file = out_root / rel
        if not dry:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(out, encoding="utf-8")

def write_runtime_config(out_root: Path, config_rel: str, vars: dict, dry: bool, verbose: bool):
    cfg_path = out_root / config_rel
    if verbose: print(f"⚙ write {config_rel}")
    if not dry:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(vars, indent=2), encoding="utf-8")

def write_env_example(out_root: Path, env_rel: str, vars: dict, dry: bool, verbose: bool):
    lines = []
    for k, v in vars.items():
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"{k}={v}")
    content = "\n".join(lines) + "\n"
    path = out_root / env_rel
    if verbose: print(f"🔑 write {env_rel}")
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Ship toddric web copy")
    ap.add_argument("--src", default=".", help="Source repo root")
    ap.add_argument("--out", default="./dist/web", help="Output web folder")
    ap.add_argument("--config", default="./ship.toml", help="ship config toml")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    ap.add_argument("--version", action="store_true", help="Print detected version and exit")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if args.version:
        print(detect_version(src))
        return 0

    cfg = load_toml(Path(args.config))

    ignore_file = src / cfg["source"]["ignore_file"]
    patterns = load_ignore(ignore_file)

    # *** Critical fix: always ignore the output folder if it's inside the source ***
    if is_within(out, src):
        out_rel = out.relative_to(src).as_posix()
        # ignore both the directory and everything under it
        patterns.extend([out_rel, out_rel + "/**"])

    version = detect_version(src)
    vars = dict(cfg.get("vars", {}))
    vars.update({
        "VERSION": version,
        "BUILD_EPOCH": int(time.time()),
        "BUILD_ISO": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    if args.verbose:
        print(f"SRC : {src}")
        print(f"OUT : {out}")
        print(f"VER : {version}")
        print(f"IGN : {ignore_file} ({len(patterns)} patterns)")
        if is_within(out, src):
            print(f"    + auto-ignored output path: {out.relative_to(src)}")

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    copy_tree(src, out, patterns, args.dry_run, args.verbose)

    tpl_cfg = cfg.get("template", {})
    if tpl_cfg.get("enabled", False):
        files = tpl_cfg.get("files", [])
        render_templates(src, out, files, vars, args.dry_run, args.verbose)

    write_runtime_config(out, cfg["output"]["config_json"], vars, args.dry_run, args.verbose)
    write_env_example(out, cfg["output"]["env_example"], vars, args.dry_run, args.verbose)

    if args.verbose:
        print("✅ done")
    return 0

if __name__ == "__main__":
    sys.exit(main())

