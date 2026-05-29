#!/usr/bin/env python3
"""
Phage replication driver.

Reads a YAML config, then submits the full pipeline as a chain of SLURM jobs
linked by --dependency=afterok:

  1. finetune    one job per (architecture, seed)
  2. embedding   one job per architecture                 ┐ submitted in parallel
                                                          ┘ with stage 1
  3. select      one job — picks winner per architecture, blocked on all of 1+2
  4. inference   one job per (architecture, diagnostic dataset), blocked on 3
  5. genome-wide one job per architecture for the genome-wide CSV, blocked on 3
  6. genome      one job per architecture, blocked on stage 5 for that arch
     analysis

Usage:
    python replicate_phage.py --config configs/phage_replication.yaml
    python replicate_phage.py --config configs/phage_replication.yaml --dry-run
    python replicate_phage.py --config configs/phage_replication.yaml \\
        --stages finetune,embedding              # submit only those stages
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ALL_STAGES = ["finetune", "embedding", "select", "inference", "genome_wide", "genome_analysis"]
REPL_DIR = Path(__file__).resolve().parent / "slurm_scripts" / "replication"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = ["lambda_dataset_dir", "output_dir", "diagnostic_datasets", "genome_wide_dataset"]
    missing = [k for k in required if k not in cfg or cfg[k] is None]
    if missing:
        sys.exit(f"ERROR: config missing required keys: {missing}")

    # Reject placeholder paths so the user can't accidentally submit jobs against /path/to/...
    placeholders = []
    for key in ("lambda_dataset_dir", "output_dir", "genome_wide_dataset"):
        if str(cfg[key]).startswith("/path/to/"):
            placeholders.append(key)
    for name, path in cfg["diagnostic_datasets"].items():
        if str(path).startswith("/path/to/"):
            placeholders.append(f"diagnostic_datasets.{name}")
    if placeholders:
        sys.exit(f"ERROR: edit the config — placeholder paths still present: {placeholders}")

    cfg.setdefault("architectures", ["prokbert-mini", "prokbert-mini-c", "prokbert-mini-long"])
    cfg.setdefault("seeds", [1, 2, 3])
    cfg.setdefault("finetune", {})
    cfg.setdefault("embedding", {})
    cfg.setdefault("slurm", {})
    return cfg


def sbatch_args(slurm_cfg, stage_cfg, job_name, dependency=None, log_dir=None):
    """Build the leading sbatch flags shared by every stage."""
    args = ["sbatch", f"--job-name={job_name}"]

    partition = stage_cfg.get("partition", slurm_cfg.get("partition", "gpu"))
    args.append(f"--partition={partition}")

    gpu = stage_cfg.get("gpu", slurm_cfg.get("gpu", "a100:1"))
    if gpu:  # empty string = CPU-only job
        args.append(f"--gres=gpu:{gpu}")

    args.append(f"--mem={stage_cfg.get('mem', '32g')}")
    args.append(f"--time={stage_cfg.get('time', '4:00:00')}")
    args.append(f"--cpus-per-task={stage_cfg.get('cpus', 8)}")

    if log_dir:
        args.append(f"--output={log_dir}/{job_name}_%j.out")
        args.append(f"--error={log_dir}/{job_name}_%j.err")

    if dependency:
        dep_ids = ":".join(str(d) for d in dependency)
        args.append(f"--dependency=afterok:{dep_ids}")

    return args


def submit(sbatch_argv, env_vars, script_path, dry_run, label=""):
    """Run sbatch (or print the equivalent command) and return the job ID."""
    export_pairs = ",".join(f"{k}={v}" for k, v in env_vars.items() if v != "")
    full = sbatch_argv + [f"--export=ALL,{export_pairs}", str(script_path)]

    if dry_run:
        print(f"  [DRY-RUN] {label or script_path.name}")
        print("    " + " ".join(shlex.quote(a) for a in full))
        # Return a synthetic ID so downstream --dependency strings still render.
        return f"<{label or 'job'}>"

    print(f"  submitting {label or script_path.name}... ", end="", flush=True)
    try:
        out = subprocess.check_output(full, text=True).strip()
    except subprocess.CalledProcessError as e:
        print("FAILED")
        sys.exit(f"sbatch failed (rc={e.returncode}): {e.output}")
    # sbatch prints "Submitted batch job 12345"
    job_id = out.split()[-1]
    print(f"job {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to phage_replication.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--stages", default=",".join(ALL_STAGES),
                        help=f"Comma-separated subset of stages to submit. "
                             f"Options: {','.join(ALL_STAGES)}")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        sys.exit(f"ERROR: unknown stages {unknown}; valid: {ALL_STAGES}")

    output_dir = os.path.abspath(cfg["output_dir"])
    lambda_dir = os.path.abspath(cfg["lambda_dataset_dir"])
    genome_csv = os.path.abspath(cfg["genome_wide_dataset"])
    diagnostics = {name: os.path.abspath(p) for name, p in cfg["diagnostic_datasets"].items()}

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    arches = cfg["architectures"]
    seeds = cfg["seeds"]
    slurm = cfg["slurm"]

    print(f"Config:        {args.config}")
    print(f"Output dir:    {output_dir}")
    print(f"Lambda dir:    {lambda_dir}")
    print(f"Architectures: {arches}")
    print(f"Seeds:         {seeds}")
    print(f"Diagnostics:   {list(diagnostics.keys())}")
    print(f"Stages:        {stages}")
    print()

    # --- Stage 1: finetune (arch × seed) -------------------------------------
    finetune_ids = []
    if "finetune" in stages:
        print("Stage 1: finetune")
        ft_cfg = slurm.get("finetune", {})
        ft_hp = cfg["finetune"]
        for arch in arches:
            for seed in seeds:
                env = {
                    "REPL_OUTPUT_DIR": output_dir,
                    "LAMBDA_DIR": lambda_dir,
                    "ARCH": arch,
                    "SEED": str(seed),
                    "LR": str(ft_hp.get("learning_rate", "1e-4")),
                    "BATCH_SIZE": str(ft_hp.get("batch_size", 32)),
                    "NUM_EPOCHS": str(ft_hp.get("num_epochs", 3)),
                    "EARLY_STOPPING_PATIENCE": str(ft_hp.get("early_stopping_patience", 3)),
                    "USE_FP16": "1" if ft_hp.get("fp16", True) else "0",
                }
                job_name = f"ft_{arch}_s{seed}"
                sb = sbatch_args(slurm, ft_cfg, job_name, log_dir=log_dir)
                jid = submit(sb, env, REPL_DIR / "finetune.sh", args.dry_run, label=job_name)
                finetune_ids.append(jid)

    # --- Stage 2: embedding analysis (one per arch) --------------------------
    embedding_ids = []
    if "embedding" in stages:
        print("Stage 2: embedding analysis")
        emb_cfg = slurm.get("embedding", {})
        emb_hp = cfg["embedding"]
        for arch in arches:
            env = {
                "REPL_OUTPUT_DIR": output_dir,
                "LAMBDA_DIR": lambda_dir,
                "ARCH": arch,
                "POOLING": str(emb_hp.get("pooling", "mean")),
                "NN_EPOCHS": str(emb_hp.get("nn_epochs", 100)),
                "NN_LR": str(emb_hp.get("nn_lr", 0.001)),
            }
            job_name = f"emb_{arch}"
            sb = sbatch_args(slurm, emb_cfg, job_name, log_dir=log_dir)
            jid = submit(sb, env, REPL_DIR / "embedding.sh", args.dry_run, label=job_name)
            embedding_ids.append(jid)

    # --- Stage 3: select winner per arch -------------------------------------
    select_id = None
    if "select" in stages:
        print("Stage 3: select_best")
        # Tiny CPU job — reuse embedding's partition/mem defaults but force CPU.
        sel_cfg = {"mem": "4g", "time": "00:15:00", "cpus": 2,
                   "partition": "norm", "gpu": ""}
        deps = finetune_ids + embedding_ids
        # When running stages independently (e.g. --stages select), no
        # dependencies exist yet — only chain if we submitted predecessors.
        dep = deps if deps and not args.dry_run else (deps or None)
        env = {
            "REPL_OUTPUT_DIR": output_dir,
            # Colon-separated so the value has no spaces (SLURM --export parsing
            # uses commas; spaces inside values work most places but not all).
            "ARCHITECTURES": ":".join(arches),
        }
        sb = sbatch_args(slurm, sel_cfg, "repl_select",
                         dependency=dep, log_dir=log_dir)
        select_id = submit(sb, env, REPL_DIR / "select.sh", args.dry_run, label="select")

    # --- Stage 4: diagnostic inference (arch × dataset) ----------------------
    diagnostic_ids = []
    if "inference" in stages:
        print("Stage 4: diagnostic inference")
        inf_cfg = slurm.get("inference", {})
        dep = [select_id] if select_id else None
        for arch in arches:
            for name, csv_path in diagnostics.items():
                env = {
                    "REPL_OUTPUT_DIR": output_dir,
                    "ARCH": arch,
                    "INPUT_CSV": csv_path,
                    "OUTPUT_FILENAME": f"{name}_predictions.csv",
                }
                job_name = f"inf_{arch}_{name}"
                sb = sbatch_args(slurm, inf_cfg, job_name,
                                 dependency=dep, log_dir=log_dir)
                jid = submit(sb, env, REPL_DIR / "inference.sh", args.dry_run, label=job_name)
                diagnostic_ids.append(jid)

    # --- Stage 5: genome-wide inference (one per arch) -----------------------
    genome_inf_ids = {}  # arch -> job id
    if "genome_wide" in stages:
        print("Stage 5: genome-wide inference")
        inf_cfg = slurm.get("inference", {})
        dep = [select_id] if select_id else None
        for arch in arches:
            env = {
                "REPL_OUTPUT_DIR": output_dir,
                "ARCH": arch,
                "INPUT_CSV": genome_csv,
                "OUTPUT_FILENAME": f"genome_wide_predictions.csv",
            }
            job_name = f"gwinf_{arch}"
            sb = sbatch_args(slurm, inf_cfg, job_name,
                             dependency=dep, log_dir=log_dir)
            jid = submit(sb, env, REPL_DIR / "inference.sh", args.dry_run, label=job_name)
            genome_inf_ids[arch] = jid

    # --- Stage 6: genome-wide threshold + clustering sweep -------------------
    if "genome_analysis" in stages:
        print("Stage 6: genome-wide analysis")
        ga_cfg = slurm.get("genome_analysis", {"partition": "norm", "gpu": "",
                                               "mem": "16g", "time": "2:00:00", "cpus": 4})
        for arch in arches:
            dep = [genome_inf_ids[arch]] if arch in genome_inf_ids else None
            env = {"REPL_OUTPUT_DIR": output_dir, "ARCH": arch}
            job_name = f"gwana_{arch}"
            sb = sbatch_args(slurm, ga_cfg, job_name,
                             dependency=dep, log_dir=log_dir)
            submit(sb, env, REPL_DIR / "genome_analysis.sh", args.dry_run, label=job_name)

    print()
    print("Done." if not args.dry_run else "Dry-run done.")
    print(f"Monitor with: squeue -u $USER")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
