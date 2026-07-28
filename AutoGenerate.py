#!/usr/bin/env python3

"""
GRFT六口袋C2对称全新骨架RFdiffusion设计。

工作流程：
1. 从七个DENOVO_GAP_RANGES中分别采样一次；
2. 将同一组gap写入A/B链；
3. 每个设计单独调用一次RFdiffusion；
4. 固定六口袋motif；
5. 使用C2对称和olig_contacts势；
6. 记录随机种子、gap、contig及完整命令。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ======================================================
# 1. 六口袋固定motif片段
# ======================================================

FIXED_RANGES = (
    (8, 12),
    (24, 31),
    (42, 45),
    (64, 71),
    (88, 91),
    (106, 113),
)


# ======================================================
# 2. 全新骨架七个连接区的长度范围
# ======================================================

DENOVO_GAP_RANGES = (
    (3, 10),    # N端到8-12
    (6, 14),    # 8-12到24-31
    (5, 13),    # 24-31到42-45
    (10, 23),   # 42-45到64-71
    (8, 20),    # 64-71到88-91
    (7, 18),    # 88-91到106-113
    (4, 11),    # 106-113到C端
)


# 每条链固定motif总长度：37 aa
MOTIF_LENGTH = sum(
    end - start + 1
    for start, end in FIXED_RANGES
)


# ======================================================
# 3. 从范围中采样一组gap
# ======================================================

def sample_denovo_gaps(
    seed: int,
) -> tuple[int, ...]:
    """
    每个范围只采样一次。

    seed使用design_number，所以重新运行时
    相同design_number会得到相同gap。
    """

    rng = random.Random(seed)

    for _ in range(1000):
        gaps = tuple(
            rng.randint(low, high)
            for low, high
            in DENOVO_GAP_RANGES
        )

        chain_length = (
            MOTIF_LENGTH + sum(gaps)
        )

        # 每条链限制为85-146 aa
        if 85 <= chain_length <= 146:
            return gaps

    raise RuntimeError(
        f"Cannot sample valid gaps "
        f"for seed {seed}"
    )


# ======================================================
# 4. 构建单条链的contig
# ======================================================

def chain_contig(
    chain: str,
    gaps: tuple[int, ...],
) -> str:
    """
    例如：
    3/A8-12/11/A24-31/.../A106-113/4
    """

    if len(gaps) != 7:
        raise ValueError(
            "Exactly seven gaps are required"
        )

    pieces = [
        str(gaps[0])
    ]

    for index, (start, end) in enumerate(
        FIXED_RANGES
    ):
        # 固定输入PDB中的motif
        pieces.append(
            f"{chain}{start}-{end}"
        )

        # RFdiffusion生成的连接区
        pieces.append(
            str(gaps[index + 1])
        )

    return "/".join(pieces)


# ======================================================
# 5. 构建C2二聚体完整contig
# ======================================================

def full_contig(
    gaps: tuple[int, ...],
) -> str:
    """
    A/B链严格使用同一组gaps。

    /0加空格表示链断点。
    """

    chain_a = chain_contig(
        "A",
        gaps,
    )

    chain_b = chain_contig(
        "B",
        gaps,
    )

    return (
        f"{chain_a}"
        f"/0 "
        f"{chain_b}"
    )


# ======================================================
# 6. 运行单个RFdiffusion设计
# ======================================================

def run_one(
    args,
    design_number: int,
    checkpoint: Path,
):
    # 每个设计只采样一次
    gaps = sample_denovo_gaps(
        design_number
    )

    # A/B链共用这组gaps
    contig = full_contig(
        gaps
    )

    chain_length = (
        MOTIF_LENGTH + sum(gaps)
    )

    total_length = (
        chain_length * 2
    )

    prefix = (
        args.output_dir
        / (
            f"{args.label}_"
            f"denovo_"
            f"{checkpoint.stem}_"
            f"L{chain_length}"
        )
    )

    command = [
        sys.executable,

        str(
            args.clean_repo
            / "scripts"
            / "run_inference.py"
        ),

        "--config-name",
        "symmetry",

        # 输出设置
        f"inference.output_prefix={prefix}",
        "inference.num_designs=1",

        (
            "inference.design_startnum="
            f"{design_number}"
        ),

        # 可复现运行
        "inference.deterministic=True",

        # 不写入扩散轨迹
        "inference.write_trajectory=False",

        # 每个设计后清理GPU缓存
        "inference.empty_cache_per_design=True",

        # C2二聚体
        "inference.symmetry=c2",

        # 扩散步数
        f"diffuser.T={args.timesteps}",

        # 标准化六口袋motif
        (
            "inference.input_pdb="
            f"{args.motif_pdb}"
        ),

        # 模型目录
        (
            "inference.model_directory_path="
            f"{args.model_dir}"
        ),

        # ActiveSite模型
        (
            "inference.ckpt_override_path="
            f"{checkpoint}"
        ),

        # 已经采样为具体整数的C2 contig
        f"contigmap.contigs=[{contig}]",

        # 二聚体接触势
        (
            'potentials.guiding_potentials='
            '["type:olig_contacts,'
            'weight_intra:1,'
            'weight_inter:0.1"]'
        ),

        "potentials.olig_intra_all=True",
        "potentials.olig_inter_all=True",
        "potentials.guide_scale=2",
        "potentials.guide_decay=quadratic",
    ]

    # 输出日志文件
    log_path = Path(
        f"{prefix}_{design_number}.log"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()

    env["PYTHONPATH"] = str(
        args.clean_repo
    )

    started = datetime.now(
        timezone.utc
    )

    print("=" * 70)
    print(
        f"Design number : {design_number}"
    )
    print(
        f"Gaps          : {gaps}"
    )
    print(
        f"Chain length  : {chain_length}"
    )
    print(
        f"Total length  : {total_length}"
    )
    print(
        f"Contig        : {contig}"
    )
    print(
        f"Output prefix : {prefix}"
    )

    # 调用RFdiffusion
    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        process = subprocess.run(
            command,
            cwd=args.clean_repo,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    finished = datetime.now(
        timezone.utc
    )

    # 保存可复现运行记录
    record = {
        "route": "denovo",
        "checkpoint": checkpoint.name,
        "design_number": design_number,
        "random_seed": design_number,
        "chain_length": chain_length,
        "total_length": total_length,
        "gaps": list(gaps),
        "gap_ranges": [
            list(value)
            for value in DENOVO_GAP_RANGES
        ],
        "contig": contig,
        "output_prefix": str(prefix),
        "expected_pdb": (
            f"{prefix}_{design_number}.pdb"
        ),
        "expected_trb": (
            f"{prefix}_{design_number}.trb"
        ),
        "log": str(log_path),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (
            finished - started
        ).total_seconds(),
        "returncode": process.returncode,
        "command": command,
    }

    manifest_path = (
        args.output_dir
        / "run_manifest.jsonl"
    )

    with manifest_path.open(
        "a",
        encoding="utf-8",
    ) as manifest:
        manifest.write(
            json.dumps(record)
            + "\n"
        )

    if process.returncode != 0:
        print(
            f"RFdiffusion failed: "
            f"{design_number}"
        )
        print(
            f"See log: {log_path}"
        )

        raise subprocess.CalledProcessError(
            process.returncode,
            command,
        )

    print(
        f"Completed design "
        f"{design_number}"
    )


# ======================================================
# 7. 主程序
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate C2-symmetric GRFT "
            "six-pocket de novo scaffolds"
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="Number of designs",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=2000,
        help="Starting design number",
    )

    parser.add_argument(
        "--label",
        default="batch20",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--checkpoint",
        default="ActiveSite_ckpt.pt",
    )

    parser.add_argument(
        "--clean-repo",
        type=Path,
        default=Path(
            "/home/xzdxzdxzd/"
            "RFdiffusion_GRFT_clean"
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/home/xzdxzdxzd/"
            "RFdiffusion/models"
        ),
    )

    parser.add_argument(
        "--motif-pdb",
        type=Path,
        default=Path(
            "/home/xzdxzdxzd/"
            "RFdiffusion/outputs/"
            "grft_6pocket/reference/"
            "grft_c2_motif.pdb"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/xzdxzdxzd/"
            "RFdiffusion/outputs/"
            "grft_6pocket/"
            "route_denovo/raw"
        ),
    )

    args = parser.parse_args()

    # 检查RFdiffusion入口
    inference_script = (
        args.clean_repo
        / "scripts"
        / "run_inference.py"
    )

    if not inference_script.exists():
        raise FileNotFoundError(
            inference_script
        )

    # 检查motif PDB
    if not args.motif_pdb.exists():
        raise FileNotFoundError(
            args.motif_pdb
        )

    # 检查checkpoint
    checkpoint = (
        args.model_dir
        / args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating {args.count} "
        f"de novo C2 designs"
    )

    print(
        f"Design numbers: "
        f"{args.start}-"
        f"{args.start + args.count - 1}"
    )

    # 每次只生成一个设计，
    # 每个设计重新采样一组gap
    for offset in range(args.count):
        design_number = (
            args.start + offset
        )

        run_one(
            args=args,
            design_number=design_number,
            checkpoint=checkpoint,
        )

    print()
    print(
        "All RFdiffusion designs completed"
    )
    print(
        f"Output directory: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
