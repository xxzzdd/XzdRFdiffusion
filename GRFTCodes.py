# 保守结构口袋间隙长度
gaps = (7, 11, 10, 18, 16, 14, 8)

contig = (
    "["
    "7/A8-12/11/A24-31/10/A42-45/18/"
    "A64-71/16/A88-91/14/A106-113/8"
    "/0 "
    "7/B8-12/11/B24-31/10/B42-45/18/"
    "B64-71/16/B88-91/14/B106-113/8"
    "]"
)

# 骨架生成
cmd = [
    "python", "scripts/run_inference.py",
    "--config-name", "symmetry",

    f"inference.input_pdb={motif_pdb}",
    f"inference.output_prefix={output_prefix}",

    "inference.symmetry=c2",
    "inference.num_designs=1",
    "inference.deterministic=True",
#功能活性位点口袋保持活性
    "inference.ckpt_override_path="
        "models/ActiveSite_ckpt.pt",

    f"diffuser.T={timesteps}",
    f"contigmap.contigs={contig}",
]
