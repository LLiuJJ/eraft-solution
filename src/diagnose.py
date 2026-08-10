"""诊断 z 塌缩：检查正常/异常样本得分区分度"""
import torch, sys
from tqdm import tqdm
sys.path.insert(0, ".")
from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.inpformer import INPFormer

ckpt = torch.load("checkpoints/all/best.pth", map_location="cpu", weights_only=False)
cfg = ckpt.get("config", get_config())
cfg.data.image_size = 518
cfg.data.batch_size = 4
cfg.data.num_workers = 4
cfg.dinov2.weights_path = "weights/dinov2_vitb14_pretrain.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dinov2 = DINOv2Extractor(
    model_name=cfg.dinov2.model_name,
    out_indices=cfg.dinov2.out_indices,
    patch_size=cfg.dinov2.patch_size,
    frozen=True,
    weights_path=cfg.dinov2.weights_path,
).to(device)

model = INPFormer(
    dinov2_dim=cfg.dinov2.embed_dim,
    d_model=cfg.inpformer.d_model,
    n_heads=cfg.inpformer.n_heads,
    n_layers=cfg.inpformer.n_layers,
    dim_ff=cfg.inpformer.dim_ff,
    dropout=0.0,
    num_views=cfg.data.num_views,
    n_flow_layers=cfg.inpformer.n_flow_layers,
    coupling_hidden=cfg.inpformer.coupling_hidden,
    score_type=cfg.inpformer.score_type,
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

loader = build_dataloader(cfg, split="Test_A")
all_z_cls, all_z_view, all_knn = [], [], []

with torch.no_grad():
    for i, batch in enumerate(tqdm(loader, desc="诊断")):
        if i >= 20:
            break
        views = batch["views"].to(device)
        feats = dinov2.extract_multi_view(views)
        out = model(feats)

        z_cls = torch.norm(out["z_cls"], dim=-1)        # [B]
        z_view = torch.norm(out["z_view"], dim=-1).mean(-1)  # [B]
        all_z_cls.extend(z_cls.cpu().tolist())
        all_z_view.extend(z_view.cpu().tolist())

        # k-NN patch score
        from src.submit import get_multiscale_patches, compute_patch_anomaly_score
        ms = get_multiscale_patches(feats)  # [B,V,N,2D]
        patch_map = out["patch_map"]       # [B,V,N,D_model]
        for j in range(views.shape[0]):
            ps = compute_patch_anomaly_score(ms[j], feats["patch_features"][j].mean(0, True).expand(1,-1,-1).squeeze(0) if False else ms[j].reshape(-1, ms.shape[-1])[:100], k=3)
            all_knn.append(ps.max().item())

import numpy as np
zc = np.array(all_z_cls)
zv = np.array(all_z_view)

print("\n" + "=" * 50)
print("Z 塌缩诊断结果")
print("=" * 50)
print(f"样本数: {len(zc)}")
print(f"\nz_cls 范围: [{zc.min():.6f}, {zc.max():.6f}]")
print(f"z_cls 均值: {zc.mean():.6f}  标准差: {zc.std():.6f}")
print(f"z_cls 变异系数 (std/mean): {zc.std()/max(zc.mean(),1e-8):.4f}")
print(f"\nz_view 范围: [{zv.min():.6f}, {zv.max():.6f}]")
print(f"z_view 均值: {zv.mean():.6f}  标准差: {zv.std():.6f}")
print(f"z_view 变异系数: {zv.std()/max(zv.mean(),1e-8):.4f}")

print("\n" + "=" * 50)
if zc.std() < 0.01 or (zc.max() - zc.min()) < 0.01:
    print("⚠️  z_cls 已塌缩! 所有样本 z≈0, Flow 无区分能力")
elif zc.std() / max(zc.mean(), 1e-8) < 0.1:
    print("⚠️  z_cls 区分度极低! 得分几乎相同, Flow 接近塌缩")
else:
    print("✅ z_cls 有区分度, Flow 正常")

if zv.std() < 0.01 or (zv.max() - zv.min()) < 0.01:
    print("⚠️  z_view 已塌缩!")
elif zv.std() / max(zv.mean(), 1e-8) < 0.1:
    print("⚠️  z_view 区分度极低!")
else:
    print("✅ z_view 有区分度")
print("=" * 50)
