# 训练
uv run python -m src.train \
  --epochs 100 \
  --batch_size 8 \
  --image_size 518 \
  --num_workers 8 \
  --dinov2_weights weights/dinov2_vitb14_pretrain.pth

# 提交
uv run python -m src.submit \
  --checkpoint checkpoints/all/best.pth \
  --test_split Test_A \
  --image_size 518 \
  --batch_size 16 \
  --num_workers 8 \
  --dinov2_weights weights/dinov2_vitb14_pretrain.pth

# 完整评估
uv run python -m src.evaluate \
    --submission_dir submission \
    --gt_dir gt \
    --per_category

# 只评估图像级 (无需 GT masks)
uv run python -m src.evaluate \
    --submission_dir submission \
    --gt_dir gt \
    --image_only

# 全量测试集，输出 Top-20 最异常样本
uv run python -m src.visualize \
  --checkpoint checkpoints/all/best.pth \
  --test_split Test_A \
  --image_size 518 \
  --dinov2_weights weights/dinov2_vitb14_pretrain.pth

# 指定类别调试
uv run python -m src.visualize \
  --checkpoint checkpoints/all/best.pth \
  --category battery \
  --top_n 10 \
  --dinov2_weights weights/dinov2_vitb14_pretrain.pth
