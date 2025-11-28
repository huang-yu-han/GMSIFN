# GMSIFN 测试指南

## 快速测试

修复完成后，可以通过以下步骤验证模型是否正常工作：

### 方法1：运行测试脚本

```bash
cd GMSIFN
python test_model.py
```

**预期输出**：
```
============================================================
Testing GMSIFN Model
============================================================
Device: cuda:0
Input shape: torch.Size([2, 9, 5120])
Number of parameters: 43

Parameter shapes (first 5):
  params[0]: torch.Size([240, 240])
  params[1]: torch.Size([240])
  params[2]: torch.Size([240, 241])
  params[3]: torch.Size([240])
  params[4]: torch.Size([720, 240])

[Test 1] Forward pass...
✓ Forward pass successful!
  Output shape: torch.Size([2, 16]) (expected: [2, 16])
  ✓ Output shape is correct!

[Test 2] Feature extraction...
✓ Feature extraction successful!
  Feature shape: torch.Size([2, 240]) (expected: [2, 240])
  ✓ Feature shape is correct!

[Test 3] Backward pass (gradient computation)...
✓ Backward pass successful!
  Loss value: X.XXXX

============================================================
✓ All tests passed! Model is working correctly.
============================================================
```

### 方法2：检查参数顺序

```bash
python debug_params.py
```

这将显示所有43个参数及其形状，用于验证参数顺序是否正确。

---

## 完整训练测试

### 1. 准备数据

确保你的数据文件在正确位置：
```
RPM/
├── data.npy    # [num_samples, 5120, 9]
└── labels.npy  # [num_samples]
```

### 2. 配置参数

编辑 `config.py`：
```python
DATA_CONFIG = {
    'data_path': r'RPM/data.npy',      # ← 更新路径
    'label_path': r'RPM/labels.npy',   # ← 更新路径
    'test_classes': [0, 1, 3, 5, 13],
    'seed': 99010
}
```

### 3. 运行训练

```bash
python train.py
```

**预期输出结构**：
```
=============================================================
GMSIFN: Cross-Category Meta-Learning
=============================================================

Device: cuda:0
Random seed: 99010

-------------------------------------------------------------
Loading Data...
-------------------------------------------------------------
Base training classes (11 classes): [2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]
Novel test classes (5 classes): [0, 1, 3, 5, 13]

[Stage 1/3] Meta-Training on Base Classes...
Meta-Train Epoch 1/1 | Acc: X.XXXX

[Stage 2/3] Fine-Tuning on Novel Classes...
Finetune Epoch 1/10 | Loss: X.XXXX | Acc: X.XXXX
...

[Stage 3/3] Evaluating on Novel Classes...

Test Accuracy: X.XXXX
Test F1-score: X.XXXX
Test AUC: X.XXXX
```

---

## 常见问题排查

### Q1: 报错 "RuntimeError: mat2 must be a matrix"
**解决**：这个问题已通过参数索引修复解决。如果仍然出现，请确保使用的是最新版本的代码。

### Q2: 参数数量不是43个
**检查**：
```python
import torch
from models import GMSIFN

model = GMSIFN(radius=2, T=2, input_feature_dim=240,
               input_bond_dim=1, fingerprint_dim=240,
               output_units_num=16, p_dropout=0.2, top_k=5)
params = list(model.parameters())
print(f"参数数量: {len(params)}")  # 应该输出: 43
```

### Q3: 维度不匹配错误
**验证维度**：
- 输入: [batch_size, 9, 5120]
- 经过投影: [batch_size, 9, 240]
- 图级特征: [batch_size, 240]
- 输出: [batch_size, 16]

### Q4: CUDA out of memory
**解决方案**：
1. 减小 batch size (通过减少 episode 数量)
2. 减小 fingerprint_dim (从240改为120)
3. 使用CPU：在 config.py 中设置 `device: 'cpu'`

---

## 验证清单

在提交最终结果前，请确认：

- [ ] test_model.py 运行成功，所有测试通过
- [ ] 无 "Multi-hop" 或 "Bidirectional" 字眼
- [ ] 参数总数为43个
- [ ] 输出层维度是 `[fingerprint_dim, output_units_num]`
- [ ] 可以正常训练和评估
- [ ] 结果与预期性能相符

---

## 性能基准

修复后的模型应该达到以下性能范围（仅供参考）：

- **Accuracy**: 0.75 - 0.90
- **F1-score**: 0.70 - 0.88
- **AUC**: 0.85 - 0.95

实际性能取决于：
- 数据质量
- 超参数设置
- 随机种子
- 训练epoch数

---

## 获取帮助

如果遇到问题：

1. 检查 BUG_FIXES.md 了解已修复的问题
2. 查看 ARCHITECTURE_UPDATES.md 了解架构变更
3. 运行 debug_params.py 检查参数
4. 检查 Python 和 PyTorch 版本兼容性
