# 实验结果记录

项目：基于视觉大模型的幼儿情绪检测系统  
模型：ViT-B/16  
类别：anger, contempt, disgust, fear, happy, neutral, sad, surprise

## 1. age5 基线实验

- 训练数据：age5 train
- 验证数据：age5 val
- 测试数据：age5 test
- Epochs：60
- Batch Size：32
- Learning Rate：5e-5
- Class Weights：Yes
- 日期：2026-07-05 14:54

| 指标 | 数值 |
|---|---:|
| Test Loss | 1.2881 |
| Test Acc | 53.01% |
| Test F1 | 0.5178 |

分类 F1：

| 类别 | F1 |
|---|---:|
| anger | 0.3438 |
| contempt | 0.4982 |
| disgust | 0.2883 |
| fear | 0.4375 |
| happy | 0.8365 |
| neutral | 0.7926 |
| sad | 0.4100 |
| surprise | 0.3852 |

## 2. 混合训练实验：age5 + age10 → age5

- 训练数据：age5 train + age10 train
- 验证数据：age5 val
- 测试数据：age5 test
- Epochs：60
- Batch Size：32
- Learning Rate：5e-5
- Class Weights：Yes
- 日期：2026-07-05 21:29
- 备注：使用修正后的 `train_mixed.py`，保持同一年龄组 train/val/test 固定 seed 拆分，避免数据泄漏。

| 指标 | 数值 |
|---|---:|
| Test Loss | 1.7101 |
| Test Acc | 59.56% |
| Test F1 | 0.5926 |

分类 F1：

| 类别 | F1 |
|---|---:|
| anger | 0.4578 |
| contempt | 0.5854 |
| disgust | 0.4795 |
| fear | 0.4472 |
| happy | 0.8739 |
| neutral | 0.8251 |
| sad | 0.4413 |
| surprise | 0.5089 |

## 3. 对比分析

| 实验 | 训练数据 | 测试数据 | Acc | F1 |
|---|---|---|---:|---:|
| Baseline | age5 | age5 | 53.01% | 0.5178 |
| Mixed | age5 + age10 | age5 | 59.56% | 0.5926 |
| 提升 | - | - | +6.55 pp | +0.0748 |

相较于 age5 单独训练，引入 age10 训练数据后，模型在 age5 测试集上的准确率提升 6.55 个百分点，F1 提升 0.0748。提升主要来自 disgust、surprise、anger、contempt 等原本较弱类别，说明跨年龄数据扩充可以缓解单一年龄组样本不足导致的泛化不足。

类别 F1 变化：

| 类别 | age5 baseline | age5+age10 | 变化 |
|---|---:|---:|---:|
| anger | 0.3438 | 0.4578 | +0.1140 |
| contempt | 0.4982 | 0.5854 | +0.0872 |
| disgust | 0.2883 | 0.4795 | +0.1912 |
| fear | 0.4375 | 0.4472 | +0.0097 |
| happy | 0.8365 | 0.8739 | +0.0374 |
| neutral | 0.7926 | 0.8251 | +0.0325 |
| sad | 0.4100 | 0.4413 | +0.0313 |
| surprise | 0.3852 | 0.5089 | +0.1237 |

## 4. 当前结论

1. age5 + age10 混合训练相较 age5 单独训练有明显提升，验证了跨年龄数据扩充的有效性。
2. happy 和 neutral 仍然是识别效果最稳定的类别。
3. disgust、surprise、anger 的提升最明显，说明额外年龄组数据对细粒度负面情绪有帮助。
4. fear 与 sad 仍存在较多混淆；从混淆矩阵看，fear/surprise/sad 之间仍是主要错误来源。
5. Test Loss 上升但 Acc/F1 提升，可能与类别权重和预测置信度校准有关，论文中应优先报告 Acc/F1，并对 Loss 做谨慎解释。

## 5. 后续建议

建议继续补充以下实验：

1. `age5 + age10 + age15 → age5`：验证引入更多年龄组是否继续提升。
2. `age5 + age10 → age5` 不使用 class weights：对比类别权重是否导致 loss 偏高或类别吸收。
3. 若严格强调“在原 age5 模型上继续加入 age10 数据”，可增加 `--resume-from` 参数，从 `vit_b16_age5_best.pth` 加载后用 age5+age10 混合数据低学习率微调。
