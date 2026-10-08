# 卡车-多无人机协同配送 IALNS

本仓库提供论文最终完整 IALNS 算法的复现材料，包括代码、实际使用的改造
Solomon 算例、生成程序、固定随机种子、完整参数，以及 IALNS 实验结果。

模型考虑异质需求重量、仅卡车客户、多客户出航、时间窗、暴露敏感包裹和碳排放成本。
最终版允许卡车在等待客户就绪期间发射无人机，暴露时间按无人机客户开始服务时刻计算。

## 运行

```bash
python -m pip install -r requirements.txt
python scripts/run_ialns.py --instance instances/base/c101改10.xlsx --seed 104729
python -m unittest discover -s tests -v
```

默认运行 4500 代，启用完整算法，坐标在内存中统一乘以 0.6。
Excel 保存的是原始坐标，不能预先缩放后再次运行默认缩放。

```bash
python scripts/run_experiments.py --suite main --workers 8
python scripts/run_experiments.py --suite small --workers 8
python scripts/run_experiments.py --suite sensitivity --workers 8
```

批处理会保存各任务的结果，并在恢复时核对已有记录的代码、输入、配置和种子。
`--cpu-ids 0,2,4,6,8,10,12,14` 可指定逻辑处理器，但应先核对自己的 CPU 拓扑，
不能假定其他电脑上的这些编号必然对应八个物理核心。

## 材料范围

- `src/ialns.py`：未改动的论文冻结版，完整算法包含 21 个复合算子。
- `instances/`：30 个基准 Excel、18 个暴露比例 Excel，以及生成用原始坐标。
- `settings/`：参数、十个固定求解种子、生成元数据与 SHA-256 校验值。
- `scripts/`：可移植运行入口、批处理、算例生成和结果验证程序。
- `results/`：240 条主实验、60 条 5/15 客户补充记录及 810 条敏感性记录，
  另含对应路线、时序和收敛历史。敏感性中的复用基准记录有独立标识。
- `tests/`：输入与生成一致性检查、费用/可行性验证和短算例复现测试。

历史 101 系列的属性生成种子未归档，因此其精确复现依据为提供的固定 Excel。
102 系列采用已归档的 Mulberry32 与 FNV-1a 种子规则，可重建全部发布算例属性。
暴露比例算例保留嵌套客户顺序与原始随机分位数，以保证不同档位只有暴露参数变化。
生成程序输出可直接求解的 CSV；发布的 Excel 为实际实验原件，均保留校验值。

求解种子均为：104729、130363、155921、181081、206369、231779、257053、
282427、307831、333287。求解种子不能代替客户属性生成种子。

详细内容见 [英文介绍](README.md)、[实验口径](docs/experiments.md)
和[算例说明](docs/instances.md)。本仓库不宣称完整实验独立于预实验调参，
也不把启发式最好解称为已经证明的全局最优解。
