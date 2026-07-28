# 线性回归与逻辑回归示例

本项目使用 scikit-learn 自带的示例数据集，在 Jupyter Notebook 中演示两类常见的监督学习模型。数据由 scikit-learn 直接加载，无需另外下载。

## 项目内容

- [线性回归：Diabetes 数据集](notebooks/linear_regression_diabetes.ipynb)
  - 使用 `sklearn.datasets.load_diabetes` 加载糖尿病回归数据集。
  - 训练线性回归模型，并通过回归指标评估预测效果。
  - 绘制真实值与预测值等可视化结果。
- [逻辑回归：Breast Cancer 数据集](notebooks/logistic_regression_breast_cancer.ipynb)
  - 使用 `sklearn.datasets.load_breast_cancer` 加载乳腺癌二分类数据集。
  - 训练逻辑回归模型，并通过分类指标评估预测效果。
  - 绘制混淆矩阵、ROC 曲线等可视化结果。

两个 Notebook 都使用 Jupyter 魔法命令：

```python
%matplotlib inline
```

该命令会让 Matplotlib 图表直接显示在 Notebook 的输出区域中。

## 环境要求

- Python 3.10 或更高版本
- JupyterLab
- NumPy
- pandas
- Matplotlib
- scikit-learn

完整依赖及最低版本见 [requirements.txt](requirements.txt)。

## 安装

在项目根目录执行以下命令。Windows PowerShell 示例：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS 或 Linux 可使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 启动与运行

在已激活虚拟环境的项目根目录启动 JupyterLab：

```bash
python -m jupyter lab
```

随后：

1. 在 JupyterLab 中打开 `notebooks/` 目录。
2. 选择 [linear_regression_diabetes.ipynb](notebooks/linear_regression_diabetes.ipynb) 或 [logistic_regression_breast_cancer.ipynb](notebooks/logistic_regression_breast_cancer.ipynb)。
3. 确认 Notebook 使用当前虚拟环境对应的 Python 内核。
4. 选择 **Kernel → Restart Kernel and Run All Cells**，从上到下运行全部单元格。

也可以在命令行中执行并保存 Notebook 输出：

```bash
python -m nbconvert --execute --to notebook --inplace notebooks/linear_regression_diabetes.ipynb
python -m nbconvert --execute --to notebook --inplace notebooks/logistic_regression_breast_cancer.ipynb
```

## 重新生成 Notebook

`scripts/` 中保留了基于 `nbformat` 的生成脚本。若需要从源代码重新构建
Notebook，可在项目根目录执行：

```bash
python scripts/generate_linear_regression_notebook.py
python scripts/generate_logistic_regression_notebook.py
```

生成脚本会创建不含缓存输出的 Notebook；随后再次使用上一节的
`python -m nbconvert --execute ...` 命令即可运行并保存所有结果与图表。
```

## 项目结构

```text
statistic_analysis/
├── notebooks/
│   ├── linear_regression_diabetes.ipynb
│   └── logistic_regression_breast_cancer.ipynb
├── scripts/
│   ├── generate_linear_regression_notebook.py
│   └── generate_logistic_regression_notebook.py
├── .gitignore
├── README.md
└── requirements.txt
```
