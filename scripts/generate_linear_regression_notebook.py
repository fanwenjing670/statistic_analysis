"""Generate the linear-regression tutorial notebook with nbformat."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "notebooks" / "linear_regression_diabetes.ipynb"


def markdown_cell(source: str):
    """Create a clean Markdown cell from an indented multiline string."""
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    """Create a clean, unexecuted code cell."""
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    """Build a deterministic Chinese tutorial for univariate linear regression."""
    cells = [
        markdown_cell(
            """
            # 线性回归教程：用 BMI 预测糖尿病疾病进展

            本教程使用 scikit-learn 内置的
            [`load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
            数据集，训练一个只使用 **BMI** 特征的线性回归模型。单特征设计便于把观测值和回归线画在同一张二维图中。

            > 说明：这个示例用于学习回归建模流程，不应被用于医学诊断。
            """
        ),
        markdown_cell(
            """
            ## Goal

            完成本教程后，你将能够：

            - 从 scikit-learn 加载内置示例数据集；
            - 选择单个 BMI 特征并划分训练集、测试集；
            - 使用 `LinearRegression` 拟合模型；
            - 计算 MSE、RMSE 和 R²；
            - 在 Jupyter 中用散点图和回归线检查模型结果。
            """
        ),
        markdown_cell(
            """
            ## Setup

            运行下面的单元格导入依赖并固定参数。`%matplotlib inline` 是 Jupyter 魔法命令，
            用于让 Matplotlib 图表直接显示在 Notebook 中。
            """
        ),
        code_cell(
            """
            %matplotlib inline

            import matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.datasets import load_diabetes
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split

            TEST_SIZE = 0.20

            print(f"scikit-learn: {sklearn.__version__}")
            print(f"matplotlib: {matplotlib.__version__}")
            """
        ),
        markdown_cell(
            """
            ## Steps

            ### 1. 加载内置 Diabetes 数据集

            `as_frame=True` 让特征和目标以 pandas 对象返回，便于按列名选择特征。
            数据直接来自 scikit-learn，不需要下载外部文件。
            """
        ),
        code_cell(
            """
            diabetes = load_diabetes(as_frame=True)
            features = diabetes.data.copy()
            target = diabetes.target.rename("disease_progression")

            print(f"Samples: {features.shape[0]}")
            print(f"Features: {features.shape[1]}")
            print(f"Feature names: {features.columns.tolist()}")
            features.head()
            """
        ),
        markdown_cell(
            """
            ### 2. 选择 BMI 单特征

            Diabetes 数据集中的特征已经过标准化。这里保留二维表结构 `[['bmi']]`，
            以符合 scikit-learn 对输入特征矩阵的要求。
            """
        ),
        code_cell(
            """
            X = features.loc[:, ["bmi"]].copy()
            y = target.copy()

            selected_data = pd.concat([X, y], axis=1)
            selected_data.describe().round(3)
            """
        ),
        markdown_cell(
            """
            ### 3. 划分训练集和测试集

            使用 80% 数据训练、20% 数据测试，并设置 `random_state=42`。
            固定随机种子可以让每次运行得到完全相同的数据划分。
            """
        ),
        code_cell(
            """
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=42,
                shuffle=True,
            )

            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            """
        ),
        markdown_cell(
            """
            ### 4. 训练线性回归模型

            模型学习一条直线：

            \\[
            \\hat{y} = \\beta_0 + \\beta_1 \\times \\mathrm{BMI}
            \\]

            其中，`intercept_` 是截距 \\(\\beta_0\\)，`coef_[0]` 是 BMI 的系数 \\(\\beta_1\\)。
            """
        ),
        code_cell(
            """
            model = LinearRegression()
            model.fit(X_train, y_train)

            print(f"Intercept: {model.intercept_:.3f}")
            print(f"BMI coefficient: {model.coef_[0]:.3f}")
            """
        ),
        markdown_cell(
            """
            ### 5. 预测并评估模型

            - **MSE（均方误差）**：预测误差平方的平均值，越小越好；
            - **RMSE（均方根误差）**：与目标变量单位一致，越小越好；
            - **R²（决定系数）**：衡量模型解释测试集变异的程度，最高为 1，也可能为负数。
            """
        ),
        code_cell(
            """
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics = pd.DataFrame(
                {
                    "Metric": ["MSE", "RMSE", "R²"],
                    "Value": [mse, rmse, r2],
                }
            )
            metrics.round(3)
            """
        ),
        markdown_cell(
            """
            ### 6. 绘制测试集散点图和回归线

            为了让回归线连续显示，先按 BMI 对测试样本排序。图表标题和坐标轴使用英文，
            避免本地环境缺少中文字体时出现乱码。
            """
        ),
        code_cell(
            """
            bmi_test = X_test["bmi"].to_numpy()
            sort_order = np.argsort(bmi_test)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(
                bmi_test,
                y_test.to_numpy(),
                color="#2563EB",
                alpha=0.75,
                edgecolors="white",
                linewidth=0.5,
                label="Test observations",
            )
            ax.plot(
                bmi_test[sort_order],
                y_pred[sort_order],
                color="#DC2626",
                linewidth=2.5,
                label="Linear regression",
            )
            ax.set_title("Diabetes Progression vs. BMI (Test Set)")
            ax.set_xlabel("Standardized BMI")
            ax.set_ylabel("Disease Progression")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Checks

            下面的断言检查输入、输出和指标是否符合预期。如果任一条件不满足，
            Notebook 会立即报错，帮助定位问题。
            """
        ),
        code_cell(
            """
            assert X.columns.tolist() == ["bmi"], "Model input must contain only BMI."
            assert X.shape[1] == 1, "Expected exactly one input feature."
            assert not X.isna().any().any(), "BMI contains missing values."
            assert not y.isna().any(), "Target contains missing values."
            assert len(y_pred) == len(y_test), "Prediction count does not match test data."
            assert mse >= 0 and rmse >= 0, "Error metrics must be non-negative."
            assert np.isfinite([mse, rmse, r2]).all(), "Metrics must be finite."
            assert np.isclose(rmse**2, mse), "RMSE squared should equal MSE."

            print("All checks passed.")
            """
        ),
        markdown_cell(
            """
            ### 查看部分预测结果

            残差定义为“实际值减去预测值”。这里仅显示前 10 条记录，避免产生过大的输出。
            """
        ),
        code_cell(
            """
            prediction_comparison = pd.DataFrame(
                {
                    "BMI": X_test["bmi"].to_numpy(),
                    "Actual": y_test.to_numpy(),
                    "Predicted": y_pred,
                    "Residual": y_test.to_numpy() - y_pred,
                }
            )
            prediction_comparison.head(10).round(3)
            """
        ),
        markdown_cell(
            """
            ## Next Steps

            可以在这个基础上继续尝试：

            1. 同时使用全部 10 个特征，比较多元线性回归与 BMI 单特征模型的 R²；
            2. 绘制残差图，检查误差是否存在明显模式；
            3. 使用交叉验证代替单次训练/测试划分；
            4. 比较 Ridge、Lasso 等带正则化的线性模型；
            5. 改变 `TEST_SIZE`，观察评估指标的稳定性。
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
    )
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    return notebook


def main():
    """Write the notebook to the repository's notebooks directory."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT_PATH)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
