# weak-learning-machine

弱学習機による学習 (Learning with Weak Learners)

A Python implementation of weak learning machines using AdaBoost algorithm with decision stumps.

## Overview / 概要

This project implements a weak learning machine (弱学習機) using the AdaBoost algorithm. Weak learners are simple models that perform slightly better than random guessing. When combined using ensemble methods like AdaBoost, they can create a strong learner with high accuracy.

このプロジェクトは、AdaBoostアルゴリズムを使用した弱学習機（弱学習器）を実装しています。弱学習器は、ランダムな推測よりわずかに優れたパフォーマンスを発揮するシンプルなモデルです。AdaBoostのようなアンサンブル手法を使用して組み合わせると、高精度の強力な学習器を作成できます。

## Features / 特徴

- **Weak Learner Base Class**: Abstract base class for implementing weak learners
- **Decision Stump**: A simple weak learner that makes decisions based on a single feature
- **AdaBoost**: Ensemble method that combines multiple weak learners
- **Example Usage**: Demonstration script showing how to use the library

## Installation / インストール

```bash
# Clone the repository
git clone https://github.com/ShinMiz/weak-learning-machine.git
cd weak-learning-machine

# Install dependencies
pip install -r requirements.txt
```

## Usage / 使用方法

### Basic Example / 基本的な例

```python
import numpy as np
from adaboost import AdaBoost
from weak_learner import DecisionStump

# Generate sample data
X_train = np.random.randn(100, 2)
y_train = np.where(X_train[:, 0] + X_train[:, 1] > 0, 1, -1)

# Train AdaBoost with weak learners
clf = AdaBoost(n_estimators=50, weak_learner_class=DecisionStump)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_train)

# Calculate accuracy
accuracy = clf.score(X_train, y_train)
print(f"Accuracy: {accuracy:.4f}")
```

### Running the Demo / デモの実行

```bash
python example.py
```

This will:
1. Generate sample training and test data
2. Train an AdaBoost classifier with decision stumps
3. Evaluate the model on both training and test sets
4. Show sample predictions

## Project Structure / プロジェクト構造

```
weak-learning-machine/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── weak_learner.py          # Weak learner implementations
├── adaboost.py              # AdaBoost algorithm
├── example.py               # Usage example
└── test_weak_learner.py     # Unit tests
```

## Testing / テスト

Run the tests using:

```bash
python -m unittest test_weak_learner.py
```

Or run with verbose output:

```bash
python -m unittest test_weak_learner.py -v
```

## Algorithm Details / アルゴリズムの詳細

### Decision Stump / 決定株

A decision stump is the simplest form of decision tree with only one split. It chooses:
- A single feature to split on
- A threshold value
- A polarity (direction of the inequality)

決定株は、1つの分割のみを持つ決定木の最も単純な形式です。以下を選択します：
- 分割する単一の特徴
- しきい値
- 極性（不等式の方向）

### AdaBoost

AdaBoost (Adaptive Boosting) works by:
1. Training weak learners sequentially
2. Adjusting sample weights to focus on misclassified examples
3. Combining predictions with weighted voting

AdaBoost（適応的ブースティング）は次のように機能します：
1. 弱学習器を順次訓練
2. 誤分類された例に焦点を当てるためにサンプルの重みを調整
3. 重み付き投票で予測を組み合わせ

## Requirements / 必要条件

- Python 3.6+
- NumPy >= 1.20.0

## License / ライセンス

This project is open source and available for educational purposes.