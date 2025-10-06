"""
Example usage of the weak learning machine.
弱学習機の使用例
"""
import numpy as np
from adaboost import AdaBoost
from weak_learner import DecisionStump


def generate_sample_data(n_samples=100):
    """
    Generate sample data for demonstration.
    デモンストレーション用のサンプルデータを生成
    """
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, 2)
    
    # Create labels based on a simple rule
    # y = 1 if x1 + x2 > 0, else -1
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] *= -1
    
    return X, y


def main():
    """
    Main function to demonstrate weak learning machine.
    弱学習機をデモンストレーションするメイン関数
    """
    print("=" * 60)
    print("Weak Learning Machine Demo")
    print("弱学習機のデモンストレーション")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    print("   サンプルデータを生成中...")
    X_train, y_train = generate_sample_data(200)
    X_test, y_test = generate_sample_data(100)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train AdaBoost with weak learners
    print("\n2. Training AdaBoost with Decision Stumps...")
    print("   決定株を用いてAdaBoostを訓練中...")
    
    clf = AdaBoost(n_estimators=50, weak_learner_class=DecisionStump)
    clf.fit(X_train, y_train)
    
    print(f"   Number of weak learners trained: {len(clf.learners)}")
    
    # Evaluate
    print("\n3. Evaluating model...")
    print("   モデルを評価中...")
    
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Show some predictions
    print("\n4. Sample predictions:")
    print("   サンプル予測:")
    for i in range(5):
        x = X_test[i]
        true_label = y_test[i]
        pred_label = clf.predict(x.reshape(1, -1))[0]
        print(f"   Sample {i+1}: Features={x}, True={true_label:+.0f}, Predicted={pred_label:+.0f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("デモが正常に完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
