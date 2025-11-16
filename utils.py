# utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import pandas as pd
import numpy as np

# ==================== 设置中文字体 ==================== 
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_model_comparison(results, save_path):
    """
    生成模型对比图（静态版，零依赖，100%兼容）
    自动过滤 'best_model' 和 'best_auc' 等非字典数据
    """
    # 核心修复：只保留字典类型的模型结果
    models = {k: v for k, v in results.items() if isinstance(v, dict) and 'accuracy' in v}
    
    if not models:
        print("⚠️ 没有可用的模型结果")
        return None
    
    # 提取指标数据
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    data = []
    
    for model_name, metrics_dict in models.items():
        for metric in metrics:
            value = metrics_dict.get(metric, 0)
            data.append({
                'model': model_name,
                'metric': metric,
                'value': value
            })
    
    df_plot = pd.DataFrame(data)
    
    # 生成静态柱状图（避免plotly所有问题）
    static_path = save_path.replace('.html', '.png')
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_plot, x='metric', y='value', hue='model')
    plt.title('模型性能对比', fontsize=16, fontweight='bold')
    plt.ylabel('分数', fontsize=12)
    plt.xlabel('评估指标', fontsize=12)
    plt.legend(title='模型', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 模型对比图已保存: {static_path}")
    return static_path

def generate_report(results, report_dir):
    """
    生成综合报告（兼容混合类型结果）
    """
    print(f"\n📄 生成评估报告...")
    
    # 创建目录
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成模型对比图
    comparison_path = os.path.join(report_dir, "model_comparison.html")
    plot_model_comparison(results, comparison_path)
    
    # 过滤模型数据（用于表格生成）
    models = {k: v for k, v in results.items() if isinstance(v, dict) and 'accuracy' in v}
    
    # 生成Markdown报告
    readme_path = os.path.join(report_dir, "summary.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 模型训练报告\n\n")
        
        # 最佳模型信息
        if 'best_model' in results:
            f.write(f"## 🏆 最佳模型\n\n")
            f.write(f"- **模型名称**: `{results['best_model']}`\n")
            f.write(f"- **ROC AUC**: {results.get('best_auc', 'N/A'):.4f}\n\n")
        
        # 模型性能表格
        if models:
            f.write("## 📊 模型性能指标\n\n")
            f.write("| 模型 | Accuracy | Precision | Recall | F1 | ROC AUC |\n")
            f.write("|------|----------|-----------|--------|----|---------|\n")
            
            for model_name, metrics_dict in models.items():
                f.write(f"| {model_name} | ")
                f.write(f"{metrics_dict.get('accuracy', 0):.4f} | ")
                f.write(f"{metrics_dict.get('precision', 0):.4f} | ")
                f.write(f"{metrics_dict.get('recall', 0):.4f} | ")
                f.write(f"{metrics_dict.get('f1', 0):.4f} | ")
                f.write(f"{metrics_dict.get('roc_auc', 0):.4f} |\n")
            
            # 混淆矩阵
            f.write("\n## 🔢 混淆矩阵\n\n")
            for model_name, metrics_dict in models.items():
                f.write(f"### {model_name}\n")
                f.write(f"```\n{metrics_dict.get('confusion_matrix', 'N/A')}\n```\n\n")
        
        # 数据信息
        f.write("## 📁 输出文件\n\n")
        f.write("- `models/` 文件夹：训练好的模型文件（.joblib）\n")
        f.write("- `eda_plots/` 文件夹：数据可视化图表\n")
        f.write("- `model_comparison.png`：模型性能对比图\n")
        f.write("- `feature_importance.csv`：特征重要性排名\n")
    
    print(f"✅ 报告已保存: {readme_path}")

def plot_confusion_matrix(cm, model_name, save_dir):
    """绘制混淆矩阵图"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'混淆矩阵: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"cm_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵图: {save_path}")

def generate_feature_importance_plot(model_path, X_sample, save_dir):
    """生成特征重要性图"""
    os.makedirs(save_dir, exist_ok=True)
    
    fi_path = os.path.join(os.path.dirname(model_path), "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path).head(20)
        
        plt.figure(figsize=(10, 12))
        sns.barplot(data=fi_df, x='importance', y='feature')
        plt.title('前20个重要特征（随机森林）', fontsize=16, fontweight='bold')
        plt.xlabel('重要性')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 特征重要性图: {save_path}")
    else:
        print("⚠️ 特征重要性文件不存在，跳过")

# ==================== 快速测试 ====================
if __name__ == '__main__':
    # 模拟结果（包含非字典数据，测试过滤逻辑）
    mock_results = {
        'logistic_regression': {
            'accuracy': 0.72, 'precision': 0.52, 'recall': 0.73,
            'f1': 0.61, 'roc_auc': 0.78, 'confusion_matrix': [[80, 20], [10, 40]]
        },
        'random_forest': {
            'accuracy': 0.85, 'precision': 0.82, 'recall': 0.82,
            'f1': 0.82, 'roc_auc': 0.91, 'confusion_matrix': [[85, 15], [15, 35]]
        },
        'best_model': 'random_forest',  # 非字典数据
        'best_auc': 0.91                 # 非字典数据
    }
    
    print("测试utils模块（含非字典数据过滤）...")
    generate_report(mock_results, "./test_reports")
    print("✅ 测试完成！请检查 test_reports 文件夹")