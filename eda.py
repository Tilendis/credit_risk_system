# eda.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    """探索性数据分析类：自动完成数据质量检测与可视化"""
    
    def __init__(self, df, output_dir):
        """
        初始化分析器
        :param df: 待分析的DataFrame（包含特征和目标）
        :param output_dir: 输出目录路径
        """
        self.df = df
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "eda_plots")
        # 自动创建图表输出目录
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 自动识别目标列（假设最后一列是目标）
        self.target_col = df.columns[-1]
        print(f"📊 目标列自动识别为: '{self.target_col}'")
    
    def run_full_analysis(self):
        """执行完整EDA流程并保存所有结果"""
        print("\n" + "="*50)
        print("开始执行探索性数据分析(EDA)...")
        print("="*50)
        
        self._basic_info()
        self._target_analysis()
        self._feature_analysis()
        self._generate_plots()
        
        print("\n✅ EDA分析完成！")
        print(f"   - 数据摘要: {os.path.join(self.output_dir, 'eda_summary.csv')}")
        print(f"   - 图表文件: {self.plots_dir}")
    
    def _basic_info(self):
        """打印基础数据信息"""
        print("\n【1】基础信息")
        print(f"   - 数据形状: {self.df.shape[0]}行 × {self.df.shape[1]}列")
        print(f"   - 内存占用: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 数据类型统计
        dtype_counts = self.df.dtypes.value_counts()
        print("   - 数据类型统计:")
        for dtype, count in dtype_counts.items():
            print(f"      * {dtype}: {count}列")
        
        # 缺失值统计
        missings = self.df.isnull().sum()
        if missings.sum() > 0:
            print("   - 缺失值警告:")
            print(missings[missings > 0])
        else:
            print("   - 缺失值: 无 (0)")
    
    def _target_analysis(self):
        """分析目标变量分布"""
        print("\n【2】目标变量分析")
        target_series = self.df[self.target_col]
        
        # 统计分布
        counts = target_series.value_counts(dropna=False)
        percents = target_series.value_counts(normalize=True, dropna=False) * 100
        
        print(f"   - 类别分布:")
        for val in counts.index:
            print(f"      * 类别 {val}: {counts[val]} 条 ({percents[val]:.1f}%)")
        
        # 检查是否为二分类
        n_unique = target_series.nunique()
        print(f"   - 唯一值数量: {n_unique}")
        if n_unique == 2:
            print("   - 数据类型: 二分类问题")
        else:
            print(f"   - 警告: 发现{n_unique}个类别，可能不是标准二分类")
    
    def _feature_analysis(self):
        """分析特征变量"""
        print("\n【3】特征变量分析")
        
        # 分割数值型和类别型特征
        self.numeric_cols = self.df.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in self.df.columns if c not in self.numeric_cols and c != self.target_col]
        
        print(f"   - 数值型特征: {len(self.numeric_cols)}个")
        if len(self.numeric_cols) > 0:
            print(f"      前5个: {self.numeric_cols[:5]}...")
        
        print(f"   - 类别型特征: {len(self.categorical_cols)}个")
        if len(self.categorical_cols) > 0:
            print(f"      前5个: {self.categorical_cols[:5]}...")
        
        # 数值型特征统计
        if len(self.numeric_cols) > 0:
            print("\n   - 数值型特征统计摘要:")
            print(self.df[self.numeric_cols].describe().T[['min', 'mean', 'max']].head())
        
        # 类别型特征统计
        if len(self.categorical_cols) > 0:
            print("\n   - 类别型特征唯一值数量:")
            for col in self.categorical_cols[:5]:  # 只显示前5个
                n_unique = self.df[col].nunique()
                print(f"      * {col}: {n_unique}个类别")
        
        # 保存完整统计摘要到CSV
        self._save_summary_csv()
    
    def _save_summary_csv(self):
        """生成并保存数据摘要CSV"""
        summary = []
        for col in self.df.columns:
            summary.append({
                'column': col,
                'dtype': str(self.df[col].dtype),
                'n_unique': int(self.df[col].nunique(dropna=False)),
                'n_missing': int(self.df[col].isnull().sum()),
                'missing_rate': round(self.df[col].isnull().mean() * 100, 2)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(self.output_dir, 'eda_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n   - 数据摘要已保存: {summary_path}")
    
    def _generate_plots(self):
        """生成所有可视化图表"""
        print("\n【4】生成可视化图表")
        
        # 1. 目标分布柱状图
        self._plot_target_distribution()
        
        # 2. 数值型特征直方图
        if len(self.numeric_cols) > 0:
            self._plot_histograms()
        
        # 3. 相关性热力图
        if len(self.numeric_cols) >= 2:
            self._plot_correlation_heatmap()
        
        # 4. 类别型特征分布图
        if len(self.categorical_cols) > 0:
            self._plot_categorical_counts()
    
    def _plot_target_distribution(self):
        """目标变量分布图"""
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=self.df, x=self.target_col)
        
        # 添加百分比标注
        total = len(self.df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 5,
                    f'{height/total*100:.1f}%', ha="center")
        
        plt.title(f'Target Distribution: {self.target_col}', fontsize=14)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'target_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   - 目标分布图: {save_path}")
    
    def _plot_histograms(self):
        """数值型特征直方图（最多8个）"""
        n_plots = min(8, len(self.numeric_cols))
        print(f"   - 生成{n_plots}个直方图...")
        
        for i, col in enumerate(self.numeric_cols[:n_plots]):
            plt.figure(figsize=(10, 4))
            
            # 绘制直方图
            sns.histplot(self.df[col].dropna(), kde=True, bins=30)
            plt.title(f'Histogram: {col} (skew={self.df[col].skew():.2f})')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            
            # 保存
            save_path = os.path.join(self.plots_dir, f'hist_{col}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
        
        print(f"      保存至: {self.plots_dir}")
    
    def _plot_correlation_heatmap(self):
        """数值型特征相关性热力图"""
        plt.figure(figsize=(12, 10))
        
        # 计算相关性矩阵
        corr = self.df[self.numeric_cols].corr()
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr, dtype=bool))  # 只显示下三角
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        plt.title('Numeric Features Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   - 相关性热力图: {save_path}")
    
    def _plot_categorical_counts(self):
        """类别型特征分布图（前3个）"""
        print(f"   - 生成类别型特征分布图（前3个）...")
        
        for col in self.categorical_cols[:3]:
            plt.figure(figsize=(10, 6))
            
            # 绘制条形图（显示前10个最常见类别）
            data = self.df[col].value_counts().head(10)
            sns.barplot(x=data.values, y=data.index)
            
            plt.title(f'Top 10 Categories: {col}')
            plt.xlabel('Count')
            plt.tight_layout()
            
            # 保存
            save_path = os.path.join(self.plots_dir, f'cat_{col}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
        
        print(f"      保存至: {self.plots_dir}")


# ==================== 模块自测代码 ====================
if __name__ == '__main__':
    print("="*60)
    print("EDA分析模块自测")
    print("="*60)
    
    # 模拟数据
    import numpy as np
    
    # 创建示例DataFrame
    df_test = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randint(0, 5, 100),
        'target': np.random.randint(1, 3, 100)
    })
    
    # 执行分析
    analyzer = EDAAnalyzer(df_test, output_dir="./test_outputs")
    analyzer.run_full_analysis()
    python
    print("\n✅ 自测完成！请检查 test_outputs 文件夹")