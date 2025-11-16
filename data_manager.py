# data_manager.py
import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

class DataManager:
    """统一数据管理类：负责UCI数据获取、清洗与持久化"""
    
    def __init__(self, dataset_id=144):
        """
        初始化数据管理器
        :param dataset_id: UCI数据集ID（德国信用数据集默认144）
        """
        self.dataset_id = dataset_id
        self.out_dir = os.path.join(os.getcwd(), "data_outputs")
        # 确保输出目录存在
        os.makedirs(self.out_dir, exist_ok=True)
    
    def fetch_and_save(self):
        """
        主方法：获取数据并保存为标准化DataFrame
        :return: 合并后的DataFrame
        """
        print(f"🔍 正在获取UCI数据集 (ID={self.dataset_id})...")
        ds = fetch_ucirepo(id=self.dataset_id)
        
        # 核心：从ucimlrepo对象中提取数据
        features, targets = self._extract_data(ds)
        
        # 合并为完整DataFrame
        df = pd.concat([features.reset_index(drop=True), 
                       targets.reset_index(drop=True)], axis=1)
        
        # 保存到 data_outputs/raw/ 文件夹
        raw_dir = os.path.join(self.out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        output_path = os.path.join(raw_dir, "german_credit_full.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ 数据获取成功！")
        print(f"   - 特征数: {features.shape[1]}")
        print(f"   - 样本数: {len(df)}")
        print(f"   - 已保存至: {output_path}")
        
        return df
    
    def _extract_data(self, ds):
        """
        核心私有方法：从ucimlrepo返回对象中提取features和targets
        （移植自原test.py的全部逻辑）
        """
        features = None
        targets = None
        
        # ==================== 提取策略1：尝试ds.data属性 ====================
        try:
            if hasattr(ds, "data") and ds.data is not None:
                print("   → 尝试从 ds.data 提取...")
                
                # 首选：ds.data.features 和 ds.data.targets
                if hasattr(ds.data, "features") and hasattr(ds.data, "targets"):
                    features = ds.data.features
                    targets = ds.data.targets
                    print("      ✓ 成功提取 ds.data.features 和 ds.data.targets")
                
                # 备选：ds.data.dataframe 是单个DataFrame
                elif hasattr(ds.data, "dataframe") and isinstance(ds.data.dataframe, pd.DataFrame):
                    df = ds.data.dataframe
                    features = df.iloc[:, :-1]  # 除最后一列外都是特征
                    targets = df.iloc[:, [-1]]  # 最后一列是目标
                    print("      ✓ 成功提取 ds.data.dataframe（假设最后一列是目标）")
                
                # 备选：ds.data本身是DataFrame
                elif isinstance(ds.data, pd.DataFrame):
                    df = ds.data
                    features = df.iloc[:, :-1]
                    targets = df.iloc[:, [-1]]
                    print("      ✓ 成功提取 ds.data 本身作为DataFrame")
        
        except Exception as e:
            print(f"   ⚠️ 从 ds.data 提取时出错: {e}")
        
        # ==================== 提取策略2：尝试ds.dataframe顶级属性 ====================
        if (features is None or targets is None) and hasattr(ds, "dataframe") and isinstance(ds.dataframe, pd.DataFrame):
            print("   → 尝试从 ds.dataframe 提取...")
            df = ds.dataframe
            features = df.iloc[:, :-1]
            targets = df.iloc[:, [-1]]
            print("      ✓ 成功提取 ds.dataframe（顶级属性）")
        
        # ==================== 错误处理：自动调试 ====================
        if features is None or targets is None:
            print("\n❌ 无法自动识别数据结构！")
            print("   可访问的非私有属性:")
            import pprint
            attrs = [attr for attr in dir(ds) if not attr.startswith('_')]
            pprint.pprint(attrs)
            
            # 保存调试信息到文件
            debug_file = os.path.join(self.out_dir, "dataset_debug.txt")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write("=== UCI数据集对象调试信息 ===\n")
                f.write(f"数据集ID: {self.dataset_id}\n")
                f.write(f"对象类型: {type(ds)}\n")
                f.write(f"\n可用属性列表:\n")
                for attr in attrs:
                    f.write(f"  - {attr}\n")
                f.write(f"\nds对象完整repr:\n{repr(ds)}\n")
            print(f"   调试信息已保存至: {debug_file}")
            
            raise ValueError(
                f"无法自动提取数据！请检查 {debug_file} 文件，"
                "手动修改_data_manager.py中的提取逻辑"
            )
        
        # ==================== 数据类型强制转换 ====================
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(targets, pd.DataFrame):
            targets = pd.DataFrame(targets)
        
        print(f"   → 最终数据形状 - Features: {features.shape}, Targets: {targets.shape}")
        return features, targets
    
    def load_from_local(self, file_path=None):
        """
        从本地加载已保存的数据（避免重复下载）
        :param file_path: CSV文件路径，默认为 data_outputs/raw/german_credit_full.csv
        :return: DataFrame
        """
        if file_path is None:
            file_path = os.path.join(self.out_dir, "raw", "german_credit_full.csv")
        
        if os.path.exists(file_path):
            print(f"📂 从本地加载数据: {file_path}")
            return pd.read_csv(file_path)
        else:
            print(f"⚠️ 本地文件不存在: {file_path}")
            print("   正在重新获取数据...")
            return self.fetch_and_save()


# ==================== 模块自测代码 ====================
if __name__ == '__main__':
    print("="*50)
    print("数据管理模块自测")
    print("="*50)
    
    # 创建实例
    dm = DataManager(dataset_id=144)
    
    # 测试1：首次获取数据
    print("\n【测试1】首次获取数据:")
    df = dm.fetch_and_save()
    print("\n数据预览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 测试2：从本地加载
    print("\n【测试2】从本地加载:")
    df_loaded = dm.load_from_local()
    print(f"加载成功，数据形状: {df_loaded.shape}")
    
    print("\n✅ 所有测试通过！")