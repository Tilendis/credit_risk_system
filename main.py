# 控制台菜单，一键执行全流程
import os
from data_manager import DataManager
from eda import EDAAnalyzer
from model_trainer import ModelBenchmark
from utils import generate_report

def main():
    print("=== 金融风险评估系统 ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 数据获取
    print("\n[1/4] 正在获取数据...")
    dm = DataManager(dataset_id=144)
    df = dm.fetch_and_save()
    
    # 2. EDA分析
    print("\n[2/4] 正在执行探索性分析...")
    eda = EDAAnalyzer(df, os.path.join(base_dir, "data_outputs"))
    eda.run_full_analysis()
    
    # 3. 模型训练
    print("\n[3/4] 正在训练模型...")
    benchmark = ModelBenchmark()
    results = benchmark.run(df)
    
    # 4. 生成报告
    print("\n[4/4] 正在生成报告...")
    generate_report(results, os.path.join(base_dir, "data_outputs", "reports"))
    
    print("\n✅ 系统运行完成！请查看 data_outputs 文件夹")
    print("\n可选操作：")
    print(" - 运行 'python app.py' 启动Web界面")
    print(" - 查看 reports/ 中的自动分析报告")

if __name__ == '__main__':
    main()