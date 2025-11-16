# model_trainer.py
import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)

class ModelBenchmark:
    """模型训练与评估基准测试类"""
    
    def __init__(self, random_state=42, test_size=0.15, val_size=0.17647):
        """
        初始化基准测试器
        :param random_state: 随机种子
        :param test_size: 测试集比例
        :param val_size: 验证集比例（相对于训练验证集）
        """
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.pipelines = {}
        self.results = {}
        self.output_dir = os.path.join(os.getcwd(), "data_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self, df):
        """执行完整训练流程"""
        print("\n" + "="*50)
        print("开始模型训练与评估...")
        print("="*50)
        
        # 1. 数据准备
        X, y = self._prepare_data(df)
        
        # 2. 数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
        print(f"\n✅ 数据准备完成:")
        print(f"   - 训练集: {len(X_train)} 条")
        print(f"   - 验证集: {len(X_val)} 条")
        print(f"   - 测试集: {len(X_test)} 条")
        
        # 3. 构建预处理管道
        preprocessor = self._build_preprocessor(X)
        
        # 4. 构建模型管道
        self._build_pipelines(preprocessor)
        
        # 5. 训练所有模型
        self._train_models(X_train, y_train)
        
        # 6. 评估所有模型
        self._evaluate_models(X_test, y_test)
        
        # 7. 保存模型和结果
        self._save_models()
        self._save_results()
        
        # 8. 生成报告
        self._generate_feature_importance_report(X_train)
        
        print("\n✅ 模型训练完成！")
        
        return self.results
    
    def _prepare_data(self, df):
        """准备特征和目标变量"""
        # 假设最后一列是目标
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 标签映射：确保是二分类 0/1
        unique_vals = sorted(y.unique())
        if len(unique_vals) == 2:
            if set(unique_vals) == {1, 2}:
                mapping = {1: 0, 2: 1}
            else:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            y = y.map(mapping)
            print(f"\n🎯 标签映射: {mapping}")
        else:
            print(f"⚠️ 警告: 目标变量不是二分类！唯一值: {unique_vals}")
        
        return X, y
    
    def _split_data(self, X, y):
        """分层分割数据（训练/验证/测试）"""
        # 第一次分割：分出测试集
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        
        # 第二次分割：从训练验证集中分出验证集
        # val_size是相对于trainval的比例，最终比例为: train(70%) / val(15%) / test(15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=self.val_size, 
            stratify=y_trainval, random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _build_preprocessor(self, X):
        """构建预处理管道"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        
        print(f"\n🔄 构建预处理管道:")
        print(f"   - 数值型特征: {len(numeric_cols)}个")
        print(f"   - 类别型特征: {len(categorical_cols)}个")
        
        # 数值型管道
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # 类别型管道（兼容不同sklearn版本）
        try:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        except TypeError:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        return preprocessor
    
    def _build_pipelines(self, preprocessor):
        """构建所有模型管道"""
        print("\n🤖 构建模型管道:")
        
        # 逻辑回归
        self.pipelines['logistic_regression'] = Pipeline(steps=[
            ('pre', preprocessor),
            ('clf', LogisticRegression(
                max_iter=2000, class_weight='balanced', random_state=self.random_state
            ))
        ])
        print("   - ✅ 逻辑回归")
        
        # 随机森林
        self.pipelines['random_forest'] = Pipeline(steps=[
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, 
                class_weight='balanced', n_jobs=-1
            ))
        ])
        print("   - ✅ 随机森林")
        
        # XGBoost（可选）
        try:
            from xgboost import XGBClassifier
            self.pipelines['xgboost'] = Pipeline(steps=[
                ('pre', preprocessor),
                ('clf', XGBClassifier(
                    use_label_encoder=False, eval_metric='logloss',
                    random_state=self.random_state, n_jobs=-1
                ))
            ])
            print("   - ✅ XGBoost")
        except ImportError:
            print("   - ⚠️ XGBoost不可用，已跳过")
        
        # TabNet（可选，用于提升算法复杂度）
        try:
            from pytorch_tabnet import TabNetClassifier
            self.pipelines['tabnet'] = Pipeline(steps=[
                ('pre', preprocessor),
                ('clf', TabNetClassifier(
                    seed=self.random_state, verbose=0
                ))
            ])
            print("   - ✅ TabNet")
        except ImportError:
            print("   - ℹ️ TabNet未安装，不影响核心功能")
    
    def _train_models(self, X_train, y_train):
        """训练所有模型"""
        print("\n🎓 开始训练模型:")
        
        for name, pipeline in self.pipelines.items():
            print(f"\n   → 训练 {name}...")
            pipeline.fit(X_train, y_train)
            print(f"      ✅ 训练完成")
    
    def _evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        print("\n📊 模型评估结果:")
        
        for name, pipeline in self.pipelines.items():
            print(f"\n   → 评估 {name}:")
            
            # 预测
            y_pred = pipeline.predict(X_test)
            y_proba = None
            
            # 预测概率或决策函数
            if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline.named_steps['clf'], 'decision_function'):
                y_proba = pipeline.decision_function(X_test)
            
            # 计算指标
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # 计算AUC（如果有概率）
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
                except Exception as e:
                    metrics['roc_auc'] = None
                    print(f"      ⚠️ AUC计算失败: {e}")
            else:
                metrics['roc_auc'] = None
            
            self.results[name] = metrics
            
            # 打印结果
            print(f"      - Accuracy: {metrics['accuracy']:.4f}")
            print(f"      - Precision: {metrics['precision']:.4f}")
            print(f"      - Recall: {metrics['recall']:.4f}")
            print(f"      - F1: {metrics['f1']:.4f}")
            print(f"      - ROC AUC: {metrics['roc_auc']:.4f}")
    
    def _save_models(self):
        """保存所有训练好的模型"""
        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        print(f"\n💾 保存模型至: {models_dir}")
        
        for name, pipeline in self.pipelines.items():
            model_path = os.path.join(models_dir, f"model_{name}.joblib")
            joblib.dump(pipeline, model_path)
            print(f"   - ✅ {name} 已保存")
    
    def _save_results(self):
        """保存评估结果到JSON"""
        results_file = os.path.join(self.output_dir, "models", "metrics.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # 添加最佳模型信息
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1].get('roc_auc', 0))
            self.results['best_model'] = best_model[0]
            self.results['best_auc'] = best_model[1].get('roc_auc')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 评估结果已保存: {results_file}")
    
    def _generate_feature_importance_report(self, X_train):
        """生成特征重要性报告（仅随机森林）"""
        if 'random_forest' not in self.pipelines:
            return
        
        print("\n📈 生成特征重要性报告:")
        
        try:
            # 获取预处理后特征名
            pre = self.pipelines['random_forest'].named_steps['pre']
            
            # 数值型特征名
            num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            # 类别型特征名（独热编码后）
            cat_features = []
            if 'cat' in pre.named_transformers_:
                ohe = pre.named_transformers_['cat'].named_steps['onehot']
                cat_cols = [c for c in X_train.columns if c not in num_features]
                try:
                    cat_features = list(ohe.get_feature_names_out(cat_cols))
                except:
                    cat_features = []
            
            all_features = num_features + cat_features
            
            # 获取重要性
            importances = self.pipelines['random_forest'].named_steps['clf'].feature_importances_
            
            if len(importances) == len(all_features):
                # 保存到CSV
                fi_df = pd.DataFrame({
                    'feature': all_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                fi_path = os.path.join(self.output_dir, "models", "feature_importance.csv")
                fi_df.to_csv(fi_path, index=False)
                print(f"   - ✅ 特征重要性已保存: {fi_path}")
            else:
                print(f"   - ⚠️ 特征数量不匹配，跳过")
        
        except Exception as e:
            print(f"   - ⚠️ 生成失败: {e}")


# ==================== 模块自测代码 ====================
if __name__ == '__main__':
    print("="*60)
    print("模型训练模块自测")
    print("="*60)
    
    # 创建模拟数据（实际使用时请替换为真实数据）
    from data_manager import DataManager
    
    # 获取数据
    dm = DataManager()
    df = dm.load_from_local() or dm.fetch_and_save()
    
    # 执行训练
    benchmark = ModelBenchmark()
    results = benchmark.run(df)
    
    print("\n✅ 自测完成！最佳模型:", results.get('best_model'))