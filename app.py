# app.py（完全人类可读版 - 面向一线操作人员）
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# 加载模型
MODEL_PATH = "./data_outputs/models/model_random_forest.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}\n请先运行 python main.py 训练模型")

model = joblib.load(MODEL_PATH)

# 完全人类可读的特征配置（选项直接显示业务含义，隐藏编码）
FEATURE_CONFIG = {
    "Attribute1": {
        "label": "支票账户状态",
        "type": "dropdown",
        "choices": [
            ("< 0 DM（透支）", "A11"),
            ("0-200 DM", "A12"),
            (">= 200 DM", "A13"),
            ("无支票账户", "A14")
        ],
        "default": "A11"
    },
    "Attribute2": {
        "label": "信用期限（月）",
        "type": "number",
        "default": 24,
        "min": 1,
        "max": 72
    },
    "Attribute3": {
        "label": "信用历史",
        "type": "dropdown",
        "choices": [
            ("无贷款历史/从未逾期", "A30"),
            ("所有贷款已还清", "A31"),
            ("当前贷款正常还款", "A32"),
            ("过去曾有逾期", "A33"),
            ("有严重逾期记录", "A34")
        ],
        "default": "A32"
    },
    "Attribute4": {
        "label": "贷款用途",
        "type": "dropdown",
        "choices": [
            ("新车", "A40"),
            ("二手车", "A41"),
            ("家具/设备", "A42"),
            ("收音机/电视机", "A43"),
            ("家用电器", "A44"),
            ("房屋维修", "A45"),
            ("教育培训", "A46"),
            ("度假", "A47"),
            ("职业培训", "A48"),
            ("商业投资", "A49")
        ],
        "default": "A43"
    },
    "Attribute5": {
        "label": "信用额度（欧元）",
        "type": "number",
        "default": 5000,
        "min": 100,
        "max": 20000
    },
    "Attribute6": {
        "label": "储蓄账户余额",
        "type": "dropdown",
        "choices": [
            ("< 100 DM", "A61"),
            ("100-500 DM", "A62"),
            ("500-1000 DM", "A63"),
            (">= 1000 DM", "A64"),
            ("未知/无储蓄", "A65")
        ],
        "default": "A65"
    },
    "Attribute7": {
        "label": "当前就业时长",
        "type": "dropdown",
        "choices": [
            ("失业", "A71"),
            ("< 1年", "A72"),
            ("1-4年", "A73"),
            ("4-7年", "A74"),
            (">= 7年", "A75")
        ],
        "default": "A73"
    },
    "Attribute8": {
        "label": "分期付款率（占收入%）",
        "type": "number",
        "default": 4,
        "min": 1,
        "max": 10
    },
    "Attribute9": {
        "label": "个人状况与性别",
        "type": "dropdown",
        "choices": [
            ("男性：离婚/分居", "A91"),
            ("女性：离婚/分居/已婚", "A92"),
            ("男性：单身", "A93"),
            ("男性：已婚/丧偶", "A94")
        ],
        "default": "A93"
    },
    "Attribute10": {
        "label": "其他债务人/担保人",
        "type": "dropdown",
        "choices": [
            ("无", "A101"),
            ("共同申请人", "A102"),
            ("担保人", "A103")
        ],
        "default": "A101"
    },
    "Attribute11": {
        "label": "现居住地时长（年）",
        "type": "number",
        "default": 4,
        "min": 1,
        "max": 10
    },
    "Attribute12": {
        "label": "财产状况",
        "type": "dropdown",
        "choices": [
            ("不动产", "A121"),
            ("建筑储蓄/人寿保险", "A122"),
            ("汽车或其他财产", "A123"),
            ("未知/无财产", "A124")
        ],
        "default": "A124"
    },
    "Attribute13": {
        "label": "年龄（岁）",
        "type": "number",
        "default": 35,
        "min": 18,
        "max": 75
    },
    "Attribute14": {
        "label": "其他分期付款计划",
        "type": "dropdown",
        "choices": [
            ("银行分期", "A141"),
            ("商店分期", "A142"),
            ("无", "A143")
        ],
        "default": "A143"
    },
    "Attribute15": {
        "label": "住房状况",
        "type": "dropdown",
        "choices": [
            ("租房", "A151"),
            ("自有住房", "A152"),
            ("免费住房", "A153")
        ],
        "default": "A152"
    },
    "Attribute16": {
        "label": "本行信用卡数量",
        "type": "number",
        "default": 2,
        "min": 1,
        "max": 4
    },
    "Attribute17": {
        "label": "职业类别",
        "type": "dropdown",
        "choices": [
            ("失业/非技术-非居民", "A171"),
            ("非技术-居民", "A172"),
            ("技术工人", "A173"),
            ("管理/个体经营/高技术", "A174")
        ],
        "default": "A173"
    },
    "Attribute18": {
        "label": "赡养人数",
        "type": "number",
        "default": 1,
        "min": 1,
        "max": 3
    },
    "Attribute19": {
        "label": "电话注册情况",
        "type": "dropdown",
        "choices": [
            ("无电话", "A191"),
            ("有注册电话", "A192")
        ],
        "default": "A191"
    },
    "Attribute20": {
        "label": "是否为外籍劳工",
        "type": "dropdown",
        "choices": [
            ("是", "A201"),
            ("否", "A202")
        ],
        "default": "A202"
    }
}

def predict_credit_risk(*args):
    """接收所有特征并预测"""
    # 将args转换为字典（键是特征名，值是用户输入）
    feature_names = list(FEATURE_CONFIG.keys())
    inputs_dict = dict(zip(feature_names, args))
    
    # 创建DataFrame（保持原始特征名）
    input_df = pd.DataFrame([inputs_dict])
    
    # 预测
    proba = model.predict_proba(input_df)[0][1]
    
    # 风险等级
    if proba > 0.7:
        risk = "🔴 高风险"
        color = "#ef4444"
    elif proba > 0.3:
        risk = "🟡 中风险"
        color = "#f59e0b"
    else:
        risk = "🟢 低风险"
        color = "#10b981"
    
    # 返回结果
    return (
        gr.Textbox(value=risk, show_label=True, label="风险等级"),
        gr.Textbox(value=f"{proba:.1%}", show_label=True, label="违约概率")
    )

def create_interface():
    """创建分组优化的Gradio界面"""
    with gr.Blocks(title="智能信用风险评估系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💳 智能信用风险评估系统")
        gr.Markdown("### 请填写以下信息完成信用风险评估")
        
        input_components = []
        
        # 按业务逻辑分组
        with gr.Tab("📊 基础财务信息"):
            with gr.Row():
                attr1 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute1"]["choices"],
                    value=FEATURE_CONFIG["Attribute1"]["default"],
                    label=FEATURE_CONFIG["Attribute1"]["label"]
                )
                attr2 = gr.Number(
                    value=FEATURE_CONFIG["Attribute2"]["default"],
                    label=FEATURE_CONFIG["Attribute2"]["label"],
                    minimum=1,
                    maximum=72
                )
                attr3 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute3"]["choices"],
                    value=FEATURE_CONFIG["Attribute3"]["default"],
                    label=FEATURE_CONFIG["Attribute3"]["label"]
                )
                attr4 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute4"]["choices"],
                    value=FEATURE_CONFIG["Attribute4"]["default"],
                    label=FEATURE_CONFIG["Attribute4"]["label"]
                )
            
            with gr.Row():
                attr5 = gr.Number(
                    value=FEATURE_CONFIG["Attribute5"]["default"],
                    label=FEATURE_CONFIG["Attribute5"]["label"],
                    minimum=100,
                    maximum=20000
                )
                attr6 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute6"]["choices"],
                    value=FEATURE_CONFIG["Attribute6"]["default"],
                    label=FEATURE_CONFIG["Attribute6"]["label"]
                )
                attr7 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute7"]["choices"],
                    value=FEATURE_CONFIG["Attribute7"]["default"],
                    label=FEATURE_CONFIG["Attribute7"]["label"]
                )
                attr8 = gr.Number(
                    value=FEATURE_CONFIG["Attribute8"]["default"],
                    label=FEATURE_CONFIG["Attribute8"]["label"],
                    minimum=1,
                    maximum=10
                )
        
        with gr.Tab("👤 个人信息"):
            with gr.Row():
                attr9 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute9"]["choices"],
                    value=FEATURE_CONFIG["Attribute9"]["default"],
                    label=FEATURE_CONFIG["Attribute9"]["label"]
                )
                attr10 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute10"]["choices"],
                    value=FEATURE_CONFIG["Attribute10"]["default"],
                    label=FEATURE_CONFIG["Attribute10"]["label"]
                )
                attr11 = gr.Number(
                    value=FEATURE_CONFIG["Attribute11"]["default"],
                    label=FEATURE_CONFIG["Attribute11"]["label"],
                    minimum=1,
                    maximum=10
                )
                attr12 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute12"]["choices"],
                    value=FEATURE_CONFIG["Attribute12"]["default"],
                    label=FEATURE_CONFIG["Attribute12"]["label"]
                )
            
            with gr.Row():
                attr13 = gr.Number(
                    value=FEATURE_CONFIG["Attribute13"]["default"],
                    label=FEATURE_CONFIG["Attribute13"]["label"],
                    minimum=18,
                    maximum=75
                )
                attr15 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute15"]["choices"],
                    value=FEATURE_CONFIG["Attribute15"]["default"],
                    label=FEATURE_CONFIG["Attribute15"]["label"]
                )
                attr16 = gr.Number(
                    value=FEATURE_CONFIG["Attribute16"]["default"],
                    label=FEATURE_CONFIG["Attribute16"]["label"],
                    minimum=1,
                    maximum=4
                )
                attr17 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute17"]["choices"],
                    value=FEATURE_CONFIG["Attribute17"]["default"],
                    label=FEATURE_CONFIG["Attribute17"]["label"]
                )
        
        with gr.Tab("🏠 附加信息"):
            with gr.Row():
                attr14 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute14"]["choices"],
                    value=FEATURE_CONFIG["Attribute14"]["default"],
                    label=FEATURE_CONFIG["Attribute14"]["label"]
                )
                attr18 = gr.Number(
                    value=FEATURE_CONFIG["Attribute18"]["default"],
                    label=FEATURE_CONFIG["Attribute18"]["label"],
                    minimum=0,
                    maximum=10
                )
                attr19 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute19"]["choices"],
                    value=FEATURE_CONFIG["Attribute19"]["default"],
                    label=FEATURE_CONFIG["Attribute19"]["label"]
                )
                attr20 = gr.Dropdown(
                    choices=FEATURE_CONFIG["Attribute20"]["choices"],
                    value=FEATURE_CONFIG["Attribute20"]["default"],
                    label=FEATURE_CONFIG["Attribute20"]["label"]
                )
        
        # 结果展示区域
        with gr.Tab("📈 评估结果"):
            with gr.Row():
                risk_output = gr.Textbox(
                    label="风险等级",
                    interactive=False,
                    show_label=True,
                    container=True
                )
                prob_output = gr.Textbox(
                    label="违约概率",
                    interactive=False,
                    show_label=True,
                    container=True
                )
            
            # 解释说明
            gr.Markdown("""
            **风险等级说明：**
            - 🔴 **高风险**: 违约概率 > 70%
            - 🟡 **中风险**: 违约概率 30%-70%
            - 🟢 **低风险**: 违约概率 < 30%
            """)
        
        # 收集所有输入组件
        input_components = [
            attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8,
            attr9, attr10, attr11, attr12, attr13, attr14, attr15, attr16,
            attr17, attr18, attr19, attr20
        ]
        
        # 绑定预测函数
        btn = gr.Button("🚀 开始评估", variant="primary", size="lg")
        btn.click(
            fn=predict_credit_risk,
            inputs=input_components,
            outputs=[risk_output, prob_output]
        )
        
        # 添加示例
        try:
            df_examples = pd.read_csv("./data_outputs/raw/german_credit_full.csv")
            example_values = df_examples.iloc[:5, :-1].values.tolist()
            
            gr.Examples(
                examples=example_values,
                inputs=input_components,
                outputs=[risk_output, prob_output],
                fn=predict_credit_risk,
                cache_examples=False,
                label="使用示例数据测试"
            )
        except Exception as e:
            print(f"⚠️ 无法加载示例数据: {e}")
    
    return demo

if __name__ == "__main__":
    print("正在启动Web界面...")
    demo = create_interface()
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )