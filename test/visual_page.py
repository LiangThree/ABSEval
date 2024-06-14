import dash
from dash import html, callback, Output, Input, State, dcc
from dataclasses import dataclass
import dash_bootstrap_components as dbc
import os
from dash.exceptions import PreventUpdate
# 假设 EvalDataBase 类在 database_util.py 文件中定义
from .visual_database_util import EvalDataBase

dash.register_page(__name__, path='/')
db_path = "/Users/three/MyProject/script-eval/data/database/script.db"

# 数据类
@dataclass
class ModelResponse:
    question_id: str
    question: str
    inference: str
    answer: str
    model_name: str


# 获取数据库文件列表的函数
def get_database_list():
    return [db_path]


# 请求模型响应的函数
def request_model_response(_db, target_view):
    res = _db.sample_one_for_human_annotate_by_view(target_view)
    if res:
        return ModelResponse(**res)
    return ModelResponse("", "All finished", "", "", "")


# 创建数据库选择器
database_selector = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id='database-dropdown',
                options=[{'label': db_name, 'value': db_name} for db_name in get_database_list()],
                value=None,  # 默认没有选择
                placeholder="Select a database",
                style={"width": "100%",
                       "font-family": "Times New Roman"}
            ),
            width=6
        )
    ],
    justify="center",
    className="mb-3"
)

# 创建 target_view 选择器的框架（具体选项将由回调动态生成）
target_view_selector = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id='target-view-dropdown',
                options=[],  # 初始为空，将由回调函数填充
                value=None,  # 默认没有选择
                placeholder="Select a target view",
                style={"width": "100%",
                       "font-family": "Times New Roman"}
            ),
            width=6
        )
    ],
    justify="center",
    className="mb-3"
)


# 创建按钮组
button_group = dbc.Row(
    [
        dbc.Col(dbc.Button("Previous answer", color="secondary", id='button-previous', className="me-2"), width='auto'),
        dbc.Col(dbc.Button("Next answer", color="secondary", id='button-next'), width='auto'),
    ],
    justify="center",
    className="mb-3"
)

# 创建界面的主要部分
arena_card = dbc.Col(
    children=[
        dbc.Row(
            dbc.Col(
                children=[
                    html.H4("Question & Answer"),
                    html.P(id="question"),
                    html.P(id="answer")
                ],
                style={
                    "background-color": "rgba(240, 242, 246, 0.5)",
                    "padding": "30px 0px 30px 30px",
                    "border-radius": "1rem 1rem 0 0",
                    "font-family": "Times New Roman",
                }
            ),
        ),
        html.Hr(style={"margin": 0}),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        html.H4(id="model_name"),
                        html.P(id="inference")
                    ],
                    style={
                        "padding": "20px",
                        "font-family": "Times New Roman",
                    }
                )
            ]
        ),
        html.Hr(style={"margin": "0 0 30px 0"}),
        button_group,
        html.Div(style={"margin": "100px"}),
    ],
    width={"size": 8, "offset": 2},
)

# 回调函数1：当数据库选择变更时，更新 target_view 的选项
@callback(
    Output('target-view-dropdown', 'options'),
    Output('target-view-dropdown', 'value'),
    Input('database-dropdown', 'value')
)

def update_target_view_options(selected_db_name):
    if selected_db_name:
        db = EvalDataBase(db_path)
        db.table_create()  # 确保数据库包含所有需要的表
        # 获取新的 target_views 列表
        target_views = db.get_target_views_in_qa()
        # 更新下拉选项
        return [{'label': view, 'value': view} for view in target_views], None
    return [], None  # 如果没有选择数据库，清空 target_view 的选项并设置默认值为 None


# 合并回调：处理 target_view 选择变更和按钮点击事件
@callback(
    Output('question', 'children'),
    Output('model_name', 'children'),
    Output('inference', 'children'),
    Output('answer', 'children'),
    Output('storage', 'data'),
    Output('sample-history', 'data'),
    Input('target-view-dropdown', 'value'),
    Input('button-previous', 'n_clicks'),
    Input('button-next', 'n_clicks'),
    State('database-dropdown', 'value'),
    State('storage', 'data'),
    State('sample-history', 'data')
)

def combined_callback(selected_target_view, button_previous, button_next, selected_db_name, stored_data, history_data):
    
    ctx = dash.callback_context

    # 检查是哪个输入触发了回调
    if not ctx.triggered:
        button_id = 'No buttons yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    db = EvalDataBase(db_path)
    

    if button_id == 'target-view-dropdown' and selected_target_view:
        # 如果回调是由 target_view 下拉菜单触发的
        all_question_with_all_inferences = db.sample_from_llm_inference()
        one_question_with_all_inferences = all_question_with_all_inferences[0]
        
        if one_question_with_all_inferences:
            all_inferences = []
            for one_inference in one_question_with_all_inferences:
                model_response = ModelResponse(**one_inference)
                stored_data = {
                    'question_id': model_response.question_id,
                    'question': model_response.question,
                    'model_name': model_response.model_name,
                    'answer': model_response.answer,
                    'selected_db_name': selected_db_name  # 确保也存储数据库名
                }
                all_inferences.append(stored_data)
            db.close()
            return model_response.question, model_response.model_name, model_response.inference, model_response.answer, stored_data, all_inferences
        else:
            db.close()
            # return "No data available", "", "", stored_data, all_inferences
    
    if button_id == 'button-previous' and button_previous:
        if history_data and len(history_data) > 0:  # 确保有历史记录可回退
            
            previous_sample = history_data[-1]  # 获取上一个样本
            print(previous_sample)
            
            print(previous_sample)
            start = history_data[-1]
            for i in range(len(history_data)-1):
                history_data[i+1] = history_data[i]
            history_data[0] = start

            # 设置上一个样本为当前样本
            stored_data = {
                'question_id': previous_sample['question_id'],
                'question': previous_sample['question'],
                'model_name': previous_sample['model_name'],
                'answer': previous_sample['answer'],
                'selected_db_name': selected_db_name  # 确保也存储数据库名
            }
            # 获取上一个样本的具体数据
            prev_sample = db.get_sample_data(previous_sample['question_id'], previous_sample['model_name'])
            model_response = ModelResponse(**prev_sample)
            # 返回更新的样本数据和历史记录
            db.close()
            return model_response.question, model_response.model_name, model_response.inference, model_response.answer, stored_data, history_data
        else:
            db.close()
            raise PreventUpdate  # 没有历史记录可回退时不进行更新
    
    if button_id == 'button-next' and button_next:
        if history_data and len(history_data) > 0:  # 确保有历史记录可回退
            previous_sample = history_data[1]  # 获取上一个样本

            print(previous_sample)
            start = history_data[0]
            for i in range(len(history_data)-1):
                history_data[i] = history_data[i+1]
            history_data[-1] = start

            # 设置上一个样本为当前样本
            stored_data = {
                'question_id': previous_sample['question_id'],
                'question': previous_sample['question'],
                'model_name': previous_sample['model_name'],
                'answer': previous_sample['answer'],
                'selected_db_name': selected_db_name  # 确保也存储数据库名
            }
            # 获取上一个样本的具体数据
            prev_sample = db.get_sample_data(previous_sample['question_id'], previous_sample['model_name'])
            model_response = ModelResponse(**prev_sample)
            # 返回更新的样本数据和历史记录
            db.close()
            return model_response.question, model_response.model_name, model_response.inference, model_response.answer, stored_data, history_data
        else:
            db.close()
            raise PreventUpdate  # 没有历史记录可回退时不进行更新

    # 如果没有触发源或其他条件不满足，则不更新任何内容
    db.close()
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, stored_data, history_data


# 页面布局
layout = html.Div(
    [
        html.Div(database_selector),
        html.Div(target_view_selector),
        arena_card,
        dcc.Store(id='storage'),  # 用于在回调之间存储数据
        dcc.Store(id='sample-history', storage_type='session')
    ]
)