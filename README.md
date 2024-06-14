### 脚本评测框架
###### 功能模块：
主要的功能模块有四部分：
1、inference：运行各个模型进行关于script question的推理；
2、learner：使用我们的方法学习全部的脚本Chans标准答案；
3. eval：根据标准答案对每一个脚本进行评价
3、execute：执行关于每一个步骤的评价
4、commonsense：常识检查；

###### 配置文件：
- 模型配置文件：
  模型配置文件在config/model_config.yaml中定义；
  如下所示是一个模型的配置信息；
  ```
  baichuan-inc:
    Baichuan-7B:
      model_type: base
      path: /home/zbl/data/llm/baichuan-inc/Baichuan-7B
  ```

- 运行配置：
  运行配置文件在config/目录中中定义；配置文件是json格式，是一个list，每一个list包含一次数据加载到评估的整个过程，也就是说一次程序运行可以执行多次上述过程；
```json
[
  {
    "db_path": "data/database/script.db",
    "model_conf_path": "config/model_config_docker.yaml",
    "target_view": [
      "1"
    ],
    "others": [
      "lmsys/vicuna-7b-v1.5",
	  "lmsys/vicuna-13b-v1.5",
	  "WizardLM/WizardLM-13B-V1.2",
	  "meta/llama2-7b-chat",
	  "meta/llama2-13b-chat",
	  "meta/Llama-2-70b-chat",
	  "qwen/Qwen-7B-Chat",
	  "qwen/Qwen-14B-Chat",
	  "qwen/Qwen-72B-Chat",
	  "01ai/Yi-6B-Chat",
	  "01ai/Yi-34B-Chat",
      "baichuan-inc/Baichuan2-7B-Chat",
      "baichuan-inc/Baichuan2-13B-Chat",
      "baichuan-inc/Baichuan-13B-Chat"

    ],
    "inference_model_repo_id": [
      "THUDM/chatglm3-6b",
	  "mistralai/Mistral-7B-Instruct-v0.2",
	  "mistralai/Mistral-7B-Instruct-v0.1"
    ],
    "metric_conf": {
      "metric_name": "model_metric",
      "model_repo_id": [
	    "openai/chatgpt"
      ],
      "acceleration_method": "vllm",
      "eval_prompt_format": "path:data/metrics/eval_prompt.txt"
    }
  }
]
```
如上述配置所示，选择题需要修改的配置有：
- db_path: 数据库路径；
- target_view: 问题难度；
- model_repo_id: 模型配置文件中的模型名称；

###### 运行命令
运行模型推理
python scripts/run_inference.py --run-specs config/run_specs.json 

运行模型学习
python scripts/run_learner.py --run-specs config/run_learner_specs.json --num-instances 5

运行模型评价
python scripts/run_eval.py --run-specs config/run_eval_specs.json --num-instance 5

运行执行者
python scripts/run_execute.py --run-specs config/run_execute_specs.json 

运行常识检查
python scripts/run_commonsense.py
