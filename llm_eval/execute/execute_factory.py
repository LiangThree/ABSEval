from .execute import Execute, ExecuteSpec
from .basic_execute import BasicExecute
from .model_execute import ModelExecute
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory

class ExecuteFactory:
    @staticmethod
    def get_execute(execute_spec: ExecuteSpec) -> Execute:
        metric_name = execute_spec.metric_name
        if metric_name == 'model_metric':
            model_conf = execute_spec.model_conf
            llm_service = LLMServiceFactory.get_llm_service(
                model_repo_id=model_conf['model_repo_id'],
                model_config_path=model_conf['model_conf_path'],
                acceleration_method=model_conf['acceleration_method']
            )
            # create model_metric
            return ModelExecute(
                llm_service=llm_service,
                eval_prompt_format=execute_spec.eval_prompt_format
            )
        raise ValueError(f'metric name "{metric_name}" is not valid')
