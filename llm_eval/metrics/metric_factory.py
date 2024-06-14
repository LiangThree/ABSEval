from .metric import Metric, MetricSpec
from .basic_metric import BasicMetric
from .model_metric import ModelMetric
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory

class MetricFactory:
    @staticmethod
    def get_metric(metric_spec: MetricSpec) -> Metric:
        metric_name = metric_spec.metric_name
        if metric_name == 'model_metric':
            # load eval model
            model_conf = metric_spec.model_conf
            llm_service = LLMServiceFactory.get_llm_service(
                model_repo_id=model_conf['model_repo_id'], 
                model_config_path=model_conf['model_conf_path'],
                acceleration_method=model_conf['acceleration_method']
            )
            # create model_metric
            return ModelMetric(
                llm_service=llm_service,
                eval_prompt_format=metric_spec.eval_prompt_format
            )
        raise ValueError(f'metric name "{metric_name}" is not valid')
