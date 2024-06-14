from .learner import Learner, LearnerSpec
from .basic_learner import BasicLearner
from .model_learner import ModelLearner
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory

class LearnerFactory:
    @staticmethod
    def get_learner(learner_spec: LearnerSpec, eval_model) -> Learner:
        metric_name = learner_spec.metric_name
        if metric_name == 'model_metric':
            # load eval model
            model_conf = learner_spec.model_conf
            llm_service = LLMServiceFactory.get_llm_service(
                model_repo_id=eval_model,
                model_config_path=model_conf['model_conf_path'],
                acceleration_method=model_conf['acceleration_method']
            )
            # create model_metric
            return ModelLearner(
                model_conf=model_conf,
                learner_spec=learner_spec,
                llm_service=llm_service,
                learner_prompt_format=learner_spec.learner_prompt_format
            )
        raise ValueError(f'metric name "{metric_name}" is not valid')
