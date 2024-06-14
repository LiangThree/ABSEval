import pdb
import yaml
from pathlib import Path

from .openai_service import OpenaiService
from .local_model_service import LocalModelService
from .vllm_model_service import VllmModelService
from .service import Service


class LLMServiceFactory:
    @staticmethod
    def get_llm_service(
            model_repo_id,
            model_config_path=None,
            acceleration_method='default'
    ) -> Service:
        if type(model_repo_id) == list:
            model_repo_id = model_repo_id[0]
        model_org, model_name = model_repo_id.split('/')
        if model_org == 'openai':
            return OpenaiService(model_name)
        if acceleration_method == 'default':
            return LocalModelService(model_repo_id, model_config_path)
        elif acceleration_method == 'vllm':
            return VllmModelService(model_repo_id, model_config_path, model_name)
