"""
该模块实现了Runner类；
该类是整个大模型评估项目的入口类，用于执行数据读取、模型inference和计算结果；
"""
import json
import ipdb
import sqlite3
import random
import ray
from dataclasses import dataclass
from typing import List
from pathlib import Path


from tqdm import tqdm
from pprint import pprint
from data.database.util.database_util import EvalDataBase

from llm_eval.data.instance import Instance
from llm_eval.data.request import Request, RequestResult, RequestLearner
from llm_eval.scenarios.scenario import Scenario
from llm_eval.scenarios.scenario_factory import ScenarioFactory
from llm_eval.adaptation.adapter import Adapter
from llm_eval.adaptation.adapter_factory import AdapterFactory
from llm_eval.executor import Executor

from llm_eval.metrics.metric import Metric, MetricSpec
from llm_eval.metrics.metric_factory import MetricFactory
from llm_eval.metrics.statistic import Stat
from llm_eval.data.request import RequestMetric
from llm_eval.data.instance import Input, Output, Reference, CORRECT_TAG
from llm_eval.data.request import Prompt
from llm_eval.scenarios.scenario import TEST_SPLIT

from llm_eval.learner.learner import Learner, LearnerSpec
from llm_eval.learner.learner_factory import LearnerFactory

@dataclass
class RunSpec:
    db_path: Path
    scenario_conf: dict
    adapter_method: str
    model_conf: dict
    metric_spec: MetricSpec
    

@dataclass
class EvalRunSpec:
    db_path: str
    target_view: str
    inference_model_repo_id: str
    metric_spec: MetricSpec


@dataclass
class LearnRunSpec:
    db_path: str
    category: str
    inference_model_repo_id: str
    learner_spec: LearnerSpec


def load_models(model_conf: dict):
    """
    从model_conf中读取模型信息，返回模型的id和模型的路径
    """
    pass


class MetaRunner:
    @staticmethod
    def get_instances(run_spec, test=False) -> List[Instance]:
        print('get instances...')
        scenario: Scenario = ScenarioFactory.get_scenario_from_dict(run_spec.scenario_conf)
        instances: List[Instance] = scenario.get_instances()
        if test is True:
            test_instances = [i for i in instances if i.split == 'test']
            train_instances = [i for i in instances if i.split == 'train']
            instances = random.sample(test_instances, 17) + random.sample(train_instances, 5)
        print(f'instances count: {len(instances)}')
        # pprint(instances[:1])
        return instances

    @staticmethod
    def convert_instances_to_requests(run_spec: RunSpec, instances: List[Instance]) -> List[Request]:
        print('adapt instances...')
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_method)
        requests: List[Request] = adapter.adapt(instances)
        # pprint(requests[:1])
        return requests
        
    @staticmethod
    def execute(run_spec, requests: List[Request]) -> List[RequestResult]:
        print('execute requests...')
        executor: Executor = Executor(run_spec=run_spec)
        request_results: List[RequestResult] = executor.execute(requests)
        # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
        if ray.is_initialized():
            # 如果已经初始化，先关闭现有的Ray实例
            ray.shutdown()
        del executor # 释放显卡空间
        return request_results
    
    @staticmethod
    def update_inference_table(run_spec, eval_db, request_results: List[RequestResult]):
        print('update inference table...')
        for request_result in tqdm(request_results):
            question_id = request_result.request.instance.id
            prompt_text = request_result.request.prompt.text
            completion = request_result.completion
            model_repo_id = run_spec.model_conf['model_repo_id']
            data = [question_id, model_repo_id, completion]
            try:
                eval_db.insert_into_llm_inference(data)
            except sqlite3.IntegrityError:
                eval_db.update_llm_inference(data)
                
    @staticmethod
    def compute_metrics(run_spec, request_results: List[RequestResult]) -> List[RequestMetric]:
        print('compute metrics...')
        metric: Metric = MetricFactory.get_metric(run_spec.metric_spec)
        request_metrics: List[RequestMetric] = metric.compute(request_results)
        # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
        if ray.is_initialized():
            # 如果已经初始化，先关闭现有的Ray实例
            ray.shutdown()
        del metric  # 释放显卡空间
        return request_metrics

    @staticmethod
    def compute_learner(run_spec, request_results: List[RequestResult], follow) -> List[RequestMetric]:
        print('compute learner...')
        print(run_spec.learner_spec.model_conf['model_repo_id'])
        eval_model_list = run_spec.learner_spec.model_conf['model_repo_id']
        if follow and 'openai/gpt-3.5-turbo' in eval_model_list:
            eval_model_list.remove('openai/gpt-3.5-turbo')
        else:
            eval_model_list = [eval_model_list[0]]
        for eval_model in eval_model_list:
            learner: Learner = LearnerFactory.get_learner(run_spec.learner_spec, eval_model)
            learner.compute(request_results)
            # request_metrics: List[RequestMetric] = learner.compute(request_results)
            # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
            if ray.is_initialized():
                # 如果已经初始化，先关闭现有的Ray实例
                ray.shutdown()
            del learner  # 释放显卡空间
            print('delete learner done')
        # return request_metrics


    @staticmethod
    def update_gold_answer(run_spec, eval_db: EvalDataBase, request_metrics: List[RequestMetric]):
        print('update eval_result table...')
        for request_metric in tqdm(request_metrics):
            request_metric: RequestMetric
            if request_metric.success:
                question_id = request_metric.request_result.request.instance.id
                model_repo_id = run_spec.model_conf['model_repo_id']
                eval_model_name = run_spec.metric_spec.model_conf['model_repo_id']
                if eval_model_name is None:
                    eval_model_name = 'none'
                eval_result = request_metric.evaluation
                data = [question_id, model_repo_id, eval_model_name, eval_result]
                try:
                    eval_db.insert_into_eval_result(*data)
                except sqlite3.IntegrityError:
                    eval_db.update_eval_result(*data)

class Runner(MetaRunner):
    def run_one(self, run_spec, test=False):
        # load db
        eval_db = EvalDataBase(run_spec.db_path)
        
        # get instances
        instances: List[Instance] = self.get_instances(run_spec, test)
        
        # convert instances to requests
        requests: List[Request] = self.convert_instances_to_requests(run_spec, instances)

        # execute requests
        request_results: List[RequestResult] = self.execute(run_spec, requests)
        
        # update inference table
        self.update_inference_table(run_spec, eval_db, request_results)
        
        # compute metrics
        request_metrics: List[RequestMetric] = self.compute_metrics(run_spec, request_results)
        
        # update eval_result table
        self.update_gold_answer(run_spec, eval_db, request_metrics)
        
        # 计算统计结果stat
        stat = Stat(name='base')
        for request_metric in request_metrics:
            if request_metric.success is True:
                stat.add(request_metric.stat.mean)
                stat.valid_count += 1
            else:
                stat.invalid_count += 1
        return request_metrics, stat
                    
class InferenceRunner(MetaRunner):
    def run_inference(self, run_spec, test=False):
        # load db
        eval_db = EvalDataBase(run_spec.db_path)
        
        # get instances
        instances: List[Instance] = self.get_instances(run_spec, test)
        
        # convert instances to requests
        requests: List[Request] = self.convert_instances_to_requests(run_spec, instances)
        
        # execute requests
        request_results: List[RequestResult] = self.execute(run_spec, requests)
        
        # update inference table
        self.update_inference_table(run_spec, eval_db, request_results)


class LearnerRunner(MetaRunner):

    @staticmethod
    def load_llm_inference_rows(learnerSpec: LearnerSpec, only_annotated_instance=False):
        print('load question_ids by category and model_repo_id in target llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(learnerSpec.db_path)
        category = learnerSpec.category

        # 只获取需要共同构成答案的模型回答
        inference_model_repo_ids = learnerSpec.inference_model_repo_id

        if inference_model_repo_ids[0] == 'ALL':
            query = """
            SELECT question_id, model_name, inference FROM llm_inference
            """
        else:
            # 构建值列表字符串
            value_list_str = ", ".join(["'{}'".format(value) for value in inference_model_repo_ids])

            if only_annotated_instance:
                query = f"""
                SELECT llm_inference.question_id, llm_inference.model_name, llm_inference.inference from llm_inference 
                INNER JOIN human_eval ON llm_inference.question_id=human_eval.question_id AND llm_inference.model_name=human_eval.model_name
                """
            else:
                query = f"""
                SELECT llm_inference.question_id, llm_inference.model_name, llm_inference.inference 
                FROM llm_inference JOIN question ON llm_inference.question_id = question.question_id
                WHERE llm_inference.model_name in ({value_list_str}) AND question.category like '{category}'
                """
        # print(query)

        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows


    def load_follow_rows(self, eval_run_spec: LearnRunSpec):
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        category = eval_run_spec.category
        question_ids = eval_db.get_gpt_3_question_ids()

        category_question_ids = question_ids[category]
        inference_model_repo_ids = eval_run_spec.inference_model_repo_id
        question_id_list_str = ", ".join(["'{}'".format(value) for value in category_question_ids])

        if inference_model_repo_ids[0] == 'ALL':
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str})"
        else:
            model_list_str = ", ".join(["'{}'".format(value) for value in inference_model_repo_ids])
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str}) AND model_name IN ({model_list_str})"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()

        return llm_inference_rows

    def load_annotated_rows(self, eval_run_spec: LearnRunSpec):
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        model_name =eval_run_spec.learner_spec.model_conf['model_repo_id'][0]
        index = model_name.find('/')
        if index != -1:
            model_name = model_name[index + 1:]
        # category = eval_run_spec.category
        # question_ids = eval_db.get_annotated_question_ids(category)
        # question_ids = random.sample(question_ids, 30)
        # question_id_list_str = ", ".join(["'{}'".format(value) for value in question_ids])

        # inference_model_repo_ids = eval_run_spec.inference_model_repo_id
        # model_list_str = ", ".join(["'{}'".format(value) for value in inference_model_repo_ids])
        # query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE EXISTS(SELECT 1 FROM human_eval WHERE llm_inference.question_id = human_eval.question_id AND llm_inference.model_name = human_eval.model_name) AND NOT EXISTS(SELECT 1 FROM gold_answer WHERE llm_inference.question_id = gold_answer.question_id AND  gold_answer.model_name = '{model_name}')"
        query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE EXISTS(SELECT 1 FROM human_eval WHERE llm_inference.question_id = human_eval.question_id AND llm_inference.model_name = human_eval.model_name)"

        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        # print(len(llm_inference_rows))
        # exit(0)

        return llm_inference_rows

    @staticmethod
    def load_num_instances_rows(eval_run_spec: LearnRunSpec, num_instances):

        print(f'load {num_instances} instances from llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        category = eval_run_spec.category
        question_ids = eval_db.get_question_ids_per_view()
        category_question_ids = question_ids[category]
        sample_question_ids = random.sample(category_question_ids, num_instances)

        inference_model_repo_ids = eval_run_spec.inference_model_repo_id
        question_id_list_str = ", ".join(["'{}'".format(value) for value in sample_question_ids])

        if inference_model_repo_ids[0] == 'ALL':
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str})"
        else:
            model_list_str = ", ".join(["'{}'".format(value) for value in inference_model_repo_ids])
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str}) AND model_name IN ({model_list_str})"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()

        return llm_inference_rows

    def load_all_instances_rows(self, eval_run_spec: LearnRunSpec):
        print(f'load all instances from llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        category = eval_run_spec.category
        question_ids = eval_db.get_question_ids_per_view()

        if category == 'ALL':
            category_question_ids = []
            for category in question_ids.keys():
                category_question_ids.extend(question_ids[category])
        else:
            category_question_ids = question_ids[category]

        eval_model = eval_run_spec.learner_spec.model_conf['model_repo_id'][0]
        index = eval_model.find('/')
        if index != -1:
            eval_model = eval_model[index+1:]
        question_ids_with_gold_answer = eval_db.get_gold_answer_question_ids(eval_model)
        category_question_ids = [item for item in category_question_ids if item not in question_ids_with_gold_answer]

        inference_model_repo_ids = eval_run_spec.inference_model_repo_id
        question_id_list_str = ", ".join(["'{}'".format(value) for value in category_question_ids])

        if inference_model_repo_ids[0] == 'ALL':
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str})"
        else:
            model_list_str = ", ".join(["'{}'".format(value) for value in inference_model_repo_ids])
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id IN ({question_id_list_str}) AND model_name IN ({model_list_str})"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()

        return llm_inference_rows

    @staticmethod
    def load_instance(question_id: str, eval_db: EvalDataBase) -> Instance:
        query = "SELECT * FROM question WHERE question_id=?"
        row = eval_db.conn.execute(query, (question_id,)).fetchone()
        references = [Reference(output=Output(row[6]), tag=CORRECT_TAG)]
        instance = Instance(input=Input(text=row[4]), references=references, id=row[0])
        return instance

    @staticmethod
    def build_sudo_request(instance: Instance) -> Request:
        """这里sudo的意思是prompt不是真实的"""
        adapter_name = 'generation_adapter'
        prompt = Prompt(instance=instance, adapter_name=adapter_name, text=None)
        request = Request(instance=instance, prompt=prompt, question_type='qa_question')
        return request

    @staticmethod
    def build_sudo_request_result(request: Request, completion: str) -> RequestResult:
        """这里sudo的意思是prompt不是真实的"""
        request_result: RequestLearner = RequestLearner(request=request, success=True, completion=completion)
        return request_result

    def load_request_results(
            self,
            eval_run_spec: LearnRunSpec,
            eval_db: EvalDataBase,
            follow=False,
            num_instances=-1,
            annotated=False
    ) -> List[RequestResult]:

        if follow:
            llm_inference_rows = self.load_follow_rows(eval_run_spec)
        elif annotated:
            llm_inference_rows = self.load_annotated_rows(eval_run_spec)
        elif num_instances > 0:
            llm_inference_rows = self.load_num_instances_rows(eval_run_spec, num_instances)
        else:
            llm_inference_rows = self.load_all_instances_rows(eval_run_spec)

        distinct_question_ids = [question_id for question_id, _, _ in llm_inference_rows]
        distinct_question_ids = list(set(distinct_question_ids))

        # load instances
        instances = [self.load_instance(question_id, eval_db) for question_id in distinct_question_ids]

        # convert instances to requests
        requests: List[Request] = [self.build_sudo_request(instance) for instance in instances]

        # load request_results
        question_ids = []
        completions = []

        for row in llm_inference_rows:
            if row[2] is not None:
                question_ids.append(row[0])
                completions.append(row[2][:2000])

        answer_group = {}
        for index, question_id in enumerate(question_ids):
            if question_id in answer_group.keys():
                answer_group[question_id].append(completions[index])
            else:
                answer_group[question_id] = [completions[index]]

        request_results: List[RequestResult] = [
            self.build_sudo_request_result(request, completion) for request, completion in
            zip(requests, list(answer_group.items()))
        ]

        return request_results

    @staticmethod
    def update_gold_answer(run_spec: EvalRunSpec, eval_db: EvalDataBase, request_metrics: List[RequestMetric]):
        print('update gold answer table...')
        for request_metric in tqdm(request_metrics):
            request_metric: RequestMetric
            # print('-----RequestMetric-----')
            # print(request_metric.learn_response)
            question_id = request_metric.request_result.request.instance.id
            eval_result = request_metric.learn_response
            data = [question_id, eval_result]
            try:
                eval_db.insert_into_gold_answer(*data)
            except sqlite3.IntegrityError:
                eval_db.insert_into_gold_answer(*data)

    def run_learner(self, learner_run_spec: LearnRunSpec, args):
        # these args are used to filter instances
        follow = args.follow
        num_instances = args.num_instances
        annotated = args.annotated

        # load db
        eval_db: EvalDataBase = EvalDataBase(learner_run_spec.db_path)

        # load request_results
        request_results: List[RequestResult] = self.load_request_results(
            learner_run_spec, eval_db, follow=follow, num_instances=num_instances,
            annotated=annotated)


        # compute metrics
        self.compute_learner(learner_run_spec, request_results, follow)

