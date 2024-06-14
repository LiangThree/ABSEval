import json
import sqlite3
import random
from typing import List
from llm_eval.data.request import Request, RequestResult, RequestLearner
from llm_eval.runner import MetaRunner, RunSpec, Runner, EvalRunSpec
from data.database.util.database_util import EvalDataBase
from llm_eval.data.instance import Instance
from tqdm import tqdm
import ray

from llm_eval.execute.execute import Execute, ExecuteSpec
from llm_eval.execute.execute_factory import ExecuteFactory
from llm_eval.data.request import RequestMetric
from llm_eval.data.instance import Input, Output, Reference, CORRECT_TAG
from llm_eval.data.request import Prompt
from llm_eval.metrics.metric import Metric, MetricSpec
from llm_eval.metrics.metric_factory import MetricFactory


class ExecuteRunner(MetaRunner):
    def load_question_ids(run_spec: RunSpec):
        print('load question_ids by target_view and model_repo_id in target llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(run_spec.db_path)

        from colorama import Fore
        print(
            Fore.YELLOW + 'load question_ids by target_view and model_repo_id in target llm_inference table...' + Fore.RESET)

        category = run_spec.scenario_conf['category']
        table_name = run_spec.scenario_conf['table_name']
        model_repo_id = run_spec.model_conf['model_repo_id']
        rows: List[tuple] = eval_db.select_question_ids_from_llm_inference(category, model_repo_id, table_name)
        question_ids: List[dir] = [row[0] for row in rows]
        print(len(question_ids))
        return question_ids

    def load_llm_inference_rows(self, eval_run_spec: EvalRunSpec, eval_results: List[tuple]):
        print('load llm_inference with gold answer...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        inferences = []
        no_inference_count = 0
        for eval_model_name, question_id, model_name in tqdm(eval_results, desc='load execute question'):
            query = f"SELECT question_id, model_name, inference FROM llm_inference WHERE question_id = '{question_id}' AND model_name = '{model_name}'"
            try:
                one_inference = eval_db.conn.execute(query).fetchall()
            except:
                print(query)
                exit(0)

            if one_inference[0][2] is not None:
                inferences.append(one_inference[0])
            else:
                no_inference_count += 1
        print(f'no inferences count: {no_inference_count}'"")
        return inferences

    def load_instance(self, question_id: str, eval_db: EvalDataBase) -> Instance:
        query = "SELECT * FROM question WHERE question_id=?"
        row = eval_db.conn.execute(query, (question_id,)).fetchone()

        references = [Reference(output=Output(row[3]), tag=CORRECT_TAG), Reference(output=Output(row[6]), tag=CORRECT_TAG)]
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
    def build_sudo_request_result(request: Request, model_repo_id: str, completion: str) -> RequestResult:
        """这里sudo的意思是prompt不是真实的"""
        request_result: RequestResult = RequestResult(request=request, success=True, completion=completion,
                                                      model_repo_id=model_repo_id)
        return request_result

    def load_request_results(
            self,
            eval_run_spec: RunSpec,
            eval_db: EvalDataBase,
            test=False,
            num_instances=-1,
            only_annotated_instance=False,
            interfere=False
    ) -> List[RequestResult]:

        eval_model = eval_run_spec.metric_spec.model_conf['model_repo_id']
        eval_model = eval_model.split('/')[-1]

        if interfere:
            llm_inference_rows = self.load_interfere_rows(eval_run_spec)
        else:
            eval_results = eval_db.select_eval_result_ids(eval_model)
            llm_inference_rows = self.load_llm_inference_rows(eval_run_spec, eval_results)

            print('executor eval number:', len(llm_inference_rows))

        if num_instances != -1:
            llm_inference_rows = random.sample(llm_inference_rows, num_instances)

        # load instances
        # question instance
        instances = [self.load_instance(question_id, eval_db) for question_id, _, _ in llm_inference_rows]

        # convert instances to requests
        requests: List[Request] = [self.build_sudo_request(instance) for instance in instances]

        # load request_results
        model_repo_ids = [row[1] for row in llm_inference_rows]
        completions = [row[2] for row in llm_inference_rows]
        request_results: List[RequestResult] = [
            self.build_sudo_request_result(request, model_repo_id, completion) for request, model_repo_id, completion in
            zip(requests, model_repo_ids, completions)
        ]

        return request_results

    @staticmethod
    def compute_execute(run_spec, request_results: List[RequestResult], interfere=True):
        print('compute metrics...')
        execute: Execute = ExecuteFactory.get_execute(run_spec.metric_spec)
        execute.compute(request_results, interfere)
        # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
        if ray.is_initialized():
            # 如果已经初始化，先关闭现有的Ray实例
            ray.shutdown()
        del execute  # 释放显卡空间

    def update_eval_result(self, run_spec: EvalRunSpec, eval_db: EvalDataBase, request_metrics: List[RequestMetric]):
        print('update eval_result table...')
        for request_metric in tqdm(request_metrics):
            request_metric: RequestMetric
            if request_metric.success:
                question_id = request_metric.request_result.request.instance.id
                model_name = request_metric.request_result.model_repo_id
                eval_result = request_metric.evaluation
                eval_result = json.loads(eval_result)
                limitation = eval_result['limitation']
                complete = eval_result['complete']
                step_order = eval_result['step_order']
                explain = eval_result['explain']

                data = [question_id, model_name, limitation, complete, step_order, explain]

                try:
                    eval_db.add_result_in_eval_result(*data)
                except sqlite3.IntegrityError:
                    # eval_db.update_eval_result(*data)
                    pass

    @staticmethod
    def load_interfere_rows(eval_run_spec: EvalRunSpec):
        print('load inference from interfere...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        query = f"SELECT question_id, model_name, inference FROM interfere WHERE limitation is null AND interfere_category in ('limitation','complete','step_order') "
        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows

    def run_metric(self, eval_run_spec: EvalRunSpec, args):
        # these args are used to filter instances
        test = args.test
        num_instances = args.num_instances
        only_annotated_instance = args.only_annotated_instance
        interfere = args.interfere

        # load db
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)

        # load request_results
        request_results: List[RequestResult] = self.load_request_results(
            eval_run_spec, eval_db, test=test, num_instances=num_instances,
            only_annotated_instance=only_annotated_instance, interfere=interfere)

        # compute metrics
        self.compute_execute(eval_run_spec, request_results, interfere)


class MetricRunner(MetaRunner):
    def load_question_ids(run_spec: RunSpec):
        print('load question_ids by target_view and model_repo_id in target llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(run_spec.db_path)

        from colorama import Fore
        print(Fore.BLUE + f"{run_spec}." + Fore.RESET)

        category = run_spec.scenario_conf['category']
        table_name = run_spec.scenario_conf['table_name']
        model_repo_id = run_spec.model_conf['model_repo_id']
        rows: List[tuple] = eval_db.select_question_ids_from_llm_inference(category, model_repo_id, table_name)
        question_ids: List[dir] = [row[0] for row in rows]
        print(len(question_ids))
        return question_ids

    @staticmethod
    def compute_execute(run_spec, request_results: List[RequestResult]):
        print('compute metrics...')
        execute: Execute = ExecuteFactory.get_execute(run_spec.metric_spec)
        execute.compute(request_results)
        # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
        if ray.is_initialized():
            # 如果已经初始化，先关闭现有的Ray实例
            ray.shutdown()
        del execute  # 释放显卡空间

    @staticmethod
    def load_llm_inference_rows(eval_run_spec: EvalRunSpec, gold_answers: List[tuple]):
        print('load llm_inference with gold answer...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        question_id_str = ", ".join(["'{}'".format(value) for value in [item[0] for item in gold_answers]])
        # 这里有改动！！！
        query = f"SELECT question_id, model_name, inference FROM llm_inference li WHERE question_id IN ({question_id_str}) AND li.inference is NOT NULl AND NOT EXISTS( SELECT 1 FROM eval_result er WHERE  li.question_id=er.question_id AND li.model_name=er.model_name)"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows

    @staticmethod
    def load_interfere_rows(eval_run_spec: EvalRunSpec):
        print('load inference from interfere...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        query = f"SELECT question_id, model_name, inference FROM interfere WHERE missing_steps is null AND interfere_category in ('missing_steps','duplicate_steps','redundant_steps')"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows

    @staticmethod
    def load_annotated_rows(eval_run_spec, eval_model):
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        model_name = eval_run_spec.metric_spec.model_conf['model_repo_id']
        index = model_name.find('/')
        if index != -1:
            model_name = model_name[index + 1:]

        query = f"""
            SELECT question_id, model_name, inference 
            FROM llm_inference li 
            WHERE EXISTS(
                SELECT 1 FROM human_eval he 
                WHERE  li.question_id=he.question_id AND li.model_name=he.model_name
            )AND NOT EXISTS (
                SELECT 1 FROM eval_result er 
                WHERE li.question_id=er.question_id AND li.model_name=er.model_name AND er.eval_model_name='{model_name}')
        """

        llm_inference_rows = eval_db.conn.execute(query).fetchall()

        return llm_inference_rows


    @staticmethod
    def load_follow_rows(eval_run_spec: EvalRunSpec, eval_model):
        print('load follow inference from interfere...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        query = (f"SELECT question_id, model_name, inference "
                 f"FROM llm_inference "
                 f"WHERE EXISTS("
                 f" SELECT 1 FROM eval_result "
                 f" WHERE eval_result.eval_model_name LIKE 'gpt-3.5-turbo' AND eval_result.question_id = llm_inference.question_id AND eval_result.model_name = llm_inference.model_name)"
                 f"AND NOT EXISTS("
                 f" SELECT 1 FROM eval_Result WHERE eval_result.eval_model_name LIKE '{eval_model}' AND eval_Result.question_id = llm_inference.question_id AND eval_Result.model_name = llm_inference.model_name"
                 f")")
        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows

    @staticmethod
    def load_llm_gold_answers(eval_run_spec: EvalRunSpec, eval_model):
        print('load gold answers...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        query = """
            SELECT question_id, answer from gold_answer WHERE model_name = ?
        """
        gold_answers = eval_db.conn.execute(query, (eval_model,)).fetchall()
        return gold_answers

    @staticmethod
    def load_instance(question_id: str, eval_db: EvalDataBase) -> Instance:
        query = "SELECT * FROM question WHERE question_id=?"
        row = eval_db.conn.execute(query, (question_id,)).fetchone()

        query = "SELECT * FROM gold_answer WHERE question_id=?"
        row_answer = eval_db.conn.execute(query, (question_id,)).fetchone()


        references = [Reference(output=Output(row_answer), tag=CORRECT_TAG)]
        instance = Instance(input=Input(text=row[6]), references=references, id=row[0])
        return instance

    @staticmethod
    def build_sudo_request(instance: Instance) -> Request:
        """这里sudo的意思是prompt不是真实的"""
        adapter_name = 'generation_adapter'
        prompt = Prompt(instance=instance, adapter_name=adapter_name, text=None)
        request = Request(instance=instance, prompt=prompt, question_type='qa_question')
        return request

    @staticmethod
    def build_sudo_request_result(request: Request, model_repo_id: str, completion: str) -> RequestResult:
        """这里sudo的意思是prompt不是真实的"""
        request_result: RequestResult = RequestResult(request=request, success=True, completion=completion,
                                                      model_repo_id=model_repo_id)
        return request_result

    def load_request_results(
            self,
            eval_run_spec: RunSpec,
            eval_db: EvalDataBase,
            follow=False,
            num_instances=-1,
            annotated=False,
            interfere=False
    ) -> List[RequestResult]:

        eval_model = eval_run_spec.metric_spec.model_conf['model_repo_id']
        eval_model = eval_model.split('/')[-1]

        if interfere:
            llm_inference_rows = self.load_interfere_rows(eval_run_spec)
        elif follow:
            llm_inference_rows = self.load_follow_rows(eval_run_spec, eval_model)
        elif annotated:
            llm_inference_rows = self.load_annotated_rows(eval_run_spec, eval_model)
        else:
            gold_answers = self.load_llm_gold_answers(eval_run_spec, eval_model)
            llm_inference_rows = self.load_llm_inference_rows(eval_run_spec, gold_answers)

            if num_instances != -1:
                llm_inference_rows = random.sample(llm_inference_rows, num_instances)


        # load instances
        instances = [self.load_instance(question_id, eval_db) for question_id, _, _ in llm_inference_rows]

        # convert instances to requests
        requests: List[Request] = [self.build_sudo_request(instance) for instance in instances]

        # load request_results
        model_repo_ids = [row[1] for row in llm_inference_rows]
        completions = [row[2] for row in llm_inference_rows]
        request_results: List[RequestResult] = [
            self.build_sudo_request_result(request, model_repo_id, completion) for request, model_repo_id, completion in
            zip(requests, model_repo_ids, completions)
        ]

        return request_results

    @staticmethod
    def update_eval_result(run_spec: EvalRunSpec, eval_db: EvalDataBase, request_metrics: List[RequestMetric]):
        print('update eval_result table...')
        for request_metric in tqdm(request_metrics):
            request_metric: RequestMetric
            if request_metric.success:
                question_id = request_metric.request_result.request.instance.id
                model_name = request_metric.request_result.model_repo_id
                eval_result = request_metric.evaluation
                eval_result = json.loads(eval_result)
                missing_steps = eval_result['missing_steps']
                redundant_steps = eval_result['redundant_steps']
                duplicate_steps = eval_result['duplicate_steps']
                explain = eval_result['explain']

                data = [question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain]
                try:
                    eval_db.insert_into_eval_result(*data)
                except sqlite3.IntegrityError:
                    eval_db.update_eval_result(*data)

    @staticmethod
    def compute_metrics(run_spec, request_results: List[RequestResult], interfere=False) -> List[RequestMetric]:
        print('compute metrics...')
        metric: Metric = MetricFactory.get_metric(run_spec.metric_spec)
        request_metrics: List[RequestMetric] = metric.compute(request_results, interfere)

        # 检查Ray是否已经初始化,vllm中会初始化，导致第二次使用卡顿
        if ray.is_initialized():
            # 如果已经初始化，先关闭现有的Ray实例
            ray.shutdown()
        del metric  # 释放显卡空间
        return request_metrics

    def run_metric(self, eval_run_spec: EvalRunSpec, args):
        # these args are used to filter instances
        follow = args.follow
        num_instances = args.num_instances
        annotated = args.annotated
        interfere = args.interfere

        # load db
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)


        # load request_results
        request_results: List[RequestResult] = self.load_request_results(
            eval_run_spec, eval_db, follow=follow, num_instances=num_instances,
            annotated=annotated, interfere=interfere)

        # compute metrics
        self.compute_metrics(eval_run_spec, request_results, interfere)
