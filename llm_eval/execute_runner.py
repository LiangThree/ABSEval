import json
import sqlite3
import random
from typing import List
from llm_eval.data.instance import Instance
from llm_eval.data.request import Request, RequestResult, RequestLearner
from llm_eval.data.request import RequestMetric
from llm_eval.data.instance import Input, Output, Reference, CORRECT_TAG
from llm_eval.data.request import Prompt
from llm_eval.scenarios.scenario import TEST_SPLIT
from llm_eval.runner import MetaRunner, RunSpec, Runner, EvalRunSpec
from data.database.util.database_util import EvalDataBase
from llm_eval.data.instance import Instance
from tqdm import tqdm


class ExecuteRunner(MetaRunner):
    def load_question_ids(run_spec: RunSpec):
        print('load question_ids by target_view and model_repo_id in target llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(run_spec.db_path)

        from colorama import Fore
        print(Fore.YELLOW + 'load question_ids by target_view and model_repo_id in target llm_inference table...' + Fore.RESET)



class MetricRunner(MetaRunner):
    def load_question_ids(run_spec: RunSpec):
        print('load question_ids by target_view and model_repo_id in target llm_inference table...')
        eval_db: EvalDataBase = EvalDataBase(run_spec.db_path)

        from colorama import Fore
        print(Fore.BLUE + f"{run_spec}." + Fore.RESET)

        target_view = run_spec.scenario_conf['target_view']
        table_name = run_spec.scenario_conf['table_name']
        model_repo_id = run_spec.model_conf['model_repo_id']
        rows: List[tuple] = eval_db.select_question_ids_from_llm_inference(target_view, model_repo_id, table_name)
        question_ids: List[dir] = [row[0] for row in rows]
        print(len(question_ids))
        return question_ids

    @staticmethod
    def load_llm_inference_rows(eval_run_spec: EvalRunSpec, gold_answers: List[tuple]):
        print('load llm_inference with gold answer...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        question_id_str = ", ".join(["'{}'".format(value) for value in [item[0] for item in gold_answers]])
        query = f"SELECT question_id, model_name,inference FROM llm_inference WHERE question_id IN ({question_id_str})"
        llm_inference_rows = eval_db.conn.execute(query).fetchall()
        return llm_inference_rows

    @staticmethod
    def load_llm_gold_answers(eval_run_spec: EvalRunSpec):
        print('load gold answers...')
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)
        query = """
            SELECT question_id, answer from gold_answer 
        """
        gold_answers = eval_db.conn.execute(query).fetchall()
        return gold_answers

    @staticmethod
    def load_instance(question_id: str, eval_db: EvalDataBase) -> Instance:
        query = "SELECT * FROM question WHERE question_id=?"
        row = eval_db.conn.execute(query, (question_id,)).fetchone()

        query = "SELECT * FROM gold_answer WHERE question_id=?"
        row_answer = eval_db.conn.execute(query, (question_id,)).fetchone()

        references = [Reference(output=Output(row_answer), tag=CORRECT_TAG)]
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

    def filter_annotated_questions(self, question_ids: List[str], model_repo_id: str, eval_db: EvalDataBase) -> List[
        str]:
        """只保留在human_label表已经被标注的问题和回答"""
        annotated_question_ids = []
        cursor = eval_db.conn.cursor()
        for question_id in question_ids:
            rows = cursor.execute('select * from human_eval where question_id=? and model_name=?',
                                  (question_id, model_repo_id)).fetchall()
            if len(rows) > 0:
                annotated_question_ids.append(question_id)
        return annotated_question_ids

    def load_request_results(
            self,
            eval_run_spec: RunSpec,
            eval_db: EvalDataBase,
            test=False,
            num_instances=-1,
            only_annotated_instance=False
    ) -> List[RequestResult]:

        gold_answers = self.load_llm_gold_answers(eval_run_spec)
        llm_inference_rows = self.load_llm_inference_rows(eval_run_spec, gold_answers)

        if test:
            llm_inference_rows = random.sample(llm_inference_rows, 20)
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
    def update_gold_answer(run_spec: EvalRunSpec, eval_db: EvalDataBase, request_metrics: List[RequestMetric]):
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

                data = [question_id, model_name, missing_steps, redundant_steps, duplicate_steps]
                try:
                    eval_db.insert_into_eval_result(*data)
                except sqlite3.IntegrityError:
                    eval_db.update_eval_result(*data)

    def run_metric(self, eval_run_spec: EvalRunSpec, args):
        # these args are used to filter instances
        test = args.test
        num_instances = args.num_instances
        only_annotated_instance = args.only_annotated_instance

        # load db
        eval_db: EvalDataBase = EvalDataBase(eval_run_spec.db_path)

        # load request_results
        request_results: List[RequestResult] = self.load_request_results(
            eval_run_spec, eval_db, test=test, num_instances=num_instances,
            only_annotated_instance=only_annotated_instance)

        # compute metrics
        request_metrics: List[RequestMetric] = self.compute_metrics(eval_run_spec, request_results)

        # update eval_result table
        self.update_gold_answer(eval_run_spec, eval_db, request_metrics)