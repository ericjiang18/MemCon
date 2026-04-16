import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from dataclasses import dataclass, field
import argparse
import random
from tqdm import tqdm

import mas
from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, QwenChat, GeminiChat, ClaudeChat, get_price
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type


with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None

@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config_content = reader.read()
        # Expand environment variables in config (e.g., $ALFWORLD_DATA)
        config_content = os.path.expandvars(config_content)
        config = yaml.safe_load(config_content)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=WORKING_DIR, namespace='total_task')
    tasks: list[dict] = get_task(task)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )   

def get_llm_model(model_type: str) -> LLMCallable:
    """
    Get the appropriate LLM model based on model type.
    Supports OpenAI, Qwen (Idealab), and Gemini (Google AI).
    """
    qwen_models = ['qwen3-flash', 'qwen3-max', 'qwen-turbo', 'qwen-plus', 'qwen3-vl-flash']

    if model_type.lower() in qwen_models or model_type.lower().startswith('qwen'):
        print(f"Using Qwen model: {model_type}")
        return QwenChat(model_name=model_type, use_idealab=True)
    elif model_type.lower().startswith('gemini'):
        print(f"Using Gemini model: {model_type}")
        return GeminiChat(model_name=model_type)
    elif model_type.lower().startswith('claude'):
        print(f"Using Claude model: {model_type}")
        return ClaudeChat(model_name=model_type)
    else:
        print(f"Using OpenAI model: {model_type}")
        return GPTChat(model_name=model_type)


def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
) -> None:
    
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")) 
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    # Use appropriate LLM based on model type
    llm_model: LLMCallable = get_llm_model(llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

def run_task(task_manager: TaskManager) -> None:

    task_manager.recorder.dataset_begin()
    
    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        task_manager.recorder.task_begin(task_id, task_config)  
        
        task_main, task_description = task_manager.mas.env.set_env(task_config)   
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name, 
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
           
        task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():    
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done = task_manager.mas.schedule(task_config) 
    
        task_manager.recorder.task_end(reward, done)             
    
    task_manager.recorder.dataset_end()



if __name__ == '__main__':
    # settings
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, choices=['alfworld', 'sciworld', 'pddl', 'math', 'aime24', 'aime25', 'gpqa', 'mmlu_pro_eng'])
    parser.add_argument('--mas_type', type=str, choices=['autogen', 'macnet', 'dylan', 'skill-mas', 'skill', 'gcn', 'goal-gcn', 'goalrl'])
    parser.add_argument('--mas_memory', type=str, default='none', help='Specify mas memory module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=50, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')

    args = parser.parse_args()

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning
    
    # dir
    WORKING_DIR = os.path.join('./.db', get_model_type(model_type), task, mas_type)
    # if os.path.exists(WORKING_DIR):
    #     shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # run tasks
    task_configs: TaskManager = build_task(task, mas_type, mas_memory_type, max_trials)
    task_configs.mas_config['successful_topk'] = args.successful_topk
    task_configs.mas_config['failed_topk'] = args.failed_topk
    task_configs.mas_config['insights_topk'] = args.insights_topk
    task_configs.mas_config['threshold'] = args.threshold
    task_configs.mas_config['use_projector'] = args.use_projector
    task_configs.mem_config.update(
        working_dir=WORKING_DIR,
        hop=args.hop
    )

    build_mas(task_configs, reasoning_type, mas_memory_type, model_type)
    run_task(task_configs)

    # postprocess
    completion_tokens, prompt_tokens, _ = get_price()
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')