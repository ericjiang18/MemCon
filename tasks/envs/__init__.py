import os
import glob

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .math_env import MathEnv, MathRecorder, MathEnvConfig
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs


MATH_TASKS = {'math', 'aime24', 'aime25', 'gpqa', 'mmlu_pro_eng'}

def _get_envs():
    """Lazy env registry to avoid importing SciWorld when not needed."""
    envs = {
        'alfworld': AlfworldEnv,
        'pddl': PDDLEnv,
    }
    for t in MATH_TASKS:
        envs[t] = MathEnv
    try:
        from .sciworld_env import SciworldEnv
        envs['sciworld'] = SciworldEnv
    except ImportError:
        pass
    return envs

def _get_recorders():
    """Lazy recorder registry to avoid importing SciWorld when not needed."""
    recorders = {
        'alfworld': AlfworldRecorder,
        'pddl': PDDLRecorder,
    }
    for t in MATH_TASKS:
        recorders[t] = MathRecorder
    try:
        from .sciworld_env import SciworldRecorder
        recorders['sciworld'] = SciworldRecorder
    except ImportError:
        pass
    return recorders


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    envs = _get_envs()
    if envs.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return envs.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    recorders = _get_recorders()
    if recorders.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return recorders.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str) -> list[dict]:
    
    if task == 'alfworld':
        alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
        games_dir = os.path.join(alfworld_data, 'json_2.1.1', 'valid_unseen')
        game_files = glob.glob(os.path.join(games_dir, '*', '*', 'game.tw-pddl'))
        
        tasks = []
        for gamefile in sorted(game_files):
            task_dir = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
            env_name = get_env_name_from_gamefile(task_dir)
            
            if env_name is None:
                continue
            
            goal = task_dir
            
            tasks.append({
                'task': goal,
                'env_kwargs': {
                    'config': 'alfworld',
                    'gamefile': gamefile,
                },
                'task_type': prefixes.get(env_name, env_name),
                'env_name': env_name
            })
        
        print(f"Loaded {len(tasks)} ALFWorld tasks from {games_dir}")
        return tasks

    elif task == 'sciworld':
        from scienceworld import ScienceWorldEnv
        
        tmp_env = ScienceWorldEnv("", envStepLimit=100)
        all_task_names = tmp_env.get_task_names()
        print(f"ScienceWorld supported tasks ({len(all_task_names)}): {all_task_names}")
        
        tasks = []
        for task_name in all_task_names:
            tmp_env.load(task_name, variationIdx=0)
            test_variations = tmp_env.get_variations_test()
            max_vars = 5
            variations = test_variations[:max_vars] if len(test_variations) > max_vars else test_variations
            
            for var_idx in variations:
                tasks.append({
                    'task': f'{task_name}_var{var_idx}',
                    'task_name': task_name,
                    'variation_idx': var_idx,
                    'simplification': 'teleportAction,openDoors,openContainers',
                    'env_name': task_name,
                })
        
        tmp_env.close()
        print(f"Loaded {len(tasks)} ScienceWorld tasks ({len(all_task_names)} task types)")
        return tasks
    
    elif task == 'math':
        import json
        data_dir = 'data/math_test' # Default to train for now
        tasks = []
        
        if not os.path.exists(data_dir):
            print(f"Warning: Math data directory {data_dir} not found.")
            return []

        for filename in os.listdir(data_dir):
            if filename.endswith(".jsonl"):
                path = os.path.join(data_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            problem_data = json.loads(line)
                            tasks.append({
                                'task': problem_data['problem'],
                                'env_name': 'math',
                                'env_kwargs': {
                                    'problem': problem_data,
                                    'config': 'math'
                                },
                                'task_type': problem_data.get('type', 'math')
                            })
        # Limit for testing if needed, or return all
        return tasks

    elif task in MATH_TASKS:
        import json
        data_map = {
            'math': 'data/math_test',
            'aime24': 'data/qa_test/aime_2024.jsonl',
            'aime25': 'data/qa_test/aime_2025.jsonl',
            'gpqa': 'data/qa_test/gpqa_diamond.jsonl',
            'mmlu_pro_eng': 'data/qa_test/mmlu_pro_eng.jsonl',
        }
        data_path = data_map.get(task, 'data/math_test')
        tasks = []

        def _load_jsonl(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if not item.get('problem'):
                            continue
                        tasks.append({
                            'task': item['problem'][:120],
                            'env_name': task,
                            'env_kwargs': {'problem': item, 'config': task},
                            'task_type': item.get('source', task),
                        })

        if os.path.isfile(data_path):
            _load_jsonl(data_path)
        elif os.path.isdir(data_path):
            for fn in sorted(os.listdir(data_path)):
                if fn.endswith('.jsonl'):
                    _load_jsonl(os.path.join(data_path, fn))

        print(f"Loaded {len(tasks)} {task} tasks from {data_path}")
        return tasks

    elif task == 'pddl':
        TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
        return get_all_environment_configs(TASK_NAMES, 'data/pddl/test.jsonl')

    raise ValueError(f'Unsupported task type: {task}')