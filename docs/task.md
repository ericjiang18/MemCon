## context: 

在搞一个memory的用rl 的upper bound selection, 去给system提memory的建议

## Task: 
### Read latex: 
/home/ubuntu/workplace/MemCon/latex/colm2024_conference.tex

### understnad the current status: 
我目前却实验，目前在alfworld sciworld还有pddl做,感觉不是很够，选了6个benchmark 都能跑 在github上更新了

### Tasks : 我初步实验差不多搞完了, 还没来得及调参数+跑别的method+跑一两个别的模型
有些baseline还没来得及去比，这些是其他的method应该跑起来挺容易的,主要是跑其他的benchmark


### Tasks: 
understand the project codebase (read all the codes under that): 

/home/ubuntu/workplace/MemCon

## requirement: 
### MultiAWS accounts: since we need a lot of API calls, so lets use the multiple aws acounts, regarding its setup, refer to
#### Refresh credentials (every session)
  source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh

#### Start LiteLLM proxy
  cd /home/ubuntu/workplace/AI-Scientist-v2
  nohup .venv/bin/litellm --config litellm_config.yaml --port 4000
  --num_workers 16 > litellm_proxy.log 2>&1 &

### reagring the experiment entry point, using a run script, just like /home/ubuntu/workplace/AI-Scientist-v2/run.sh, it can invoke multi-aws accounts, save logs , and results, 

## Resrouces: 

Other baselines: 

GMemory: /home/ubuntu/workplace/GMemory/mas/memory/mas_memory
latentmem: /home/ubuntu/workplace/LatentMem/latentmem/mas_core/memory/backbone

Optional: their github bases: 
https://github.com/bingreeky/GMemory/tree/main/mas/memory/mas_memory
https://github.com/KANABOON1/LatentMem/tree/main/latentmem/mas_core/memory/backbone

