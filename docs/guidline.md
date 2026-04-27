To Run Experiments

  # 1. Setup proxy
  ./scripts/run.sh setup

  # 2. HP sweep (quick, ~20 trials on 30 ALFWorld tasks)
  .venv/bin/python scripts/sweep_hyperparams.py

  # 3. Full experiments — Sonnet 4 (mapped as gpt-4.1-mini)
  ./scripts/run.sh run --all --model gpt-4.1-mini

  # 4. Full experiments — Haiku 3.5
  ./scripts/run.sh run --all --model haiku-3.5

  # 5. Generate LaTeX tables
  .venv/bin/python scripts/generate_table.py results/exp_gpt-4.1-mini*/




  Step 1: Setup proxy (one-time per session)                                                       
   
  ./scripts/run.sh setup                                                                           
                                                                                                   
  Step 2: Run baselines first (just the two new ones on all benchmarks)                            
                                                                                                   
  # G-Memory-Orig baseline across all frameworks × benchmarks                                      
  ./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \              
    --framework lobster,langgraph,agent_framework \                                                
    --memory g-memory-orig,latentmem                                                               
                                                                                                   
  # This launches 3 fw × 2 mem × 6 bench = 36 jobs                                                 
                                                                                                   
  Or if you want to start smaller — just Lobster on interactive benchmarks first:                  
  ./scripts/run.sh run --benchmark alfworld,pddl,sciworld \                                        
    --framework lobster \                                                                          
    --memory g-memory-orig,latentmem                                                               
                                                                                                   
  Step 3: HP sweep (after baselines finish)                                                        
                                                                                                   
  .venv/bin/python scripts/sweep_hyperparams.py                                                    
  This runs 20 random configs on ALFWorld+Lobster with 30 tasks each. Takes the best and saves to  
  configs/best_memcon_policy.json.                                                                 
                                                                                                   
  Step 4: Re-run MemCon with tuned params                                                          
                                                                                                   
  After the sweep, you'd manually update the best config into the experiment. Right now the sweep  
  saves the best config but doesn't auto-feed it back into run_full_experiment.py.                 
                                                                                                   
  Want me to add that — make run_full_experiment.py auto-load configs/best_memcon_policy.json if it
   exists, so after the sweep your MemCon runs automatically use the tuned params?



pkill -f "litellm.*--port.*4001"; sleep 2
  source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh
  nohup .venv/bin/litellm --config litellm_config.yaml --port 4001 --num_workers 16 >
  litellm_proxy.log 2>&1 &
  sleep 10

pkill -f "litellm.*--port.*4001"; sleep 2
  source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh
  cd /home/ubuntu/workplace/MemCon
  nohup .venv/bin/litellm --config litellm_config.yaml --port 4001 --num_workers 16 &
  sleep 10


./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \              
      --framework lobster,langgraph,agent_framework \                                              
      --memory g-memory-orig,latentmem  


# 1. Refresh creds
  source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh

  # 2. Regenerate config from live profiles only
  cd /home/ubuntu/workplace/MemCon
  python3 scripts/gen_litellm_config.py

  # 3. Restart proxy

cd /home/ubuntu/workplace/MemCon
pkill -9 -f "litellm.*4001"; sleep 3 && nohup .venv/bin/litellm --config litellm_config.yaml --port 4001 --num_workers 32 > litellm_proxy.log 2>&1 & sleep 15

  # 4. Run experiments
  ./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \
      --framework lobster,langgraph,agent_framework \
      --model haiku-3 \
      --memory g-memory,latentmem \
      --max-parallel 36 -b

./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \
      --framework lobster \
      --model sonnet-4 \
      --memory oagent \
      --max-parallel 6 --exp-name lobster_oagent -b

,langgraph,agent_framework


./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \
      --framework lobster \
      --model sonnet-4 \
      --memory voyager \
      --max-parallel 18 --exp-name lobster_oagent -b


## ,langgraph,agent_framework

./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia \
      --framework lobster,langgraph,agent_framework \
      --model gpt-4.1-mini \
      --api_key openai \
      --memory oagent \
      --max-parallel 18 --exp-name lobster_oagent -b



  ./scripts/run.sh run --benchmark                                          
  alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia --framework lobster      
  --memory oagent --model sonnet-4 --max-parallel 6 --exp-name              
  lobster_oagent -b     

./scripts/run.sh run --benchmark alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia --framework
  lobster,langgraph,agent_framework --memory
  empty,g-memory,latentmem,memcon,metagpt,voyager,generative,chatdev,memorybank,oagent,experiencebank
  --model haiku-3 --max-parallel 72 -b


# Kill all running experiments
  pkill -f "run_full_experiment.py"

  # Kill the LiteLLM proxy
  pkill -f "litellm.*4001"

  Or to kill everything at once:

  pkill -f "run_full_experiment.py"; pkill -f "litellm.*4001"


Then launch with --model sonnet-4:                                            


# Run all 11 methods (each gets its own log: run_metagpt.log, run_voyager.log,
  etc.)
./scripts/run_all_baselines.sh

# Run specific ones only
./scripts/run_all_baselines.sh metagpt voyager generative chatdev memorybank oagent experiencebank

# Different model
MODEL=haiku-3 ./scripts/run_all_baselines.sh

# Check progress
tail -1 run_*.log

Each baseline gets run_{name}.log — e.g., run_metagpt.log, run_voyager.log,
run_chatdev.log, etc.


./scripts/run.sh run --benchmark                                              
alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia --framework                  
lobster,langgraph,agent_framework --memory empty,g-memory,memcon,latentmem,met
agpt,voyager,generative,chatdev,memorybank,oagent,experiencebank --model      
sonnet-4 --max-parallel 72 -b        


<!-- LLM_MODEL=sonnet-4 ./scripts/rerun_missing.sh -b --max-parallel 72 -->

LLM_MODEL=sonnet-4 ./scripts/rerun_missing.sh --auto-fix -b --max-parallel 72



./scripts/run_sequential.sh --memory oagent -b
