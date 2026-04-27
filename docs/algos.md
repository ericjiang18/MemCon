You're right. The g-memory entry in module_map already points to your
  colleague's GMemory at:

  /home/ubuntu/workplace/MemCon/mas/memory/mas_memory/GMemory.py

  This is the one used for ALL combinations — every framework, every benchmark,
  every model. When you run:

  ./scripts/run.sh run --memory g-memory --framework lobster,langgraph,agent_framework --benchmark alfworld,pddl,sciworld --model haiku-3 --max-parallel 72 -b