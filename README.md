## Resource-Allocation-Reinforcement-Learning

This repository is part of my Master Thesis titled: "__*Design and implementation of an intelligent agent,
capable of sharing resources in multicore systems, using Deep Reinforcement Learning*__".

Implementation of a Deep Reinforcement Learning agent that is capable to share the last-level-cache of a multi-core system, between a Latency Critical Service and a number of Best Effort applications. The agent by utilising the DQN family of algorithms, achieved to keep the SLAs violation of the critical service below 3% and in the same time succeeded even a 4x speed up for the Best Effort apps, by allocating cache ways to them when possible.

Please use this identifier to cite or link to this item: http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/17662

### Dependencies

In a new conda environment execute: 
    
    pip install -r requirements.txt
    
