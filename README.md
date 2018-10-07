# NL-FuN
* Recreate baselines produced by O. Vinyals et al (2017) in StarCraft II: A New Challenge for Reinforcement Learning
* Modify FeUdal Networks (FUN) by A. S. Vezhnevets et al (2017) to suit the PySC2 observations.
* Generalize FUN to additional layers.

## Use
I've added .bat files with examples of how to run the train.py file (need to change its name). The .bat files produce a shell command for each worker specified. Add *--linux* to produce .sh files instead. Use --python_v [python_cmd] to specify what command to run python with. For example: --python_v python3 if you have both python 2.x and python 3.x installed.

## References:
### Papers:    
A3C: https://arxiv.org/pdf/1602.01783.pdf    
PySC2 + Baselines: https://arxiv.org/pdf/1708.04782.pdf    
FeUdal Networks: https://arxiv.org/pdf/1703.01161.pdf    

### Repositories:
Working on this project would not be possible without being able to use the following projects as references:    
#### PySC2:
https://github.com/deepmind/pysc2    

#### A3C:     
https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient   
https://github.com/xhujoy/pysc2-agents    
https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb    
https://github.com/chris-chris/pysc2-examples/tree/master/a2c    

#### A3C + FullyConv:    
https://github.com/H-Park/starcraft2ai/tree/master/A3C    
https://github.com/pekaalto/sc2aibot    

#### A3C + Distributed Tensorflow        
https://github.com/openai/universe-starter-agent/blob/master/a3c.py    

#### Feudal Networks:    
https://github.com/dmakian/feudal_networks    

## Packages:
PySC2 == 1.2    
tensorflow-gpu == 1.9
Python 3.x
