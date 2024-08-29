import pandas as pd
import numpy as np
from collections import Counter
from copy import copy

net_s = { 0: 39,
          1: 81,
          2: 83,
          3: 6,
          4: 47
         }

sched_s = net_s.keys()

s_count = 0

net_t = { 0: 48,
          1: 92,
          2: 2,
          3: 13,
          4: 24
          }

sched_t = net_t.keys();

t_count = 0

slots = { 0: None,
          1: None,
          2: None,
          3: None,
          4: None
        }

while (list(slots.values()).count(None) != 0):
    for slot in slots:
        print(slot)
        if slots[slot] == None:
            if net_s[slot] > net_t[slot]:
                slots[slot] = 's'
                s_count += 1
            elif net_s[slot] < net_t[slot]:
                slots[slot] = 't'
                t_count += 1
                
            
        
print(slots)
