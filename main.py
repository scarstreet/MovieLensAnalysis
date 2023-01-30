# Use this file as main place for the data analysis

import numpy as np
import pandas as pd

rand = np.random.RandomState(1)
df = pd.DataFrame({'A': ['foo', 'bar'] * 3,
                   'B': rand.randn(6),
                   'C': rand.randint(0, 20, 6)})
gb = df.groupby(['A'])
print(type(gb.get_group('bar')))

