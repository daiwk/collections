#encoding=utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model_score = pd.read_csv('xx.csv')
model_score['reason'] = model_score['reason'].map(lambda x: x.replace('[','').replace('"','').replace(']',''
                                                        ).replace('metric1,',''))
model_score = model_score.set_index(['reason'])


model_score.dropna(inplace=True,axis=1)
for ms in ['combined_score','score1','score2']:
    plt.figure()
    for r in ['recall1', 'recall2']:
        plt.plot(np.arange(0,105,5)[:-4],
                 [float(x) for x in model_score.loc[r,ms].strip('[|]').split(',')][:-4],
                 label = r
                )
    plt.title(ms)
#     plt.axvline(x = 85)
    plt.legend()
plt.show();
