import sys
import os
sys.path.append(os.path.abspath(os.path.join('..' ,'..', 'src')))
import pickle
import importlib
import matplotlib.pyplot as plt
from list_best_performers import options, init
from hyperopt import space_eval
import utils
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

matplotlib.use("TkAgg")
openF = "fixed_3_l1_k_s_3"#sys.argv[1]

xaxe = "units1"#sys.argv[2]
module = importlib.import_module(openF)
init(openF)
header = "\ntrain: "+str(len(module.x_train)) + "\nval: " + str(len(module.x_val)) + "\ntest: "+str(len(module.x_test))+"\n"



def my_space_eval(vals):
    mydict = {}
    for x in vals:
        if len(vals[x]) == 1:
            mydict[x] = vals[x][0]
        if len(vals[x]) > 1:
            print("ERROR1")

    return space_eval(module.space, mydict)


def normalize(a, amin):
    amax = max(a)
    for i, val in enumerate(a):
        a[i] = ((val-amin) / (amax-amin)) * 100
    return a, amax


trials = pickle.load(open('../../trials/'+ openF, "rb"))
res = trials.trials

y = [x['result']['loss'] for x in res]
first_timestamp = res[0]['book_time'].timestamp()
x, amax = normalize([round(x['refresh_time'].timestamp(), 1) for x in res], first_timestamp)


table, n = options[openF]()
#sorted_table = sorted(table, key=lambda k: k['batch_size'])


x = [x["units1"] for x in table] #populate graph!
z = [x["units2"] for x in table]
y = [x['loss'] for x in table]


from datetime import datetime
total_seconds = (datetime.fromtimestamp(amax) - res[0]['book_time']).total_seconds()
total_time = "Total Time in hours :" + str((total_seconds/60)/60) + '\n'
nOE = total_time + "Number of Evaluations :" + str(len(trials.trials)) + '\n'
best = str(my_space_eval(trials.best_trial['misc']['vals']))+" Result :"+str(trials.best_trial['result'])
space = module.space
text = '\n' + nOE + "Best: " + best + '\n' + module.space_str + '\n'

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title(openF+header+text, loc='left')





ax.invert_yaxis()
ax.set_xlabel("units1")
ax.set_ylabel("units2")
ax.set_zlabel("loss")
#ax.set_zlim([0.017, 0.021])

surf = ax.plot_trisurf(x, z, y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.2, aspect=2)


fig.savefig('../../plots/grid-loss-plot3d/'+str(openF)+'.png')
plt.show()
print("Figure saved in figures/")