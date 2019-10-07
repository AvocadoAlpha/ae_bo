Architectures for Autoencoder are in the objective function of corresponding file in src MNIST 784 input/output nodes
You can start any python file in src which will start BO/girdSearch and quit it at anytime and continue it whenever
you want all progress/evaluations are saved in trials object in trials folder
default is gridSearch right now
include src to pythonpath / mark as source in pycharm
install packages with pip install -r requirememnts.txt
tensorflow version 1.14.0 !

You can copy the files from trials-yunus into trials and evaluate them or run your own experiments
which create trials objects in the trial folder which you then can evaluate)
Default is gridSearch right now.


Yuan you can check objective function (which builts model for evaluation) in grid_batch.py and grid_nodes.py which produced the two charts i sent you. The data gets preprocessed in utils generate_data_small()

read_trials folder has methods to evaluate trials
To start python processes via terminal:
"""
python3 prediction_plot.py 128 144 #trains model with batch_size = 128 and units1 (nodes) = 144 
#prints prediction-truth and digits plot in plots/prediction-truth

python3 grid_loss_plot.py grid_batch batch_size # first argument name of trials and python file (must be equal) second is x #axes of plot
python3 grid_loss_plot.py grid_nodes units1 #second example
#plots units or batch on x axes and loss on y axes and 20 best performers

python3 time_loss_plot.py fixed_5 #name of pyhon file and identical trials object plots time and loss

"""



space = {
    'units1': hp.choice('units1', [80]),
    'batch_size': hp.quniform('batch_size', 0, 1000, 25)  #implementation of hq.uniform is weird see github.com/hyperopt/hyperopt/issues/321
    }
Space for BO, gridSearch Space quniform is discrete uniform distribution (always starts at 0)
the main function defines which kind of search is done (gridSearch or BO (random also possible))
Once the programm starts every new trial/evaluation is saved in a trials object in the trials folder
program can be stopped at anytime and continued all progress will be saved in trials object

if __name__ == "__main__":
    while True:
    	#utils.run_trials(script_name, space, objective) #save at every iteration BO
    	#utils.run_trials_grid(script_name, space, objective) #dont save at any iteration, save after grid search is finished (not recommended)
        utils.run_trials_grid_2(script_name, space, objective)  #gird Search save after every iteration standart
        
