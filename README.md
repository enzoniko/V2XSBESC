# Requirements
First, use a venv to install the requirements, we are are not savages.
```bash
python -m venv venv
source venv/bin/activate
```
You need to have pandas, numpy and h5py installed for now (psutil is optional, but it is used to show the memory usage of the script).:
```bash
pip install -r requirements.txt
```

# Files
- Data/getVehicleStartEnd.py: Script to get the start and end of each vehicle in the simulation.
- Data/calculate_optimal_window_size_by_switches.py: Script to calculate the optimal window size based on the number of switches.
- Data/clean_sim_data_and_separate_rx_and_idle.py: Code to clean the simulation data and separate the RX and idle data.
- Data/generate_windows.py: Code to generate the windows for the cleaned data.
- Data/summarize_windows.py: Code to summarize the windows.
- Data/generate_x_and_y.py: Code to generate the data for the models.
- preprocessing.py: Script with the functions to preprocess the data.
- models.py: Script with the models.
- training.py: Script with the functions to train the models.
- predicting.py: Script with the functions to predict with the models.
- callbacks.py: Script with the callbacks for the models.
- ploting.py: Script with the functions to plot the results.
- experiments.py: Script with code that runs models several times (grid search) and saves the results.
- exportEmbeddedModels.py: Script to export the models to a format that can be used in the embedded system.
- testSelectedModels.py: Script to test the selected models.
- scratchfile.py: Script with code that is used to test new ideas.


# Running
To run the code you need to have in the Data folder the csv file with the simulation data and the sca file with the start and end of each vehicle. You can run the getVehicleStartEnd.py to get the start and end of each vehicle. Then you can run clean_sim_data_and_separate_rx_and_idle.py to clean the data and separate the RX and idle data. After that you can run the calculate_optimal_window_size_by_switches.py to calculate the optimal window size based on the number of switches. Then you can run generate_windows.py to generate the windows for the cleaned data. Then you can run summarize_windows.py to summarize the windows. Finally, you can run the pipeline to generate the data for the models and train the models. You can also run the experiments.py to run the models several times (grid search) and save the results. You can run the exportEmbeddedModels.py to export the models to a format that can be used in the embedded system (you need to put the data and the information that are generated on the exporting folder into the V2XSBESCEMBEDDED/exported folder). You can run the testSelectedModels.py to test the selected models. The V2XSBESCEMBEDDED folder has the code to run the models in the embedded system with an accompanying README.md file with instructions on how to run the code.