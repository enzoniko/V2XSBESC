import os
import pandas as pd

SIMULATION_END = '7200.0'

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = 0 # 0: RX, 1: TX, 2: IDLE, 3: Switching
        self.idleStart = []
        self.idleEnd = []
        self.rxStart = []
        self.rxEnd = []
        self.txStart = []
        self.txEnd = []
        self.switchingStart = []
        self.switchingEnd = []

def get_paths():
    base_path = 'Data/iot/'
    paths = []
    
    # Get the name of the files in the directory
    for file in os.listdir(base_path):
        if file.endswith('.txt'):
            paths.append(base_path + file)
    
    return paths
    
def generate_csv():
    paths = get_paths()[:1]
    print(paths)
    
    for path in paths:
        nodes = {}
        print(f'Processing: {path}')
        
        with open(path, 'r') as f:
            data = f.readlines()
            count = 0
            for line in data:
                line = line.strip().split('_')[1:]
                #print(line)
                if len(line) < 3:
                    continue
                
                timestamp = line[0]
                node_id = int(line[1].split('.')[1].strip('node[').strip(']'))
                
                if node_id not in nodes:
                    nodes[node_id] = Node(node_id)
                
                command = line[2].split(' ')
                if command[0] == 'SET':
                    if nodes[node_id].state == 0:
                        nodes[node_id].rxEnd.append(timestamp)
                    elif nodes[node_id].state == 1:
                        nodes[node_id].txEnd.append(timestamp)
                    elif nodes[node_id].state == 2:
                        nodes[node_id].idleEnd.append(timestamp)
                    
                    if nodes[node_id].state != 3:
                        nodes[node_id].state = 3 # Switching
                        nodes[node_id].switchingStart.append(timestamp)
                elif command[0] == 'completing' and timestamp != 0:
                    state = int(command[-2])
                    nodes[node_id].state = state
                    nodes[node_id].switchingEnd.append(timestamp)
                    if state == 0:
                        nodes[node_id].rxStart.append(timestamp)
                    elif state == 1:
                        nodes[node_id].txStart.append(timestamp)
                    elif state == 2:
                        nodes[node_id].idleStart.append(timestamp)
                elif command[0] == 'completing' and timestamp == 0:
                   nodes[node_id].state = 0
                   nodes[node_id].rxStart.append(timestamp)        

                count += 1
                # Print the percentage of the file processed
                print(f'Processing: {count / len(data) * 100:.2f}%', end='\r')
        
        # Create a dataframe
        df = pd.DataFrame(columns=['name', 'vecvalue', 'vehicle'])
        for key, items in nodes.items():
            
            # Calculate length difference between start and end
            rx_diff = len(items.rxStart) - len(items.rxEnd)
            tx_diff = len(items.txStart) - len(items.txEnd)
            idle_diff = len(items.idleStart) - len(items.idleEnd)
            switching_diff = len(items.switchingStart) - len(items.switchingEnd)

            # If any of the differences is 1 while the other ones are 0, then add an extra element to the end
            if rx_diff == 1 and tx_diff == 0 and idle_diff == 0:
                items.rxEnd.append(SIMULATION_END)
            elif rx_diff == 0 and tx_diff == 1 and idle_diff == 0:
                items.txEnd.append(SIMULATION_END)
            elif rx_diff == 0 and tx_diff == 0 and idle_diff == 1:
                items.idleEnd.append(SIMULATION_END)

            # If the switching difference is -1, remove the first element from the start
            if switching_diff == -1:
                items.switchingEnd.pop(0)
            
            df = pd.concat([df, pd.DataFrame({'name': ['rxStart'], 'vecvalue': [' '.join(items.rxStart)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['rxEnd'], 'vecvalue': [' '.join(items.rxEnd)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['txStart'], 'vecvalue': [' '.join(items.txStart)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['txEnd'], 'vecvalue': [' '.join(items.txEnd)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['idleStart'], 'vecvalue': [' '.join(items.idleStart)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['idleEnd'], 'vecvalue': [' '.join(items.idleEnd)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['switchingStart'], 'vecvalue': [' '.join(items.switchingStart)], 'vehicle': [key]})], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'name': ['switchingEnd'], 'vecvalue': [' '.join(items.switchingEnd)], 'vehicle': [key]})], ignore_index=True)
        
        df.to_csv(path.replace('.txt', '.csv'), index=False)
        # delete text file
        os.remove(path)

def csv_to_parquet():
    # get all .csv files in the directory
    for file in os.listdir('Data/iot/'):
        if file.endswith('.csv'):
            print(f'Compressing: {file}')
            df = pd.read_csv('Data/iot/' + file)
            df.to_parquet('Data/iot/' + file.replace('.csv', '.parquet'), index=False)
            os.remove('Data/iot/' + file)

if __name__ == '__main__':
    generate_csv()
    csv_to_parquet()