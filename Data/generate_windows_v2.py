import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import h5py
import gc
import psutil
import random
import pyarrow.parquet as pq
import pyarrow as pa
import tracemalloc
import time

WINDOW_SIZE_IN_SECONDS = 5.4  # seconds
DATA_PATH = 'Data/iot/'  # Leave the last slash

def process_vehicle(vehicle, group, window_size_in_ms):
    start_time = time.time()
    try:
        # Extrair valores de cada tipo de evento
        def get_values(name):
            values = group.loc[group['name'] == name, 'vecvalue'].values
            return list(map(float, values[0].split())) if len(values) > 0 else []

        rx_start = get_values('rxStart')
        rx_end = get_values('rxEnd')
        idle_start = get_values('idleStart')
        idle_end = get_values('idleEnd')
        tx_start = get_values('txStart')
        tx_end = get_values('txEnd')
        switching_start = get_values('switchingStart')
        switching_end = get_values('switchingEnd')

        # Combinar dados de início e fim
        start = [(mode, time * 1000) for mode, time in zip(
            ['rx'] * len(rx_start) + ['idle'] * len(idle_start) + ['tx'] * len(tx_start),
            rx_start + idle_start + tx_start
        )]
        index_order = np.argsort([time for mode, time in start])
        start_sorted = [start[i] for i in index_order]

        end = [(mode, time * 1000) for mode, time in zip(
            ['rx'] * len(rx_end) + ['idle'] * len(idle_end) + ['tx'] * len(tx_end),
            rx_end + idle_end + tx_end
        )]

        if len(index_order) > len(end):
            raise ValueError(f"Index order length {len(index_order)} is greater than end length {len(end)} for vehicle {vehicle}")

        end_sorted = [end[i] for i in index_order]

        start_array = np.array(start_sorted)
        end_array = np.array(end_sorted)

        durations = np.column_stack((start_array[:, 0], end_array[:, 1].astype(float) - start_array[:, 1].astype(float)))

        # Ajustar janelas conforme a duração
        i = 0
        while i < len(durations):
            if float(durations[i][1]) > window_size_in_ms:
                num_windows = int(np.ceil(float(durations[i][1]) / window_size_in_ms))
                window_duration = float(durations[i][1]) / num_windows
                windows = np.array([[durations[i][0], window_duration]] * num_windows, dtype=object)
                durations = np.insert(durations, i + 1, windows, axis=0)
                durations = np.delete(durations, i, axis=0)
                i += num_windows - 1
            i += 1

        # Algoritmo de windowing
        current_duration = 0
        duration_index = 0
        window_index = 0
        windows_for_vehicle = []

        while duration_index < len(durations) - 1:
            while current_duration < window_size_in_ms:
                if duration_index >= len(durations):
                    break

                # Adicionar nova janela se necessário
                if len(windows_for_vehicle) <= window_index:
                    windows_for_vehicle.append([])

                if float(durations[duration_index][1]) + current_duration <= window_size_in_ms:
                    windows_for_vehicle[window_index].append(durations[duration_index])
                    current_duration += float(durations[duration_index][1])
                else:
                    windows_for_vehicle.append([])
                    windows_for_vehicle[window_index].append(
                        np.array([durations[duration_index][0], window_size_in_ms - current_duration])
                    )
                    windows_for_vehicle[window_index + 1].append(
                        np.array([durations[duration_index][0], float(durations[duration_index][1]) - (window_size_in_ms - current_duration)])
                    )
                    current_duration = window_size_in_ms  # Ajusta a duração para sair do loop interno

                duration_index += 1

            # Ajustar duração atual para a próxima janela
            try:
                current_duration = float(windows_for_vehicle[window_index + 1][0][1])
            except IndexError:
                current_duration = 0

            # Verificar a soma das durações na janela atual
            sum_duration = sum([float(duration[1]) for duration in windows_for_vehicle[window_index]])
            window_index += 1

        # Remover a última janela se não tiver a duração correta
        if windows_for_vehicle and sum([float(duration[1]) for duration in windows_for_vehicle[-1]]) != window_size_in_ms:
            windows_for_vehicle.pop(-1)

        # Limpeza de memória
        del start_sorted, end_sorted, start_array, end_array, durations
        gc.collect()

        return vehicle, windows_for_vehicle
    except Exception as e:
        raise e
    finally:
        print(f"Processed vehicle {vehicle} in {time.time() - start_time:.2f} seconds")


def save_to_hdf5(filename, results, lock):
    with lock:
        with h5py.File(filename, 'a') as file:
            for vehicle, windows_for_vehicle in results:
                for j, window in enumerate(windows_for_vehicle):
                    window = np.array(window, dtype='S32')
                    group_path = f'vehicle_{vehicle}'
                    window_path = f'window_{j}'
                    if group_path not in file:
                        file.create_group(group_path)
                    file.create_dataset(f'{group_path}/{window_path}', data=window)

def process_chunk(chunk, window_size_in_ms, filename, lock):
    results = []
    processed = 0
    total = len(chunk)
    for vehicle, group in chunk:
        try:
            vehicle, windows = process_vehicle(vehicle, group, window_size_in_ms)
            results.append((vehicle, windows))
            processed += 1
            if processed % 100 == 0 or processed == total:
                print(f"Generating Chunk Windows - Processed {processed}/{total} vehicles", end='\r', flush=True)
        except Exception as e:
            print(f"Error processing vehicle {vehicle}: {e}")

    save_to_hdf5(filename, results, lock)
    # Limpeza de memória
    del results, chunk
    gc.collect()

def generate_windows_from_parquet(parquet_path, window_size_in_seconds=WINDOW_SIZE_IN_SECONDS, num_processes=1):
    window_size_in_ms = window_size_in_seconds * 1000
    manager = mp.Manager()
    lock = manager.Lock()

    filename = f'{DATA_PATH}windows_{int(window_size_in_seconds * 1000)}ms.hdf5'
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    # Inicializar o pool de processos uma vez
    pool = mp.Pool(processes=num_processes)

    buffered_data = pd.DataFrame()
    batch_size = 1000  # Ajuste baseado na sua capacidade de memória

    parquet_file = pq.ParquetFile(parquet_path)

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()

            # Concatenar com os dados em buffer
            buffered_data = pd.concat([buffered_data, batch_df], ignore_index=True)

            # Identificar veículos completos (com 8 linhas)
            vehicle_counts = buffered_data['vehicle'].value_counts()
            complete_vehicles = vehicle_counts[vehicle_counts >= 8].index.tolist()

            # Agrupar veículos completos
            complete_groups = list(buffered_data[buffered_data['vehicle'].isin(complete_vehicles)].groupby('vehicle'))

            # Atualizar buffer removendo veículos completos
            buffered_data = buffered_data[~buffered_data['vehicle'].isin(complete_vehicles)]

            if complete_groups:
                # Dividir os grupos completos em chunks para processamento paralelo
                chunk_size = max(1, len(complete_groups) // (num_processes * 2))
                chunks = [complete_groups[i:i + chunk_size] for i in range(0, len(complete_groups), chunk_size)]

                # Enviar cada chunk para processamento paralelo
                for chunk in chunks:
                    pool.apply_async(process_chunk, args=(chunk, window_size_in_ms, filename, lock))

            break  # Remover esta linha para processar todos os batches

        # Processar quaisquer veículos restantes no buffer
        if not buffered_data.empty:
            remaining_vehicles = buffered_data['vehicle'].unique()
            remaining_groups = list(buffered_data.groupby('vehicle'))

            chunk_size = max(1, len(remaining_groups) // (num_processes * 2))
            chunks = [remaining_groups[i:i + chunk_size] for i in range(0, len(remaining_groups), chunk_size)]

            for chunk in chunks:
                pool.apply_async(process_chunk, args=(chunk, window_size_in_ms, filename, lock))

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        pool.close()
        pool.join()

    print("\nGenerated windows")

if __name__ == "__main__":
    # Caminho para o arquivo Parquet
    parquet_path = f"{DATA_PATH}combined.parquet"

    # Definir tamanhos de janela
    window_size_in_seconds = WINDOW_SIZE_IN_SECONDS

    # Número de processos (CPU cores)
    num_processes = mp.cpu_count()

    # Inicializar o arquivo HDF5
    hdf5_filename = f'{DATA_PATH}windows_{int(window_size_in_seconds * 1000)}ms.hdf5'
    try:
        os.remove(hdf5_filename)
    except FileNotFoundError:
        pass

    # Iniciar o processamento
    print("Starting window generation from Parquet file...")
    generate_windows_from_parquet(parquet_path, window_size_in_seconds, num_processes)
    print("Window generation completed.")
