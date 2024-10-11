import pyarrow.parquet as pq
import pyarrow as pa
import os

# Diretório dos arquivos Parquet
input_dir = 'Data/iot/'
output_file = 'Data/iot/combined.parquet'

# Listar todos os arquivos .parquet no diretório
files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

# Função para extrair o número i do nome do arquivo
def extract_i_from_filename(filename):
    return int(filename.split('.')[0][-1])

# Criar um ParquetWriter apenas uma vez no início
writer = None

# Processar cada arquivo em batches e salvar incrementalmente
for idx, file in enumerate(files):
    i = extract_i_from_filename(file)
    file_path = os.path.join(input_dir, file)
    
    # Ler o arquivo Parquet em batches usando PyArrow
    parquet_file = pq.ParquetFile(file_path)
    
    # Processar por batches
    for batch in parquet_file.iter_batches():
        # Converter batch em tabela PyArrow
        table = pa.Table.from_batches([batch])

        # Modificar a coluna 'vehicle' somando 200 * i
        vehicle_col = table['vehicle'].to_pandas() + (200 * i)

        # Remover a coluna antiga de 'vehicle'
        table = table.drop(['vehicle'])

        table = table.append_column('vehicle', pa.array(vehicle_col))
        
        # Inicializar o writer na primeira iteração
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression='SNAPPY', use_dictionary=True)
        
        # Escrever o batch processado no arquivo Parquet final
        writer.write_table(table)

# Fechar o writer após terminar de processar todos os arquivos
if writer:
    writer.close()

print(f"Arquivos combinados e salvos em {output_file}")
