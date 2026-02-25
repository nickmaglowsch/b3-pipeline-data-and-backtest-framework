import pandas as pd
import requests

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json&dataInicial=01/01/2000&dataFinal=31/12/2026"
response = requests.get(url)
df = pd.DataFrame(response.json())
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
df['valor'] = pd.to_numeric(df['valor']) / 100.0
df.set_index('data', inplace=True)
df.index = df.index + pd.offsets.MonthEnd(0) # shift to month end

print(df.tail(12))
