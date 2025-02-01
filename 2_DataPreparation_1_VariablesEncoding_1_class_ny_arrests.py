import pandas as pd
import numpy as np

# Carregar os dados
filename = "datasets/0_1b_class_ny_arrests.csv"
file_output = "datasets/1_1_1_variables_encoding_class_ny_arrests.csv"
data: pd.DataFrame = pd.read_csv(filename, index_col="ARREST_KEY", na_values=["", "UNKNOWN"])
target = "CLASS"
"""

'binary': ['CLASS', 'PERP_SEX'],

'symbolic': ['LAW_CODE', 'ARREST_BORO', 'AGE_GROUP', 'PERP_RACE']}
"""

# Ensure ARREST_DATE is in datetime format
data['ARREST_DATE'] = pd.to_datetime(data['ARREST_DATE'])

# Create new columns for month and year
data['arrest_month'] = data['ARREST_DATE'].dt.month
data['arrest_quarter'] = data['ARREST_DATE'].dt.quarter
data['arrest_year'] = data['ARREST_DATE'].dt.year
data = data.drop(columns=['ARREST_DATE'])

"""
# Selecionar colunas de interesse e obter valores únicos
columns_of_interest = ['arrest_month', 'arrest_year', "ARREST_DATE", 'LAW_CODE', 'ARREST_BORO', 'AGE_GROUP', 'PERP_RACE']
unique_values = {col: data[col].dropna().unique() for col in columns_of_interest}

# Criar um novo DataFrame para exportar
unique_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unique_values.items()]))

# Exportar para CSV
output_filename = "datasets/unique_values_ny_arrests.csv"
unique_df.to_csv(output_filename, index=False)
"""

"""
'arrest_year'
"""
# Create a mapping for years from 2006 to 2021
year_mapping = {year: idx + 1 for idx, year in enumerate(range(2006, 2022))}

# Apply the mapping to the 'arrest_year' column
data['arrest_year'] = data['arrest_year'].map(year_mapping)

"""
'arrest_month'
"""
from math import pi, sin, cos

# Assuming 'arrest_month' contains months from 1 to 12
data['arrest_month_sin'] = data['arrest_month'].apply(lambda x: round(sin(2 * pi * x / 12), 3))
data['arrest_month_cos'] = data['arrest_month'].apply(lambda x: round(cos(2 * pi * x / 12), 3))
data = data.drop(columns=['arrest_month'])

data['arrest_quarter_sin'] = data['arrest_quarter'].apply(lambda x: round(sin(2 * pi * x / 4), 3))
data['arrest_quarter_cos'] = data['arrest_quarter'].apply(lambda x: round(cos(2 * pi * x / 4), 3))
data = data.drop(columns=['arrest_quarter'])

"""
'PERP_SEX'
"""
data["PERP_SEX"] = data["PERP_SEX"].map({"M": 1, "F": 0})

"""
'LAW_CAT_CD'
"""
data["LAW_CAT_CD"] = data["LAW_CAT_CD"].map({"M": 1, "F": 0})

"""
'CLASS'
"""
data["CLASS"] = data["CLASS"].apply(lambda x: 1 if x == "nonNY" else 0)

"""
'LAW_CODE'
"""
# Obter os valores únicos de 'LAW_CODE', ordenados alfabeticamente
unique_law_code = sorted(data["LAW_CODE"].dropna().unique())

# Criar um dicionário com o índice + 1 para cada valor único
law_code_dict = {code: idx + 1 for idx, code in enumerate(unique_law_code)}

data["LAW_CODE"] = data["LAW_CODE"].map(law_code_dict)

"""
'ARREST_BORO'
"""
arrest_boro_mapping = {
    "B": 1,
    "K": 2,
    "M": 3,
    "Q": 4,
    "S": 5
}

data["ARREST_BORO"] = data["ARREST_BORO"].map(arrest_boro_mapping)

"""
'AGE_GROUP'
"""
age_group_mapping = {
    "<18": 1,
    "18-24": 2,
    "25-44": 3,
    "45-64": 4
}

# Para garantir que AGE_GROUP mapeie corretamente, vou mapear com os valores de 'age_group_mapping'
data["AGE_GROUP"] = data["AGE_GROUP"].map(age_group_mapping)

"""
'PERP_RACE'
"""
perp_race_dict = {
    "BLACK": 1,
    "ASIAN / PACIFIC ISLANDER": 2,
    "WHITE": 3,
    "AMERICAN INDIAN/ALASKAN NATIVE": 4,
    "BLACK HISPANIC": 5,
    "WHITE HISPANIC": 6
}

data["PERP_RACE"] = data["PERP_RACE"].map(perp_race_dict)

# Salvar as alterações em um novo arquivo CSV
data.to_csv(file_output)
