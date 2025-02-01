from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart, get_variable_types

filename = "datasets/0_1_class_ny_arrests.csv"
file_tag = "1_Dimenstionality_1_class_ny_arrests"
subject = " [ny arrests]"
file_output = "datasets/0_1b_class_ny_arrests.csv"
data: DataFrame = read_csv(filename, na_values="", index_col="ARREST_KEY")

# Criação da coluna 'CLASS' com base na lógica fornecida.
data['CLASS'] = data['JURISDICTION_CODE'].apply(lambda x: 'NY' if x < 3 else 'nonNY')

# Remoção da coluna 'JURISDICTION_CODE'.
data.drop('JURISDICTION_CODE', axis=1, inplace=True)

# print(data.shape)

"""
figure(figsize=(4, 2))


values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables" + subject
)
savefig(f"images/ProfilingDimensionality/{file_tag}_records_variables.png", bbox_inches='tight')
show()
"""

"""
variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type" + subject
)
savefig(f"images/ProfilingDimensionality/{file_tag}_variable_types.png", bbox_inches='tight')
# show()
"""
""" 
{'numeric': ['ARREST_KEY', 'PD_CD', 'KY_CD', 'ARREST_PRECINCT', 'JURISDICTION_CODE', 'X_COORD_CD', 'Y_COORD_CD', 'Latitude', 
'Longitude'], 'binary': ['LAW_CAT_CD', 'PERP_SEX'],LAW_CODE', 'ARREST_BORO', 'AGE_GROUP', 'PERP_RACE

'date': ['ARREST_DATE'], 

'symbolic': ['PD_DESC', 'OFNS_DESC', 'LAW_CODE', 'ARREST_BORO', 'AGE_GROUP', 'PERP_RACE']}

"""


mv: dict[str, int] = {}
for var in data.columns:
    # Somar os valores NaN e os "UNKNOWN" na coluna 'var'
    nr = data[var].isna().sum() + (data[var] == "UNKNOWN").sum()
    if nr > 0:
        mv[var] = nr

figure()
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable" + subject,
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/ProfilingDimensionality/{file_tag}_mv.png", bbox_inches='tight')
# show()

# Salvar as alterações em um novo arquivo CSV
data.to_csv(file_output)