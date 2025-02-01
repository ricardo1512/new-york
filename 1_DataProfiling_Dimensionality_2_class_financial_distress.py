from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart, get_variable_types

filename = "datasets/0_2_class_financial_distress.csv"
file_tag = "1_Dimenstionality_2_class_financial_distress"
subject = " [financial distress]"
data: DataFrame = read_csv(filename, na_values="")

# print(data.shape)


figure(figsize=(4, 2))

"""
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables" + subject
)
savefig(f"images/ProfilingDimensionality/{file_tag}_records_variables.png", bbox_inches='tight')
show()
"""


mv: dict[str, int] = {}
for var in data.columns:
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
show()

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
show()
"""
""" 
{'numeric': ['Company', 'Time', 'Financial Distress', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 
'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 
'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 
'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61', 
'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 
'x79', 'x80', 'x81', 'x82', 'x83'], 

'binary': ['CLASS'], 

'date': [], 'symbolic': []}

"""