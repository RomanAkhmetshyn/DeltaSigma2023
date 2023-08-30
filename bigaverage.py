import numpy as np



data = np.loadtxt('new 6.txt', skiprows=1, dtype=float)
column_averages = np.mean(data, axis=0)

print(column_averages)
# output_file = 'column_averages.txt'
# np.savetxt(output_file, column_averages, delimiter='\t', newline='\n', fmt='%f')
