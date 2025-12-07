
# Matrix is 16 rows, 128 columns
# Write columns (well-strided)
# Read consecurive segments of rows (poorly-strided)
# Make matrix have 15 stride by observing that the diagonal element in the transpose can just be kept
# And so only 15 elements need to be written in each 16 element block

def flatten_idx(row, col, num_cols):
    return row * num_cols + col

def new_idx(row, col, num_cols):
    as1d = flatten_idx(row, col, num_cols)
    stride15 = as1d - as1d // 16
    if (col % 16) > row: 
        stride15 -= 1
    return stride15


for i in range(16):
    for j in range(128):
        if j % 16 == i:
            print("r", end =" ")
        else:
            print(new_idx(i,j,128), end =" ")
    print()