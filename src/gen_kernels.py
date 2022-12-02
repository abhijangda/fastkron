import os

def pow_range(start, end):
    l = []
    while start <= end:
        l += [start]
        start = start * 2
    return l

MinColsA = 16
MaxColsA = 4096

MinKronRows = 2
MaxKronRows = 64

MaxKronRowsTile = 32

AllColsA    = pow_range(MinColsA, MaxColsA)
AllKronRows = pow_range(MinKronRows, MaxKronRows)

with open("kernel_decl.inc", "w") as f:
    f.write("#define KERNEL_DECL(T, VecT, K_EQUALS_VAR) \\\n")
    contents = ""
    for colsA in AllColsA:
        for kronRows in AllKronRows:
            if colsA < kronRows:
                contents += "    NULL"
            else:
                contents += f"    (void*)kronGemmKernel<T, VecT, N_THREADS,1,TILE_X,{colsA},{kronRows},{kronRows},{MaxKronRowsTile},K_EQUALS_VAR,1>"
        
            contents += ",\\\n"

    #Remove last comma and backslash
    contents = contents[:contents.rfind(",")]
    contents += "\n"
    f.write(contents)