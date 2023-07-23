with open('demo.def') as f:
    demo = f.read()

for L in range(4, 16 + 1, 2):
    with open('L{}.def'.format(L), 'w') as f:
        f.write(demo.format(L))

with open('run.sh', 'w') as f:
    for L in range(4, 16 + 1, 2):
        f.write('./HPhi -s L{}.def > L{}.result\n'.format(L, L))  
