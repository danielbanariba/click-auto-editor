# Abre el archivo en modo de lectura
with open('xd.txt', 'r', encoding='utf-8') as f:
    nombres = f.readlines()

# Ordena la lista en orden alfabético y elimina los nombres duplicados
nombres = sorted(set(nombres))

# Abre el archivo original en modo de escritura
with open('xd.txt', 'w', encoding='utf-8') as f:
    for nombre in nombres:
        f.write(nombre)