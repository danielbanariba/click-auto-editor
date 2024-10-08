# Abre el archivo en modo de lectura
with open('bandas-subidas-al-canal.txt', 'r', encoding='utf-8') as f:
    nombres = f.readlines()

# Ordena la lista en orden alfabético y elimina los nombres duplicados
nombres = sorted(set(nombres))

# Abre el archivo original en modo de escritura
with open('bandas-subidas-al-canal.txt', 'w', encoding='utf-8') as f:
    for nombre in nombres:
        f.write(nombre)