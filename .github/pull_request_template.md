> **Escribir para quien revisa.** La descripción debe leerse en un minuto: qué cambia, por
> qué, y qué mirar con atención. Sin tablas, sin secciones decorativas, sin repetir lo que
> el diff ya muestra. Si no entra en pocas líneas, el PR es demasiado grande: dividirlo.
> El detalle técnico va en el issue o en el mensaje del commit.
>
> **Un diagrama cuando explique mejor que el texto.** Para flujos, orden de ejecución o
> relaciones entre componentes, un diagrama reemplaza párrafos: incluirlo en lugar de
> describirlos en prosa, no además de.

## 📝 Descripción
<!-- Resumen de los cambios realizados siguiendo el formato del commit template -->
<!-- Ejemplo: :sparkles: feat(vhs): portar la transición de Arsenal VHS a CUDA -->

**Contexto detallado:**
<!-- Qué cambia y por qué, en pocas líneas. Mencionar incidencias vinculadas (Closes #123) -->

---

## 🧪 Verificación Técnica
- [ ] **Tests locales**: He ejecutado `env/bin/python -m pytest tests/ -q` y han pasado con éxito.
- [ ] **Renderizador C++/CUDA**: Si se tocó `cpp/`, compila sin warnings nuevos con `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j`.
- [ ] **Render de prueba**: Si se tocó el render (`cpp/` o `2. ffmpeg_render.py`), probé con `vhs_render` directo sobre copias de un disco, sin `--test` (mueve carpetas reales).
- [ ] **Dependencias**: ¿Se modificó `requirements.txt`? (Si es sí, requiere `pip install -r requirements.txt`).
- [ ] **Seguridad de Entorno**: No se suben credenciales (`.env`, tokens de YouTube) ni binarios pesados.

---

## 📸 Anexos (Opcional)
<!-- Adjuntar capturas, cuadros de render o logs de ejecución si aplica -->
