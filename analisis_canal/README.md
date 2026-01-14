# Analisis de canal

Proyecto simple para extraer datos de tu canal con YouTube Data API v3 y generar un reporte basico para tomar decisiones de crecimiento.

## Flujo rapido

1. Ajusta `analisis_canal/config.py`.
2. Ejecuta `python "analisis_canal/0. preparar_entorno.py"`.
3. Ejecuta `python "analisis_canal/1. extraer_datos.py"`.
4. Ejecuta `python "analisis_canal/2. analizar_datos.py"`.

## Salidas

- `analisis_canal/data/raw/`:
  - `canal.json`
  - `video_ids.json`
  - `videos.json`
- `analisis_canal/data/processed/`:
  - `videos_resumen.json`
- `analisis_canal/reports/`:
  - `reporte_resumen.md`

## Configuracion clave

- `CHANNEL_ID`: si esta vacio se usa `mine=True` y se consulta el canal autenticado.
- `MAX_VIDEOS`: limite de videos a analizar.
- `CLIENT_SECRETS_FILE`: ruta de tu OAuth client.

## Limites importantes

- La Data API no entrega retencion, minutos vistos ni CTR. Para eso necesitas YouTube Analytics API.
- Tampoco expone strikes o reclamos de copyright.

## Dependencias

Las librerias necesarias ya estan en `requirements.txt` en la raiz del repo.
