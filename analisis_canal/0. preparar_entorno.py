from config import CLIENT_SECRETS_FILE, TOKEN_FILE, ensure_directories


def main():
    print("=== Preparar entorno ===")
    ensure_directories()
    print("[OK] Directorios listos")

    if CLIENT_SECRETS_FILE.exists():
        print(f"[OK] client_secrets.json encontrado: {CLIENT_SECRETS_FILE}")
    else:
        print(f"[ERROR] Falta client_secrets.json en: {CLIENT_SECRETS_FILE}")
        print("Copia tu client_secrets.json a esa ruta o actualiza analisis_canal/config.py")

    if TOKEN_FILE.exists():
        print(f"[OK] Token encontrado: {TOKEN_FILE}")
    else:
        print("[INFO] No hay token guardado aun. Se creara al autenticar.")


if __name__ == "__main__":
    main()
