from googleapiclient.errors import HttpError
from authenticate import authenticate

def consulta_disponibilidad_API():
    try:
        youtube = authenticate()
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode="US",
            maxResults=1,
        )
        response = request.execute()
        print("La solicitud se ha ejecutado correctamente.")
        print(response)
    except HttpError as e:
        if e.resp.status == 403:
            print("Has excedido tu cuota de la API de YouTube. Por favor, espera hasta que se restablezca.")
        else:
            print("Ocurrió un error: %s" % e)
    except Exception as e:
        print("Ocurrió un error inesperado: %s" % e)
        
consulta_disponibilidad_API()