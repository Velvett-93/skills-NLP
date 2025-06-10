import pandas as pd
from langdetect import detect
import spacy
from deep_translator import GoogleTranslator


#Carga y preprocesamiento de texto: Leer datos desde un archivo y limpiarlos.
df = pd.read_csv("C:\\Users\\Velvett\\Documents\\Python\\NLP\\mensajes_clientes.csv")  

#Detección de idioma: Uso de la libreria langdetect
def detectar_y_traducir(texto, destino='es'):
    idioma_original = detect(texto)
    if idioma_original != destino:
        #Uso de deep_translator para traducir mensajes a un idioma base en este caso 'Español'.
        texto = GoogleTranslator(source=idioma_original, target=destino).translate(texto)
    return texto


# Aplicar a los mensajes
df['mensaje_traducido'] = df['mensaje'].apply(lambda x: detectar_y_traducir(str(x)))

# Tokenización: Uso de spaCy para dividir el texto traducido en palabras.
# Cargar modelo en español
nlp = spacy.load("es_core_news_sm")

#Lematización:  Uso de spaCy para obtener los lemas de cada token.
#Reducir las palabras a su forma base o lema (por ejemplo: "corriendo" → "correr").
def procesar_texto(texto):
    doc = nlp(texto)
    #Eliminación de stopwords: Filtro durante la tokenización.
    # Quitar palabras comunes que no aportan mucho significado (como “el”, “la”, “y”).
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

df['tokens'] = df['mensaje_traducido'].apply(procesar_texto)

#Extracción de entidades nombradas (Named Entity Recognition - NER):Uso de spaCy para detectar entidades.
#Detectar y clasificar elementos importantes del texto como nombres de personas, empresas, fechas, ubicaciones.
def extraer_entidades(texto):
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return entidades

df['entidades'] = df['mensaje_traducido'].apply(extraer_entidades)

print(df[['mensaje', 'mensaje_traducido']])

print(df[['mensaje', 'mensaje_traducido', 'tokens', 'entidades']])