import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# Configuración
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicialización
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("📥 Cargando modelo de embeddings local...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

class ChatRequest(BaseModel):
    message: str

def get_relevant_docs(query: str, limit: int = 4):
    """Búsqueda semántica en Supabase con umbral más flexible, optimizada para ahorrar tokens."""
    embedding = embed_model.encode(query).tolist()
    
    rpc_response = supabase.rpc(
        "buscar_documentos_contables",
        {
            "query_embedding": embedding,
            "match_threshold": 0.1, # Bajamos el umbral para captar más detalles
            "match_count": limit
        }
    ).execute()
    
    return rpc_response.data

@app.post("/chat")
async def chat(request_data: ChatRequest, req: Request):
    try:
        # Obtener IP del cliente (priorizando X-Forwarded-For si está tras proxy)
        client_ip = req.headers.get("x-forwarded-for", req.client.host).split(",")[0].strip()
        
        # Calcular límite temporal de 12 horas
        time_limit = (datetime.utcnow() - timedelta(hours=12)).isoformat()
        
        # Verificar en base de edad cuántas preguntas van en ese periodo de tiempo
        response_limits = supabase.table("notebooklm_chat_logs").select("id").eq("ip_address", client_ip).gte("created_at", time_limit).execute()
        
        if len(response_limits.data) >= 6:
            msg = "Has alcanzado el límite gratuito de 6 consultas tributarias por cada 12 horas. 🚫\n\nPara seguir resolviendo todas tus dudas y potenciar tu negocio sin límites con EFI AI, te invitamos a suscribirte a nuestro plan Premium. 🚀"
            return {"response": msg, "sources": [], "limit_reached": True}

        user_msg = request_data.message
        
        # 1. Recuperar contexto relevante
        docs = get_relevant_docs(user_msg)
        context = ""
        for doc in docs:
            nice_title = format_source_title(doc['titulo'])
            # Redujimos de 2500 a 1200 caracteres para ahorrar casi un 50% de tokens por bloque
            context += f"\n--- FUENTE REAL ({nice_title}) ---\n{doc['contenido_completo'][:1200]}\n"

        # 2. Formular respuesta con Groq
        system_prompt = f"""
        Eres la 'Contadora EFI', una experta contable y tributaria CHILENA.
        Tu misión es responder consultas de la 'Comunidad Contable' usando UNICAMENTE el lenguaje técnico del SII (Servicio de Impuestos Internos) y la TGR (Tesorería General de la República).

        REGLAS CRÍTICAS DE CONOCIMIENTO:
        - RAF: Revisión de la Actuación Fiscalizadora. (OJO: Según tus fuentes, la RAF NO tiene plazo de presentación y se puede presentar después del día 91 de una liquidación o giro).
        - RAV: Reposición Administrativa Voluntaria (plazo 30 días).
        - PA: Petición Administrativa.
        - F29: Declaración mensual de IVA.
        - F22: Declaración de Renta.
        
        REGLAS DE RESPUESTA:
        - Sé extremadamente técnica y precisa. Si el contexto dice algo específico (como un número de ley o un plazo), úsalo.
        - BAJO NINGÚN MOTIVO inventes o deduzcas códigos de línea numérica para los formularios (como el F29, F22). Si el usuario te pide la línea exacta y no la ves explícitamente en tu CONTEXTO DE LA COMUNIDAD, responde: "Para el llenado exacto de las líneas del formulario respectivo por este concepto, te sugiero revisar las instrucciones de llenado vigentes del SII."
        - No asumas reglas generales si el contexto habla de un caso particular. Confirma primero el tipo de contribuyente si es necesario.
        - No inventes procedimientos. Si no está en el contexto, di: 'Esa información específica no la tengo en mis registros de la comunidad, pero te sugiero revisar la circular vigente en el SII'.
        - Usa el tono de 'EFI': Amable, cercana ('colega', 'un gusto'), pero muy profesional.
        - Formato WhatsApp: Breve, con emojis y listas de puntos.

        CONTEXTO DE LA COMUNIDAD (Usa esto para responder):
        {context}
        """

        import groq
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3, # Bajamos temperatura para menos creatividad y más precisión
                max_tokens=1500,
            )
        except Exception as groq_err:
            if "Rate limit reached" in str(groq_err) or "rate_limit_exceeded" in str(groq_err):
                print("⚠️ Límite de tokens en Groq primario alcanzado, cambiando a API KEY de Backup...")
                backup_key = os.getenv("GROQ_API_KEY_BACKUP")
                if not backup_key:
                    raise Exception("Límite de tokens alcanzado y no hay API Key de respaldo configurada.")
                
                groq_client_backup = Groq(api_key=backup_key)
                chat_completion = groq_client_backup.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1500,
                )
            else:
                raise groq_err

        response_text = chat_completion.choices[0].message.content
        sources_list = list(set([format_source_title(d['titulo']) for d in docs])) # Usamos set para quitar duplicados lógicos (Parte 1 y Parte 2 del mismo doc)
        
        # 3. Guardar el registro (log) en la base de datos
        try:
            supabase.table("notebooklm_chat_logs").insert({
                "user_message": user_msg,
                "bot_response": response_text,
                "sources_cited": sources_list,
                "model": "llama-3.3-70b-versatile",
                "ip_address": client_ip
            }).execute()
            print("📝 Log de conversación guardado en Supabase.")
        except Exception as log_error:
            print(f"⚠️ Error al guardar el log en base de datos: {log_error}")

        return {"response": response_text, "sources": sources_list}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class LeadRequest(BaseModel):
    nombre: str
    email: str
    phone: str

def format_source_title(title):
    title = title.strip()
    # Si el título es muy feo como un id crudo o 'drive_pdf'
    if title.lower().startswith('drive_pdf') or title.lower().startswith('video_youtube'):
        return "Documento Normativo / Respaldo SII"
    # Limpiamos parte de texto con guiones bajos o extensiones
    title = title.replace('.pdf', '').replace('.txt', '').replace('_', ' ')
    if len(title) > 60:
        title = title[:60] + "..."
    return title.title()

@app.post("/lead")
async def save_lead(request_data: LeadRequest, req: Request):
    try:
        # Obtener IP del cliente para asociarla al chat
        client_ip = req.headers.get("x-forwarded-for", req.client.host).split(",")[0].strip()
        
        supabase.table("leads").insert({
            "name": request_data.nombre,
            "email": request_data.email,
            "phone_number": request_data.phone,
            "ip_address": client_ip
        }).execute()
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error al guardar lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
