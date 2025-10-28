import streamlit as st

# 🟢 BU SATIRI EN ÜSTE KOY
st.set_page_config(page_title="Türkiye Chatbot", page_icon="🇹🇷", layout="centered")

from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# OpenAI istemcisi
# -----------------------------
from dotenv import load_dotenv
import os

load_dotenv()  # .env dosyasını oku
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OpenAI API Key bulunamadı! Lütfen .env dosyasını kontrol edin.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# Model ve index yükleme
# -----------------------------
@st.cache_resource
def load_components():
    try:
        import os
        # Dosyaların varlığını kontrol et
        # Streamlit Cloud kök dizinde çalıştığı için TürkiyeChatbot alt klasörünü ekle
        base_path = "TürkiyeChatbot" if os.path.exists("TürkiyeChatbot") else "."
        faiss_path = os.path.join(base_path, "turkiye_index.faiss")
        npy_path = os.path.join(base_path, "turkiye_files.npy")
        
        if not os.path.exists(faiss_path):
            st.error(f"❌ turkiye_index.faiss dosyası bulunamadı!")
            st.stop()
        if not os.path.exists(npy_path):
            st.error(f"❌ turkiye_files.npy dosyası bulunamadı!")
            st.stop()
        
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        index = faiss.read_index(faiss_path)
        file_names = np.load(npy_path, allow_pickle=True)
        return model, index, file_names
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {e}")
        st.stop()

# Session state ile güvenli bileşen erişimi
if "model" not in st.session_state:
    st.session_state.model, st.session_state.index, st.session_state.file_names = load_components()

# -----------------------------
# Benzer içerik arama
# -----------------------------
def get_relevant_texts(query, top_k=2):
    query_embedding = st.session_state.model.encode([query])
    distances, indices = st.session_state.index.search(np.array(query_embedding).astype("float32"), top_k)
    relevant_files = [st.session_state.file_names[i] for i in indices[0]]
    return relevant_files

# -----------------------------
# Cevap üretimi (RAG)
# -----------------------------
def generate_answer(query):
    try:
        relevant_files = get_relevant_texts(query)
        context = ""
        # Streamlit Cloud için dosya yolu
        import os
        docs_path = os.path.join("TürkiyeChatbot", "docs", "temizlenmis") if os.path.exists("TürkiyeChatbot") else "docs/temizlenmis"
        for f in relevant_files:
            file_path = os.path.join(docs_path, f)
            with open(file_path, "r", encoding="utf-8") as file:
                context += file.read() + "\n\n"

        # Uzun bağlam durumunda kesme (isteğe bağlı güvenlik katmanı)
        if len(context) > 8000:
            context = context[:8000]

        prompt = f"""
        Aşağıda Türkiye hakkında bazı bilgiler ve bir kullanıcı sorusu var.
        Soruyu bu bilgilerden yararlanarak yanıtla.
        Cevabın doğal, kısa ve bilgilendirici olsun.

        Bilgiler:
        {context}

        Soru: {query}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Hata oluştu: {e}"

# -----------------------------
# Streamlit Arayüzü
# -----------------------------
st.title("🇹🇷 Türkiye Bilgi Chatbot")
st.write("Temizlenmiş dokümanlardan bilgiye dayalı olarak yanıt verir.")

# Başarı mesajı
st.success("✅ Sistem hazır! Sorularınızı sorabilirsiniz.")

user_input = st.text_input("Sorunuzu yazın:", placeholder="Örneğin: Türkiye'nin başkenti neresidir?")

if st.button("Sor"):
    if user_input:
        with st.spinner("Yanıt üretiliyor..."):
            answer = generate_answer(user_input)
            st.markdown(f"**Yanıt:**\n\n{answer}")
    else:
        st.warning("Lütfen bir soru yazın.")

