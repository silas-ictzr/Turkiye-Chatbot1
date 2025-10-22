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
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # 🔧 FAISS ve Numpy dosyalarının doğru yoldan yüklenmesi
        base_path = os.path.dirname(__file__)  # Bu dosyanın bulunduğu klasör
        faiss_path = os.path.join(base_path, "turkiye_index.faiss")
        npy_path = os.path.join(base_path, "turkiye_files.npy")

        if not os.path.exists(faiss_path):
            st.error(f"❌ FAISS dosyası bulunamadı: {faiss_path}")
            st.stop()

        index = faiss.read_index(faiss_path)
        file_names = np.load(npy_path)

        return model, index, file_names
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {e}")
        st.stop()

# -----------------------------
# Benzer içerik arama
# -----------------------------
def get_relevant_texts(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    relevant_files = [file_names[i] for i in indices[0]]
    return relevant_files

# -----------------------------
# Cevap üretimi (RAG)
# -----------------------------
def generate_answer(query):
    try:
        relevant_files = get_relevant_texts(query)
        context = ""
        for f in relevant_files:
            with open(f"docs/temizlenmis/{f}", "r", encoding="utf-8") as file:
                context += file.read() + "\n\n"

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
            st.success(answer)
    else:
        st.warning("Lütfen bir soru yazın.")

