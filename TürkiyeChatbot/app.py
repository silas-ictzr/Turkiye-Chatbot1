import os as _os
try:
    import streamlit as _st  # type: ignore
    _is_streamlit = True
except Exception:
    _is_streamlit = False

if _is_streamlit:
    import streamlit as st
    st.set_page_config(page_title="Türkiye Chatbot", page_icon="🇹🇷", layout="centered")

    from openai import OpenAI
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv

    load_dotenv()
    api_key = _os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OpenAI API Key bulunamadı! Lütfen .env dosyasını kontrol edin.")
        st.stop()

    client = OpenAI(api_key=api_key)

    @st.cache_resource
    def load_components():
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
        
        try:
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            index = faiss.read_index(faiss_path)
            file_names = np.load(npy_path, allow_pickle=True)
            return model, index, file_names
        except Exception as e:
            st.error(f"❌ Model yüklenirken hata: {e}")
            st.stop()

    if "model" not in st.session_state:
        st.session_state.model, st.session_state.index, st.session_state.file_names = load_components()

    def get_relevant_texts(query, top_k=2):
        query_embedding = st.session_state.model.encode([query])
        distances, indices = st.session_state.index.search(np.array(query_embedding).astype("float32"), top_k)
        relevant_files = [st.session_state.file_names[i] for i in indices[0]]
        return relevant_files

    def generate_answer(query):
        relevant_files = get_relevant_texts(query)
        context = ""
        # Streamlit Cloud için dosya yolu
        docs_path = os.path.join("TürkiyeChatbot", "docs", "temizlenmis") if os.path.exists("TürkiyeChatbot") else "docs/temizlenmis"
        for f in relevant_files:
            file_path = os.path.join(docs_path, f)
            with open(file_path, "r", encoding="utf-8") as file:
                context += file.read() + "\n\n"
        if len(context) > 8000:
            context = context[:8000]
        prompt = f"Türkiye hakkında bilgiler:\n{context}\nSoru: {query}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    st.title("🇹🇷 Türkiye Bilgi Chatbot")
    st.write("Temizlenmiş dokümanlardan bilgiye dayalı olarak yanıt verir.")
    query = st.text_input("Sorunuzu yazın:")
    if st.button("Sor"):
        if query:
            with st.spinner("Yanıt üretiliyor..."):
                answer = generate_answer(query)
                st.markdown(f"**Yanıt:**\n\n{answer}")
        else:
            st.warning("Lütfen bir soru yazın.")

    # Streamlit çalışırken aşağıdaki notebook benzeri bölümlerin çalışmasını durdur
    st.stop()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hücre A: docs klasörü oluştur ve turkiye_bilgileri.txt yaz
from pathlib import Path

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

content = """#### 🏛️ Genel Bilgiler
Türkiye, resmî adıyla Türkiye Cumhuriyeti, hem Avrupa hem de Asya kıtalarında toprakları bulunan bir ülkedir. Başkenti Ankara; en kalabalık şehri İstanbul'dur. Türkiye’nin yüzölçümü yaklaşık 783.000 kilometrekaredir. Ülke, kuzeyde Karadeniz, batıda Ege Denizi, güneyde Akdeniz ile çevrilidir. Avrupa ve Asya arasında köprü konumunda olması, stratejik önemini artırmaktadır.

#### 📜 Tarih ve Kuruluş
Türkiye Cumhuriyeti 29 Ekim 1923'te Mustafa Kemal Atatürk liderliğinde kurulmuştur. Osmanlı İmparatorluğu’nun ardından yeni bir ulus-devlet olarak şekillenmiştir. Atatürk, Cumhuriyetin ilanıyla birlikte eğitim, hukuk, ekonomi ve toplumsal yaşamda çok sayıda reform gerçekleştirmiştir. Kadınlara seçme ve seçilme hakkı verilmesi, Latin alfabesinin kabulü, medreselerin kapatılması ve laik eğitim sistemine geçilmesi bu reformlardan bazılarıdır.

#### ⚖️ Yönetim ve Siyasi Sistem
Türkiye’nin yönetim biçimi cumhuriyettir. Devlet, yasama, yürütme ve yargı olmak üzere üç erk üzerine kuruludur. Yasama görevi Türkiye Büyük Millet Meclisi (TBMM) tarafından yürütülür. Cumhurbaşkanı, yürütmenin başıdır ve aynı zamanda devletin başkanıdır. Türkiye, çok partili demokratik bir sistemle yönetilmektedir.

#### 🗺️ Coğrafya ve Bölgeler
Türkiye coğrafi olarak yedi bölgeye ayrılır: Marmara, Ege, Akdeniz, İç Anadolu, Karadeniz, Doğu Anadolu ve Güneydoğu Anadolu. Marmara Bölgesi, ülkenin sanayi ve ticaret merkezi konumundadır. Ege Bölgesi zeytin, incir ve pamuk üretimiyle öne çıkar. Akdeniz Bölgesi tarım ve turizm açısından gelişmiştir. İç Anadolu Bölgesi geniş ovalarıyla bilinir. Karadeniz Bölgesi çay ve fındık üretimiyle tanınır. Doğu ve Güneydoğu Anadolu bölgeleri dağlık yapısı ve tarımsal faaliyetleriyle ön plandadır.

#### 💰 Ekonomi
Türkiye ekonomisi sanayi, tarım, hizmet ve turizm sektörlerine dayanır. Otomotiv, tekstil, inşaat, savunma sanayi ve beyaz eşya üretimi ülkenin ihracatında önemli yer tutar. Tarımda buğday, arpa, pamuk, zeytin, fındık, üzüm ve çay öne çıkan ürünlerdir. Turizm, ülke ekonomisine büyük katkı sağlar; Antalya, Kapadokya, İstanbul, İzmir ve Muğla en çok ziyaret edilen yerlerdendir.

#### 💱 Para Birimi ve Dil
Türkiye’nin resmi para birimi Türk Lirası’dır. Resmî dili Türkçedir. Nüfusu 2024 yılı itibarıyla yaklaşık 85 milyon kişidir. Ülke nüfusu gençtir ve büyük kısmı şehirlerde yaşamaktadır. İstanbul, Ankara, İzmir, Bursa, Antalya, Adana ve Konya en büyük şehirler arasındadır.

#### 🌍 Komşular ve Stratejik Konum
Türkiye’nin komşuları arasında Yunanistan ve Bulgaristan (batı), Gürcistan, Ermenistan, Nahçıvan (Azerbaycan), İran (doğu), Irak ve Suriye (güney) yer alır. Ülke ayrıca üç denize kıyısı olan nadir devletlerden biridir. Boğazlar sistemi (İstanbul ve Çanakkale Boğazları) uluslararası deniz ulaşımı açısından stratejik öneme sahiptir.

#### ☀️ İklim
Türkiye’nin iklimi bölgeden bölgeye değişir. Karadeniz kıyıları nemli ve yağışlı, İç Anadolu kurak ve karasal, Akdeniz kıyıları ise sıcak ve kuraktır. Bu çeşitlilik tarım ürünlerini ve bitki örtüsünü doğrudan etkiler. Ülke, dağlık alanlar ve verimli ovalarla doğal çeşitliliğe sahiptir.

#### 🤝 Uluslararası İlişkiler
Türkiye, NATO üyesi olup Avrupa Konseyi, OECD, G20, İslam İşbirliği Teşkilatı ve Türk Devletleri Teşkilatı gibi uluslararası örgütlerde yer alır. Avrupa Birliği ile 1963’ten bu yana ortaklık ilişkisi yürütülmektedir. Dış politika, barışçıl diplomasi ve bölgesel iş birliği üzerine kuruludur.

#### 🎭 Kültür ve Sanat
Türk kültürü, tarih boyunca Anadolu’da yaşamış uygarlıkların mirasını taşır. Geleneksel el sanatları, müzik, halk dansları, edebiyat ve mutfak kültürü önemli yer tutar. Türk mutfağında kebaplar, dolmalar, börekler, zeytinyağlılar ve tatlılar öne çıkar. Ayrıca çay ve kahve kültürü günlük yaşamın ayrılmaz bir parçasıdır.

#### 🎓 Eğitim
Türkiye’nin eğitim sistemi zorunlu 12 yıllık temel eğitimi kapsar. Yükseköğretim kurumları arasında Orta Doğu Teknik Üniversitesi, Boğaziçi Üniversitesi, İstanbul Teknik Üniversitesi ve Hacettepe Üniversitesi gibi üniversiteler uluslararası başarılar elde etmiştir. Son yıllarda yapay zeka, bilişim ve savunma teknolojileri alanlarında büyük gelişmeler yaşanmıştır.

#### 🏥 Sağlık Sistemi
Türkiye’de sağlık hizmetleri devlet ve özel sektör tarafından sağlanmaktadır. Şehir hastaneleri projeleri ile sağlık altyapısı güçlendirilmiştir. Sosyal güvenlik sistemi geniş nüfus kesimlerini kapsar. Türkiye, medikal turizmde de önemli bir destinasyon haline gelmiştir.

#### 🎨 Sanat ve Miras
Türkiye sanatı tarih boyunca farklı dönemlerin etkilerini taşır. Osmanlı mimarisi, Selçuklu taş işçiliği, modern resim ve heykel örnekleri Türk sanatının zenginliğini yansıtır. Mimar Sinan, Osman Hamdi Bey, Bedri Rahmi Eyüboğlu gibi sanatçılar kültürel mirasın öncülerindendir.

#### 🏞️ Turizm ve Doğal Güzellikler
Türkiye, hem doğası hem de tarihiyle dünya çapında turistik bir ülkedir. Kapadokya’nın peribacaları, Pamukkale travertenleri, Efes Antik Kenti, Topkapı Sarayı, Ayasofya, Nemrut Dağı ve Göbeklitepe gibi yerler UNESCO Dünya Mirası Listesi’ndedir. Bu alanlar her yıl milyonlarca turist tarafından ziyaret edilir.

#### 🛰️ Teknoloji ve Savunma Sanayi
Türkiye, son yıllarda teknoloji ve savunma sanayi alanında önemli ilerlemeler kaydetmiştir. Bayraktar İHA’lar, Togg yerli otomobili, Türksat uyduları ve TEKNOFEST gibi girişimler ülkenin inovasyon kapasitesini göstermektedir. Genç nüfusun teknolojiye ilgisi bu gelişmeleri desteklemektedir.

#### 🇹🇷 Sonuç
Türkiye, hem tarihi kökleri hem modern gelişmeleriyle dinamik bir ülkedir. Coğrafi konumu, kültürel zenginliği, genç nüfusu ve stratejik önemiyle bölgesinde ve dünyada etkili bir konuma sahiptir.

"""



file_path = DOCS_DIR / "turkiye_bilgileri.txt"
file_path.write_text(content, encoding="utf-8")
print(f"Oluşturuldu: {file_path.resolve()}")


# In[2]:


# Hücre B: turkiye_bilgileri.txt dosyasını '####' başlıklarına göre ayır
import os

# Girdi dosyası
input_file = DOCS_DIR / "turkiye_bilgileri.txt"

# Çıktı klasörü
output_folder = DOCS_DIR / "bolumler"
os.makedirs(output_folder, exist_ok=True)

# Dosyayı oku
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Bölümleri ayır
sections = content.split("####")
for section in sections:
    section = section.strip()
    if not section:
        continue

    lines = section.split("\n")
    title = lines[0].strip().lower()
    title = (
        title.replace("🏛️", "")
        .replace("📜", "")
        .replace("⚖️", "")
        .replace("🗺️", "")
        .replace("💰", "")
        .replace("💱", "")
        .replace("🌍", "")
        .replace("☀️", "")
        .replace("🤝", "")
        .replace("🎭", "")
        .replace("🎓", "")
        .replace("🏥", "")
        .replace("🎨", "")
        .replace("🏞️", "")
        .replace("🛰️", "")
        .replace("🇹🇷", "")
        .strip()
        .replace(" ", "_")
    )

    # Dosya adı oluştur
    filename = f"turkiye_{title}.txt"
    filepath = output_folder / filename

    # Bölüm içeriğini kaydet
    with open(filepath, "w", encoding="utf-8") as out:
        out.write("\n".join(lines[1:]).strip())

print("✅ Bölme tamamlandı! -> 'docs/bolumler' klasörünü kontrol et.")


# In[3]:


#pip install openai tiktoken


# In[4]:


with open(".gitignore", "w", encoding="utf-8") as f:
    f.write("""# Gizli dosyalar ve anahtarlar
.env

# Python önbellekleri
__pycache__/

# FAISS ve numpy vektör dosyaları
*.faiss
*.npy

# Jupyter geçici dosyaları
.ipynb_checkpoints/

""")

print(".gitignore dosyası başarıyla oluşturuldu ✅")


# In[5]:


# Hücre C2: OpenAI istemcisi ve dokümanları yükle
from openai import OpenAI
import os

from dotenv import load_dotenv
import os

load_dotenv()  # .env dosyasını oku
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# Bölümlerin bulunduğu klasör
docs_path = Path("docs/temizlenmis")

# Tüm dokümanları oku
documents = {}
for file in docs_path.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        documents[file.stem] = f.read()

print(f"{len(documents)} doküman yüklendi ✅")


# In[6]:


# Hücre C3: Basit RAG fonksiyonu

def turkiye_chatbot(question):
    """
    1️⃣ Kullanıcının sorusunu alır
    2️⃣ En alakalı dosyayı bulur
    3️⃣ O dosyanın içeriğini GPT'ye gönderip yanıt oluşturur
    """

    from openai import OpenAI
    import difflib

    # En alakalı dosyayı başlıktan tahmin et (basit eşleşme)
    keywords = list(documents.keys())
    closest = difflib.get_close_matches(question.lower(), keywords, n=1)
    selected_doc = closest[0] if closest else list(documents.keys())[0]

    context = documents[selected_doc]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen Türkiye hakkında bilgi veren bir asistansın."},
            {"role": "user", "content": f"Soru: {question}\n\nBilgiler:\n{context}"}
        ],
        temperature=0.4,
    )

    answer = response.choices[0].message.content
    return f"🧭 Kaynak: {selected_doc.replace('turkiye_', '').replace('_', ' ')}\n\n💬 {answer}"


# In[7]:


# Hücre C4: Chatbot testleri
sorular = [
    "Türkiye'nin başkenti neresidir?",
    "Türkiye ekonomisi hangi sektörlere dayanır?",
    "Türkiye'nin komşuları kimlerdir?",
    "Türk mutfağında hangi yemekler meşhurdur?",
    "Türkiye'nin iklimi nasıldır?"
]

for soru in sorular:
    print(f"❓ {soru}")
    print(turkiye_chatbot(soru))
    print()


# In[8]:


#pip install nltk


# In[9]:


# Hücre C: Veri temizleme işlemleri (lowercase, noktalama kaldırma, stopword çıkarma)
import os
import string
import nltk
from nltk.corpus import stopwords

# Gerekli verileri indir
nltk.download("stopwords")

# Türkçe stopword listesi
turkish_stopwords = stopwords.words("turkish")

# Girdi ve çıktı klasörleri
input_folder = DOCS_DIR / "bolumler"
output_folder = DOCS_DIR / "temizlenmis"
os.makedirs(output_folder, exist_ok=True)

def temizle_metin(text):
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama kaldır
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Stopword temizleme
    kelimeler = text.split()
    temiz = [k for k in kelimeler if k not in turkish_stopwords]
    return " ".join(temiz)

# Her dosyayı sırayla işle
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = input_folder / filename
        output_path = output_folder / filename

        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        temizlenmis = temizle_metin(content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(temizlenmis)

        print(f"✅ Temizlendi: {filename}")

print("\n🎯 Tüm dosyalar temizlenip 'docs/temizlenmis' klasörüne kaydedildi!")


# In[10]:


#!pip install sentence-transformers


# In[11]:


#!pip install sentence-transformers faiss-cpu


# In[12]:


# Hücre D: Embedding (vektör) oluşturma ve hafıza kaydı

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Modeli yükle (Türkçe destekli)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Temizlenmiş veri klasörü
input_folder = DOCS_DIR / "temizlenmis"

# Tüm metinleri oku
texts = []
file_names = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        path = input_folder / filename
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(content)
            file_names.append(filename)

# Embedding oluştur
embeddings = model.encode(texts)

# FAISS index oluştur
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Kaydet
faiss.write_index(index, "turkiye_index.faiss")
np.save("turkiye_files.npy", np.array(file_names))

print("✅ Embedding ve FAISS index başarıyla oluşturuldu!")


# In[13]:


# Hücre E: Chatbot - Soru Cevaplama Sistemi

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Modeli tekrar yükle (embedding oluştururkenki ile aynı olmalı)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# FAISS ve dosya isimlerini yükle
index = faiss.read_index("turkiye_index.faiss")
file_names = np.load("turkiye_files.npy", allow_pickle=True)

# Fonksiyon: Kullanıcı sorusuna göre en ilgili dokümanı bul
def en_ilgili_dosya_bul(soru, top_k=1):
    soru_embed = model.encode([soru])
    distances, indices = index.search(soru_embed, top_k)
    en_yakin = indices[0][0]
    return file_names[en_yakin]

# Chatbot fonksiyonu
def turkiye_chatbot(soru):
    dosya = en_ilgili_dosya_bul(soru)
    with open(DOCS_DIR / "temizlenmis" / dosya, "r", encoding="utf-8") as f:
        cevap = f.read()
    return f"💬 **Kaynak:** {dosya}\n\n{cevap[:1200]}..."  # ilk 1200 karakteri göster

# Test et
sorular = [
    "Türkiye'nin başkenti neresidir?",
    "Türkiye ekonomisi hangi sektörlere dayanır?",
    "Türkiye'nin komşuları kimlerdir?",
    "Türk mutfağında hangi yemekler meşhurdur?"
]

for soru in sorular:
    print(f"❓ {soru}")
    print(turkiye_chatbot(soru))
    print("-" * 80)


# In[14]:


# Hücre E: RAG – Retrieval + Generation (Chatbot)

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# OpenAI istemcisi
from dotenv import load_dotenv
import os

load_dotenv()  # .env dosyasını oku
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# Modeli ve FAISS index'i yükle
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index("turkiye_index.faiss")
file_names = np.load("turkiye_files.npy", allow_pickle=True)

# Temizlenmiş dosyaların bulunduğu klasör
temizlenmis_klasor = DOCS_DIR / "temizlenmis"

def rag_chatbot(question, top_k=2):
    """
    RAG tabanlı chatbot:
    1️⃣ Soruyu embedding'e çevirir
    2️⃣ FAISS ile en benzer bölümleri bulur
    3️⃣ GPT modeline bağlam olarak gönderir ve doğal cevap üretir
    """

    # 1️⃣ Soruyu vektöre dönüştür
    query_embedding = model.encode([question])

    # 2️⃣ FAISS ile en benzer dokümanları bul
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), top_k)

    # 3️⃣ En alakalı metinleri oku
    context_texts = []
    for i in indices[0]:
        file_path = temizlenmis_klasor / file_names[i]
        with open(file_path, "r", encoding="utf-8") as f:
            context_texts.append(f.read())

    # Bağlamı birleştir
    context = "\n\n".join(context_texts)

    # 4️⃣ GPT'ye gönder – Generation aşaması
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "Sen Türkiye hakkında güvenilir bilgiler sunan bir asistansın."},
            {"role": "user", "content": f"Soru: {question}\n\nKaynak bilgiler:\n{context}\n\nCevabı açık, özet ve doğal biçimde oluştur."}
        ]
    )

    answer = response.choices[0].message.content

    kaynaklar = [file_names[i].replace("turkiye_", "").replace(".txt", "").replace("_", " ") for i in indices[0]]
    kaynaklar_str = ", ".join(kaynaklar)

    return f"📚 Kaynaklar: {kaynaklar_str}\n\n💬 {answer}"


# In[15]:


sorular = [
    "Türkiye'nin başkenti neresidir?",
    "Türkiye ekonomisi hangi sektörlere dayanır?",
    "Türkiye'nin iklimi nasıldır?",
    "Türkiye'nin en önemli turistik yerleri hangileridir?",
    "Türkiye'nin eğitim sistemi nasıl işler?"
]

for soru in sorular:
    print(f"❓ {soru}")
    print(rag_chatbot(soru))
    print()


# In[16]:


#pip install streamlit


# In[17]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\n\n# 🟢 BU SATIRI EN ÜSTE KOY\nst.set_page_config(page_title="Türkiye Chatbot", page_icon="🇹🇷", layout="centered")\n\nfrom openai import OpenAI\nimport faiss\nimport numpy as np\nfrom sentence_transformers import SentenceTransformer\n\n# -----------------------------\n# OpenAI istemcisi\n# -----------------------------\nfrom dotenv import load_dotenv\nimport os\n\nload_dotenv()  # .env dosyasını oku\napi_key = os.getenv("OPENAI_API_KEY")\n\nclient = OpenAI(api_key=api_key)\n\n# -----------------------------\n# Model ve index yükleme\n# -----------------------------\n@st.cache_resource\ndef load_components():\n    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")\n    index = faiss.read_index("turkiye_index.faiss")\n    file_names = np.load("turkiye_files.npy")\n    return model, index, file_names\n\nmodel, index, file_names = load_components()\n\n# -----------------------------\n# Benzer içerik arama\n# -----------------------------\ndef get_relevant_texts(query, top_k=2):\n    query_embedding = model.encode([query])\n    distances, indices = index.search(np.array(query_embedding), top_k)\n    relevant_files = [file_names[i] for i in indices[0]]\n    return relevant_files\n\n# -----------------------------\n# Cevap üretimi (RAG)\n# -----------------------------\ndef generate_answer(query):\n    relevant_files = get_relevant_texts(query)\n    context = ""\n    for f in relevant_files:\n        with open(f"docs/temizlenmis/{f}", "r", encoding="utf-8") as file:\n            context += file.read() + "\\n\\n"\n\n    prompt = f"""\n    Aşağıda Türkiye hakkında bazı bilgiler ve bir kullanıcı sorusu var.\n    Soruyu bu bilgilerden yararlanarak yanıtla.\n    Cevabın doğal, kısa ve bilgilendirici olsun.\n\n    Bilgiler:\n    {context}\n\n    Soru: {query}\n    """\n    response = client.chat.completions.create(\n        model="gpt-4o-mini",\n        messages=[{"role": "user", "content": prompt}]\n    )\n    return response.choices[0].message.content\n\n# -----------------------------\n# Streamlit Arayüzü\n# -----------------------------\nst.title("🇹🇷 Türkiye Bilgi Chatbot")\nst.write("Temizlenmiş dokümanlardan bilgiye dayalı olarak yanıt verir.")\n\nuser_input = st.text_input("Sorunuzu yazın:", placeholder="Örneğin: Türkiye\'nin başkenti neresidir?")\n\nif st.button("Sor"):\n    if user_input:\n        with st.spinner("Yanıt üretiliyor..."):\n            answer = generate_answer(user_input)\n            st.success(answer)\n    else:\n        st.warning("Lütfen bir soru yazın.")\n')


# In[18]:


# Streamlit uygulamasını çalıştırmak için terminal'de şu komutu kullanın:
# streamlit run app.py


# In[19]:


#pip install python-dotenv


# In[21]:


with open("requirements.txt", "w") as f:
    f.write("""streamlit
openai
python-dotenv
google-generativeai
PyPDF2
langchain
chromadb
""")


# In[ ]:




