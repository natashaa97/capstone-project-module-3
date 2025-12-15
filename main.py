import os
import json
import time
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

#-------------------------------------------------------
# Vector Store Setup & LLM Initialization 
#-------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="imdb-movies",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

#-------------------------------------------------------        
# AGENT TOOLS FOR MOVIE EXPERT
#-------------------------------------------------------

@tool
def get_relevant_docs(question: str):
    """Retrieve movie documents."""
    results = qdrant.similarity_search(question, k=5)
    out = []

    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        text = getattr(r, "page_content", "")

        out.append({
            "title": meta.get("Series_Title", "Unknown Title"),
            "rating": meta.get("IMDB_Rating"),
            "genre": meta.get("Genre"),
            "overview": meta.get("Overview") or text[:600]
        })

    return json.dumps(out, ensure_ascii=False)


# ============================================================
# CLEAN DOC HELPER
# ============================================================

def clean_doc(r):
    """Normalize movie metadata from Qdrant into a clean dictionary."""
    meta = getattr(r, "metadata", {}) or {}
    text = getattr(r, "page_content", "")

    return {
        "title": meta.get("Series_Title", "Unknown Title"),
        "rating": float(meta.get("IMDB_Rating", 0) or 0),
        "genre": meta.get("Genre", "Unknown"),
        "overview": meta.get("Overview") or text[:600]
    }
# ============================================================


@tool
def recommend_movies(payload: str):
    """Recommend movies, accepting JSON or raw string."""
    try:
        params = json.loads(payload)
    except:
        params = {"query": payload, "k": 5}

    q = params.get("query", "")
    genre = params.get("genre")
    year_from = params.get("year_from")
    year_to = params.get("year_to")
    k = params.get("k", 5)

    enriched = q
    if genre:
        enriched += f" genre:{genre}"
    if year_from or year_to:
        enriched += f" year:{year_from}-{year_to}"

    results = qdrant.similarity_search(enriched, k=k)
    out = []

    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        text = getattr(r, "page_content", "")

        out.append({
            "title": meta.get("Series_Title", "Unknown Title"),
            "rating": meta.get("IMDB_Rating"),
            "genre": meta.get("Genre"),
            "overview": meta.get("Overview") or text[:600]
        })

    return json.dumps(out, ensure_ascii=False)



@tool
def summarize_docs(payload: str):
    """Summarize movie list with strong formatting."""
    try:
        docs = json.loads(payload)
    except:
        return "Invalid summarization payload."

    combined = "\n".join([d.get("overview", "") for d in docs])[:20000]

    prompt = f"Ringkas film berikut menjadi bullet points:\n\n{combined}"

    result = llm.invoke([{"role": "user", "content": prompt}])


    if hasattr(result, "content"):
        return result.content
    return result["messages"][-1].content

@tool
def similar_movies(movie_title: str):
    """Return movies most similar to the given movie title using vector similarity."""

    results = qdrant.similarity_search(movie_title, k=6)

    # Skip the first result because it's usually the same movie
    movies = [clean_doc(r) for r in results[1:6]]

    out = f"### 🎬 Film yang Mirip Dengan **{movie_title.title()}**\n\n"
    for i, m in enumerate(movies, 1):
        out += f"""
**{i}. {m['title']}** (⭐ {m['rating']})
Genre: {m['genre']}
Ringkasan:
- {m['overview'][:140]}...
---
"""
    return out


@tool
def recommend_by_mood(mood: str):
    """Recommend movies based on emotional mood: healing, cozy, sad, thrilling, inspiring, etc."""
    
    mood_map = {
        "healing": ["drama", "family"],
        "cozy": ["romance", "comedy", "feel good"],
        "sedih": ["drama", "tragedy"],
        "sad": ["drama", "tragedy"],
        "thrilling": ["thriller", "mystery", "crime"],
        "inspiring": ["biography", "drama", "sport"],
        "romantic": ["romance"],
        "scary": ["horror"],
        "fun": ["comedy", "adventure"]
    }

    mood_key = mood.lower().strip()
    genres = mood_map.get(mood_key, ["drama"])

    query = " ".join([f"genre:{g}" for g in genres])
    docs = qdrant.similarity_search(query, k=5)

    movies = [clean_doc(d) for d in docs]

    out = (
        f"🎭 **Rekomendasi Film Berdasarkan Mood *{mood.title()}***\n"
        f"------------------------------------------------\n\n"
    )

    for i, m in enumerate(movies, 1):
        out += (
            f"**{i}. {m['title']}** (⭐ {m['rating']})\n"
            f"   Genre: {m['genre']}\n"
            f"   Ringkasan:\n"
            f"   - {m['overview'][:150]}...\n"
            f"---\n"
        )

    return out




@tool
def compare_movies(payload: str):
    """Compare 2 movies safely."""
    try:
        params = json.loads(payload)
    except:
        parts = payload.replace("vs", ",").split(",")
        if len(parts) >= 2:
            params = {"movie1": parts[0].strip(), "movie2": parts[1].strip()}
        else:
            return "Format compare tidak valid."

    m1 = params["movie1"]
    m2 = params["movie2"]

    docs1 = qdrant.similarity_search(m1, k=1)
    docs2 = qdrant.similarity_search(m2, k=1)

    def extract(doc):
        meta = getattr(doc, "metadata", {}) or {}
        t = getattr(doc, "page_content", "")
        return {
            "title": meta.get("Series_Title"),
            "rating": meta.get("IMDB_Rating"),
            "genre": meta.get("Genre"),
            "year": meta.get("Released_Year"),
            "overview": meta.get("Overview") or t[:600]
        }

    return json.dumps({
        "movie1": extract(docs1[0]),
        "movie2": extract(docs2[0])
    }, ensure_ascii=False)



@tool
def top_movies_by_genre(payload: str):
    """Return top 5 movies of a genre."""
    try:
        params = json.loads(payload)
        genre = params.get("genre") or payload
    except:
        genre = payload

    enriched = f"genre:{genre}"
    results = qdrant.similarity_search(enriched, k=25)

    out = []
    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        text = getattr(r, "page_content", "")

        out.append({
            "title": meta.get("Series_Title"),
            "rating": float(meta.get("IMDB_Rating", 0) or 0),
            "genre": meta.get("Genre"),
            "overview": meta.get("Overview") or text[:400]
        })

    
    out = sorted(out, key=lambda x: x["rating"], reverse=True)

    return json.dumps(out[:5], ensure_ascii=False)



tools = [
    get_relevant_docs,
    recommend_movies,
    summarize_docs,
    compare_movies,
    top_movies_by_genre,
    recommend_by_mood,
    similar_movies
]


def chat_movie_expert(question: str):

    system_prompt = """
Kamu adalah Autonomous Movie Expert.

=== GUNAKAN FORMAT INI 100% SAMA SETIAP KALI ===

1. Oldboy (Rating: 8.4)
   Genre: Action, Mystery, Thriller
   Ringkasan:
   - Seorang pria dikurung selama 15 tahun tanpa alasan jelas.
   - Setelah bebas, ia mencari dalang di balik penderitaannya.
---

2. The Chaser (Rating: 7.9)
   Genre: Crime, Thriller
   Ringkasan:
   - Mantan detektif mengejar pembunuh berantai.
   - Cerita intens dan penuh ketegangan.
---

RULE WAJIB:
- Tiap film HARUS bernomor (1., 2., 3., ...)
- Rating selalu dalam format (Rating: X.X)
- Genre HARUS baris baru
- Ringkasan HARUS bullet points "- "
- Antar film HARUS dipisahkan '---'
- Tidak boleh paragraf panjang
- Tidak boleh JSON mentah
- Output harus bersih, rapi, estetis

FORMAT COMPARE:
Judul 1 vs Judul 2
Rating:
- Judul 1: X.X
- Judul 2: X.X

Genre:
- Judul 1: ...
- Judul 2: ...

Ringkasan:
- poin 1
- poin 2

FORMAT TOP GENRE:
Gunakan format list film di atas.

Gunakan data dari tools secara akurat.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})


    # FIX: AIMessage or dict
    if hasattr(result, "content"):
        answer = result.content
    else:
        answer = result["messages"][-1].content

    # Token usage tracking
    total_in = 0
    total_out = 0
    tool_msgs = []

    for m in result.get("messages", []) if isinstance(result, dict) else []:
        if isinstance(m, ToolMessage):
            tool_msgs.append(m.content)

        meta = getattr(m, "response_metadata", {})
        if "usage_metadata" in meta:
            total_in += meta["usage_metadata"].get("input_tokens", 0)
            total_out += meta["usage_metadata"].get("output_tokens", 0)

    price = 17000 * (total_in * 0.15 + total_out * 0.6) / 1_000_000

    return {
        "answer": answer,
        "tool_messages": tool_msgs,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "price": price
    }


#-------------------------------------------------------
# STREAMLIT CHAT INTERFACE
#-------------------------------------------------------

st.title("🎬 Movie Expert Chat Bot")
# st.image("./Movie Expert/header_img.png")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Tanyakan apa saja tentang film…")

if prompt:
    st.session_state.messages.append({"role": "Human", "content": prompt})

    with st.chat_message("Human"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        response = chat_movie_expert(prompt)

        final_text = response["answer"]

        box = st.empty()
        streamed = ""

        for w in final_text.split():
            streamed += w + " "
            box.markdown(streamed)
            time.sleep(0.018)

        st.session_state.messages.append({"role": "AI", "content": final_text})

    if response["tool_messages"]:
        with st.expander("Tool Calls"):
            st.code(response["tool_messages"])

    with st.expander("Usage Details"):
        st.code(
            f"Input tokens : {response['total_input_tokens']}\n"
            f"Output tokens: {response['total_output_tokens']}\n"
            f"Estimated cost: Rp {response['price']:.4f}"
        )
