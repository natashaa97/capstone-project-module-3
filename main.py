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

# Pastikan collection sudah ada
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="imdb-movies",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

#-------------------------------------------------------        
# HELPER FUNCTION
#-------------------------------------------------------

def clean_doc(r):
    """Normalize movie metadata from Qdrant into a clean dictionary."""
    meta = getattr(r, "metadata", {}) or {}
    text = getattr(r, "page_content", "")

    return {
        "title": meta.get("Series_Title", "Unknown Title"),
        "rating": float(meta.get("IMDB_Rating", 0) or 0),
        "genre": meta.get("Genre", "Unknown"),
        "overview": meta.get("Overview") or text[:600],
        "year": meta.get("Released_Year", "N/A")
    }

#-------------------------------------------------------        
# AGENT TOOLS - RETURN JSON ONLY
#-------------------------------------------------------

@tool
def get_relevant_docs(question: str):
    """Retrieve movie documents based on search query. Can return up to 20 results."""
    results = qdrant.similarity_search(question, k=20)
    
    out = []
    for r in results:
        movie = clean_doc(r)
        out.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "genre": movie['genre'],
            "overview": movie['overview'][:300]
        })
    
    return json.dumps(out, ensure_ascii=False)


@tool
def recommend_movies(payload: str):
    """Recommend movies based on query, genre, or year range. Can return up to 20 results based on request."""
    try:
        params = json.loads(payload)
    except:
        params = {"query": payload, "k": 10}

    q = params.get("query", "")
    genre = params.get("genre")
    year_from = params.get("year_from")
    year_to = params.get("year_to")
    k = params.get("k", 10)
    
    # Batasi maksimal 20 film
    k = min(k, 20)

    enriched = q
    if genre:
        enriched += f" genre:{genre}"
    if year_from or year_to:
        enriched += f" year:{year_from}-{year_to}"

    results = qdrant.similarity_search(enriched, k=k)
    
    out = []
    for r in results:
        movie = clean_doc(r)
        out.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "genre": movie['genre'],
            "overview": movie['overview'][:300]
        })
    
    return json.dumps(out, ensure_ascii=False)


@tool
def similar_movies(movie_title: str):
    """Return up to 10 movies most similar to the given movie title."""
    results = qdrant.similarity_search(movie_title, k=11)
    
    # Skip first result (usually the same movie)
    out = []
    for r in results[1:11]:
        movie = clean_doc(r)
        out.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "genre": movie['genre'],
            "overview": movie['overview'][:250]
        })
    
    return json.dumps(out, ensure_ascii=False)


@tool
def recommend_by_mood(mood: str):
    """Recommend up to 10 movies based on emotional mood."""
    mood_map = {
        "healing": ["drama", "family"],
        "cozy": ["romance", "comedy"],
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
    docs = qdrant.similarity_search(query, k=10)

    out = []
    for r in docs:
        movie = clean_doc(r)
        out.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "genre": movie['genre'],
            "overview": movie['overview'][:250]
        })

    return json.dumps({"mood": mood, "movies": out}, ensure_ascii=False)


@tool
def compare_movies(payload: str):
    """Compare 2 movies side by side."""
    try:
        params = json.loads(payload)
    except:
        parts = payload.replace("vs", ",").split(",")
        if len(parts) >= 2:
            params = {"movie1": parts[0].strip(), "movie2": parts[1].strip()}
        else:
            return json.dumps({"error": "Format tidak valid"})

    m1 = params["movie1"]
    m2 = params["movie2"]

    docs1 = qdrant.similarity_search(m1, k=1)
    docs2 = qdrant.similarity_search(m2, k=1)

    if not docs1 or not docs2:
        return json.dumps({"error": "Film tidak ditemukan"})

    movie1 = clean_doc(docs1[0])
    movie2 = clean_doc(docs2[0])

    return json.dumps({
        "movie1": movie1,
        "movie2": movie2
    }, ensure_ascii=False)


@tool
def top_movies_by_genre(genre: str):
    """Return top 10 highest rated movies of a genre."""
    enriched = f"genre:{genre}"
    results = qdrant.similarity_search(enriched, k=30)

    movies = [clean_doc(r) for r in results]
    movies = sorted(movies, key=lambda x: x["rating"], reverse=True)[:10]

    out = []
    for m in movies:
        out.append({
            "title": m['title'],
            "rating": m['rating'],
            "genre": m['genre'],
            "overview": m['overview'][:250]
        })

    return json.dumps(out, ensure_ascii=False)


@tool
def search_by_person(payload: str):
    """Search movies by actor or director name. Returns up to 15 movies."""
    try:
        params = json.loads(payload)
        person_name = params.get("name", payload)
        role = params.get("role", "any")  # actor, director, or any
    except:
        person_name = payload
        role = "any"
    
    # Search with person name
    results = qdrant.similarity_search(person_name, k=20)
    
    out = []
    for r in results[:15]:
        meta = getattr(r, "metadata", {}) or {}
        
        # Check if person is in cast or director
        director = meta.get("Director", "")
        stars = meta.get("Star1", "") + " " + meta.get("Star2", "") + " " + meta.get("Star3", "") + " " + meta.get("Star4", "")
        
        # Filter based on role
        if role == "director" and person_name.lower() not in director.lower():
            continue
        elif role == "actor" and person_name.lower() not in stars.lower():
            continue
        
        movie = clean_doc(r)
        movie["director"] = director
        out.append(movie)
    
    return json.dumps({"person": person_name, "role": role, "movies": out[:15]}, ensure_ascii=False)


@tool
def filter_by_rating(payload: str):
    """Filter movies by rating range. Can specify min and max rating."""
    try:
        params = json.loads(payload)
        min_rating = float(params.get("min", 0))
        max_rating = float(params.get("max", 10))
        genre = params.get("genre", "")
        limit = int(params.get("limit", 10))
    except:
        # Parse from string like "above 8.5" or "between 7 and 9"
        if "above" in payload.lower() or "di atas" in payload.lower():
            min_rating = float(''.join(filter(lambda x: x.isdigit() or x == '.', payload)))
            max_rating = 10.0
        elif "below" in payload.lower() or "di bawah" in payload.lower():
            min_rating = 0.0
            max_rating = float(''.join(filter(lambda x: x.isdigit() or x == '.', payload)))
        else:
            min_rating = 7.0
            max_rating = 10.0
        genre = ""
        limit = 10
    
    # Build query
    query = genre if genre else "highly rated movies"
    results = qdrant.similarity_search(query, k=50)
    
    # Filter by rating
    filtered = []
    for r in results:
        movie = clean_doc(r)
        if min_rating <= movie['rating'] <= max_rating:
            filtered.append(movie)
    
    # Sort by rating descending
    filtered = sorted(filtered, key=lambda x: x['rating'], reverse=True)[:limit]
    
    out = []
    for m in filtered:
        out.append({
            "title": m['title'],
            "rating": m['rating'],
            "genre": m['genre'],
            "overview": m['overview'][:250]
        })
    
    return json.dumps({
        "min_rating": min_rating,
        "max_rating": max_rating,
        "count": len(out),
        "movies": out
    }, ensure_ascii=False)


@tool
def movies_by_decade(payload: str):
    """Get top movies from a specific decade (1970s, 1980s, 1990s, 2000s, 2010s, etc)."""
    try:
        params = json.loads(payload)
        decade = params.get("decade", payload)
        limit = int(params.get("limit", 10))
    except:
        decade = payload
        limit = 10
    
    # Extract decade number (e.g., "1990s" -> 1990, "90s" -> 1990)
    import re
    decade_match = re.search(r'(\d{2,4})', decade)
    if decade_match:
        year = decade_match.group(1)
        if len(year) == 2:
            year = "19" + year if int(year) >= 20 else "20" + year
        decade_start = int(year)
    else:
        decade_start = 2000  # default
    
    # Search movies from that decade
    query = f"movies from {decade_start}s"
    results = qdrant.similarity_search(query, k=50)
    
    # Filter by year
    filtered = []
    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        year = meta.get("Released_Year", "")
        
        try:
            year_int = int(year) if year else 0
            if decade_start <= year_int < decade_start + 10:
                movie = clean_doc(r)
                filtered.append(movie)
        except:
            continue
    
    # Sort by rating
    filtered = sorted(filtered, key=lambda x: x['rating'], reverse=True)[:limit]
    
    out = []
    for m in filtered:
        out.append({
            "title": m['title'],
            "rating": m['rating'],
            "genre": m['genre'],
            "year": m['year'],
            "overview": m['overview'][:250]
        })
    
    return json.dumps({
        "decade": f"{decade_start}s",
        "count": len(out),
        "movies": out
    }, ensure_ascii=False)


@tool
def movies_by_country(payload: str):
    """Find movies from a specific country or region (Korea, Japan, USA, India, etc)."""
    try:
        params = json.loads(payload)
        country = params.get("country", payload)
        limit = int(params.get("limit", 10))
    except:
        country = payload
        limit = 10
    
    # Country keyword mapping
    country_keywords = {
        "korea": ["korean", "korea"],
        "japan": ["japanese", "japan", "anime"],
        "india": ["indian", "bollywood", "india"],
        "usa": ["american", "hollywood", "usa"],
        "uk": ["british", "uk", "england"],
        "france": ["french", "france"],
        "china": ["chinese", "china", "hong kong"],
        "italy": ["italian", "italy"],
        "spain": ["spanish", "spain"],
        "germany": ["german", "germany"]
    }
    
    # Get search keywords
    search_key = country.lower()
    keywords = country_keywords.get(search_key, [country])
    
    # Search with country keywords
    query = f"{' '.join(keywords)} cinema movies"
    results = qdrant.similarity_search(query, k=30)
    
    out = []
    for r in results[:limit]:
        movie = clean_doc(r)
        out.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "genre": movie['genre'],
            "year": movie['year'],
            "overview": movie['overview'][:250]
        })
    
    return json.dumps({
        "country": country,
        "count": len(out),
        "movies": out
    }, ensure_ascii=False)


tools = [
    get_relevant_docs,
    recommend_movies,
    similar_movies,
    compare_movies,
    top_movies_by_genre,
    recommend_by_mood,
    search_by_person,
    filter_by_rating,
    movies_by_decade,
    movies_by_country
]


#-------------------------------------------------------
# MAIN CHAT FUNCTION (Logic Original User)
#-------------------------------------------------------

def chat_movie_expert(question, history):
    """Main function to handle movie expert chatbot."""

    system_prompt = """You are a movie expert assistant. When presenting movies:

**Standard Format:**
**[Number]. [Title]** (⭐ [Rating])
- **Genre:** [Genre]
- **Ringkasan:** [Brief overview]
---

**Comparison Format:**
| Aspek | Film 1 | Film 2 |
|-------|--------|--------|
| Rating | ⭐ X.X | ⭐ X.X |
| Genre | ... | ... |

**Rules:**
- Always use tools for movie data
- Return exactly N movies when user specifies quantity
- Use consistent markdown formatting
- Answer in user's language
- Tools support up to 20 movies per request

**Available Search Options:**
- By genre, mood, or similarity
- By actor or director name
- By rating range (e.g., "above 8.5")
- By decade (e.g., "1990s movies")
- By country (e.g., "Korean cinema")
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    # Structured prompt 
    result = agent.invoke({
        "messages": history + [{"role": "user", "content": question}]
    })

    answer = result["messages"][-1].content

    # Token usage calculation (Sesuai kode user)
    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        # Cek berbagai kemungkinan format metadata
        meta = message.response_metadata
        if "usage_metadata" in meta:
            total_input_tokens += meta["usage_metadata"].get("input_tokens", 0)
            total_output_tokens += meta["usage_metadata"].get("output_tokens", 0)
        elif "token_usage" in meta:
            total_input_tokens += meta["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += meta["token_usage"].get("completion_tokens", 0)

    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    tool_messages = []
    tool_names = []
    
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            tool_messages.append(msg.content)
            # Get tool name from the message if available
            if hasattr(msg, 'name'):
                tool_names.append(msg.name)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
        "tool_names": tool_names
    }


#-------------------------------------------------------
# STREAMLIT CHAT INTERFACE 
#-------------------------------------------------------

st.set_page_config(page_title="Movie Expert", page_icon="🎬")
st.title("🎬 Movie Expert Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize recently asked questions (max 5)
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

# Sidebar: Recently Asked
with st.sidebar:
    st.subheader("🕒 Recently Asked")
    if st.session_state.recent_queries:
        for q in reversed(st.session_state.recent_queries):
            st.markdown(f"- {q}")
    else:
        st.write("Belum ada pertanyaan.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Tanyakan tentang film..."):
    # Save to recent queries
    st.session_state.recent_queries.append(prompt)
    st.session_state.recent_queries = st.session_state.recent_queries[-5:]

    # Build history (last 20 messages)
    messages_history = st.session_state.get("messages", [])[-20:]
    
    # Convert to LangChain format
    history = []
    for msg in messages_history:
        role = "user" if msg["role"] == "Human" else "assistant"
        history.append({"role": role, "content": msg["content"]})
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Assistant response
    with st.chat_message("AI"):
        with st.spinner("🎬 Sedang memilih film terbaik untukmu... 🍿"):
            response = chat_movie_expert(prompt, history)
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

            # ------------------------------------------------------------------
            # UI UPDATE: TOOLS & METRICS
            # ------------------------------------------------------------------
            
            # 1. Menampilkan Nama Tools yang Digunakan
            if response["tool_names"]:
                unique_tools = list(set(response['tool_names']))
                st.info(f"🛠️ Tools yang digunakan: **{', '.join(unique_tools)}**")

            # 2. Estimasi Biaya & Token + Chart
            with st.expander("💰 Estimasi Biaya & Token"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Input Tokens", response["total_input_tokens"])
                col2.metric("Output Tokens", response["total_output_tokens"])
                col3.metric("Biaya (Rp)", f"{response['price']:.2f}")

                # Token Usage Chart
                import pandas as pd
                chart_data = pd.DataFrame({
                    "Tokens": ["Input Tokens", "Output Tokens"],
                    "Jumlah": [response["total_input_tokens"], response["total_output_tokens"]]
                })
                st.bar_chart(chart_data, x="Tokens", y="Jumlah")
