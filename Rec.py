import streamlit as st
import pandas as pd
import numpy as np
import ast
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Система рекомендации фильмов",
    page_icon="🎬",
    layout="wide"
)

# Кастомный CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .movie-card {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 1px solid #2d2d44;
    }
    .movie-title {
        color: #f8f9fa;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    .movie-meta {
        color: #adb5bd;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .movie-overview {
        color: #dee2e6;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .badge {
        background-color: #6c757d;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0 5px 5px 0;
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        padding: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Заголовок
st.markdown("""
    <div style="background-color:#1a1a2e;padding:25px;border-radius:10px;margin-bottom:30px;">
        <h1 style="color:#f8f9fa;text-align:center;margin:0;">🎬 Система рекомендации фильмов</h1>
        <p style="color:#adb5bd;text-align:center;margin:10px 0 0 0;">Популярные • По жанру • Контентные • Коллаборативные</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Загрузка и обработка данных"""
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)
    movies = parse_json_column(movies, "genres")
    
    ratings = pd.read_csv("ratings.csv")
    merged_df = ratings.merge(movies, left_on="movieId", right_on="id", how="inner")
    
    return movies, ratings, merged_df

def parse_json_column(df, column_name):
    """Парсинг JSON-колонок"""
    def parse_func(txt):
        if pd.isna(txt):
            return []
        try:
            return ast.literal_eval(txt)
        except:
            return []
    if column_name in df.columns:
        new_col = column_name + "_parsed"
        df[new_col] = df[column_name].apply(parse_func)
    return df

BASE_POSTER_URL = "https://image.tmdb.org/t/p/w185"

def get_poster_url(row):
    """Генерация URL постера"""
    if "poster_path" not in row or pd.isna(row["poster_path"]):
        return "https://via.placeholder.com/185x278?text=No+Poster"
    path = row["poster_path"]
    return BASE_POSTER_URL + path if isinstance(path, str) and path.startswith("/") else "https://via.placeholder.com/185x278?text=No+Poster"

# 3.1 Популярные фильмы
def get_popular_movies(movies_df, n=10, min_votes=500):
    df = movies_df.copy()
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    
    qualified = df[df["vote_count"] >= min_votes].copy()
    if qualified.empty:
        return qualified
    
    C = qualified["vote_average"].mean()
    m = qualified["vote_count"].quantile(0.9)
    qualified["weighted_rating"] = (
        (qualified["vote_average"] * qualified["vote_count"]) + (C * m)
    ) / (qualified["vote_count"] + m)
    
    return qualified.sort_values("weighted_rating", ascending=False).head(n)

# 3.2 По жанру
def get_movies_by_genre(movies_df, genre, n=10):
    if "genres_parsed" not in movies_df.columns:
        return pd.DataFrame()
    
    df = movies_df.copy()
    filtered = df[df["genres_parsed"].apply(
        lambda glist: any(g.get("name","") == genre for g in glist)
    )]
    if filtered.empty:
        return filtered
    
    C = filtered["vote_average"].mean()
    m = filtered["vote_count"].quantile(0.7)
    filtered["weighted_rating"] = (
        (filtered["vote_average"] * filtered["vote_count"]) + (C * m)
    ) / (filtered["vote_count"] + m)
    
    return filtered.sort_values("weighted_rating", ascending=False).head(n)

# 3.3 Контентная (описание)
def build_tfidf_cosine(movies_df):
    df = movies_df.copy()
    df["overview"] = df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, cosine_sim, indices

def get_recommendations_by_overview(movies_df, title, n=10):
    df, cosine_sim, indices = build_tfidf_cosine(movies_df)
    
    if title not in indices:
        possible = df[df["title"].str.contains(title, case=False, na=False)]
        if possible.empty:
            return pd.DataFrame()
        title = possible["title"].iloc[0]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [s[0] for s in sim_scores]
    
    result = df.iloc[movie_indices].copy()
    result["similarity_score"] = [s[1] for s in sim_scores]
    return result

# 3.4 Контентная (жанры)
def build_soup_cosine(movies_df):
    df = movies_df.copy()
    def get_genres_str(glist):
        return " ".join([str(g.get("name", "")).replace(" ", "").lower() for g in glist])
    
    df["genres_str"] = df["genres_parsed"].apply(get_genres_str)
    
    count = CountVectorizer(stop_words="english")
    matrix = count.fit_transform(df["genres_str"])
    cosim = cosine_similarity(matrix, matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, cosim, indices

def get_recommendations_by_soup(movies_df, title, n=10):
    df, cosim, indices = build_soup_cosine(movies_df)
    
    if title not in indices:
        possible = df[df["title"].str.contains(title, case=False, na=False)]
        if possible.empty:
            return pd.DataFrame()
        title = possible["title"].iloc[0]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    movie_indices = [s[0] for s in sim_scores]
    result = df.iloc[movie_indices].copy()
    result["similarity_score"] = [s[1] for s in sim_scores]
    return result

# 3.5 Коллаборативная (SVD)
def train_svd_model(merged_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(merged_df[["userId","movieId","rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=20, random_state=42)
    algo.fit(trainset)
    return algo

def get_svd_recommendations(algo, merged_df, user_id, n=10):
    if user_id not in merged_df["userId"].unique():
        return pd.DataFrame()
    rated_movies = merged_df[merged_df["userId"]==user_id]["movieId"].unique()
    all_ids = merged_df["movieId"].unique()
    to_predict = [m for m in all_ids if m not in rated_movies]
    
    preds = [(mid, algo.predict(user_id, mid).est) for mid in to_predict]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_ids = [p[0] for p in preds[:n]]
    
    subset = merged_df[merged_df["movieId"].isin(top_ids)].drop_duplicates("movieId")
    results = []
    for mid in top_ids:
        row = subset[subset["movieId"]==mid]
        if not row.empty:
            r = row.iloc[0].copy()
            r["predicted_rating"] = round([p[1] for p in preds if p[0]==mid][0], 2)
            results.append(r)
    return pd.DataFrame(results)

def display_movie_card(row, show_similarity=False, show_predicted=False):
    """Отображение карточки фильма"""
    with st.container():
        st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(get_poster_url(row), width=185)
        
        with col2:
            # Заголовок и метаданные
            st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
            
            meta = []
            if pd.notna(row.get("release_date")):
                meta.append(f'<span class="badge">🗓️ {str(row["release_date"])[:4]}</span>')
            if pd.notna(row.get("vote_average")):
                meta.append(f'<span class="badge">⭐ {row["vote_average"]:.1f}</span>')
            if pd.notna(row.get("vote_count")):
                meta.append(f'<span class="badge">👥 {row["vote_count"]}</span>')
            if pd.notna(row.get("weighted_rating")):
                meta.append(f'<span class="badge">📊 {row["weighted_rating"]:.2f}</span>')
            if show_similarity and pd.notna(row.get("similarity_score")):
                meta.append(f'<span class="badge" style="background:#17a2b8;">🎯 {row["similarity_score"]:.2f}</span>')
            if show_predicted and pd.notna(row.get("predicted_rating")):
                meta.append(f'<span class="badge" style="background:#28a745;">🔮 {row["predicted_rating"]}</span>')
            
            st.markdown(f'<div class="movie-meta">{"".join(meta)}</div>', unsafe_allow_html=True)
            
            # Описание
            overview = row.get("overview", "Нет описания")
            st.markdown(f'<div class="movie-overview">{overview}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Загрузка данных
with st.spinner("Загрузка данных..."):
    movies_metadata, ratings, joined_df = load_data()

# Сайдбар
with st.sidebar:
    st.title("⚙️ Настройки")
    st.markdown("---")
    method = st.selectbox("**Алгоритм рекомендаций**", [
        "Популярные фильмы",
        "По жанру",
        "Контентная (по описанию)",
        "Контентная (по жанрам)",
        "Персональные (SVD)"
    ])
    num_recs = st.slider("**Количество рекомендаций**", 5, 20, 10)
    st.markdown("---")
    st.markdown("""
    **Описание алгоритмов:**
    - 🎖️ Популярные: Рейтинг с учетом количества оценок
    - 🎭 По жанру: Лучшие фильмы выбранного жанра
    - 📝 По описанию: Анализ текста описания
    - 🍲 По жанрам: Комбинации жанров
    - 👥 SVD: Коллаборативная фильтрация
    """)

# Основной интерфейс
if method == "Популярные фильмы":
    st.subheader("🎖️ Популярные фильмы")
    min_votes = st.slider("Минимальное количество оценок", 100, 5000, 500, 100)
    if st.button("Показать"):
        with st.spinner("Поиск популярных фильмов..."):
            results = get_popular_movies(movies_metadata, num_recs, min_votes)
        if not results.empty:
            st.success(f"Найдено {len(results)} фильмов")
            for _, row in results.iterrows():
                display_movie_card(row)
        else:
            st.warning("Фильмы не найдены")

elif method == "По жанру":
    st.subheader("🎭 Рекомендации по жанру")
    genres = sorted(set(g["name"] for glist in movies_metadata["genres_parsed"] for g in glist))
    genre = st.selectbox("Выберите жанр", genres)
    if st.button("Найти"):
        with st.spinner(f"Ищем {genre} фильмы..."):
            results = get_movies_by_genre(movies_metadata, genre, num_recs)
        if not results.empty:
            st.success(f"Найдено {len(results)} {genre} фильмов")
            for _, row in results.iterrows():
                display_movie_card(row)
        else:
            st.warning("Фильмы не найдены")

elif method == "Контентная (по описанию)":
    st.subheader("📝 Рекомендации по описанию")
    titles = movies_metadata["title"].dropna().unique()
    title = st.selectbox("Выберите фильм", titles)
    if st.button("Найти похожие"):
        with st.spinner("Анализ описаний..."):
            results = get_recommendations_by_overview(movies_metadata, title, num_recs)
        if not results.empty:
            st.success(f"Найдено {len(results)} похожих фильмов")
            for _, row in results.iterrows():
                display_movie_card(row, show_similarity=True)
        else:
            st.warning("Фильмы не найдены")

elif method == "Контентная (по жанрам)":
    st.subheader("🍲 Рекомендации по жанрам")
    titles = movies_metadata["title"].dropna().unique()
    title = st.selectbox("Выберите фильм", titles)
    if st.button("Найти похожие"):
        with st.spinner("Анализ жанров..."):
            results = get_recommendations_by_soup(movies_metadata, title, num_recs)
        if not results.empty:
            st.success(f"Найдено {len(results)} похожих фильмов")
            for _, row in results.iterrows():
                display_movie_card(row, show_similarity=True)
        else:
            st.warning("Фильмы не найдены")

elif method == "Персональные (SVD)":
    st.subheader("👥 Персональные рекомендации")
    users = joined_df["userId"].unique()[:1000]
    user = st.selectbox("Выберите ID пользователя", users)
    if st.button("Сгенерировать"):
        with st.spinner("Обучение модели..."):
            algo = train_svd_model(joined_df)
            results = get_svd_recommendations(algo, joined_df, user, num_recs)
        if not results.empty:
            st.success(f"Рекомендации для пользователя {user}")
            for _, row in results.iterrows():
                display_movie_card(row, show_predicted=True)
        else:
            st.warning("Недостаточно данных")

# Футер
st.markdown("---")
st.markdown("""
    <div style="text-align:center;color:#6c757d;font-size:0.9rem;padding:20px;">
        Система рекомендации фильмов • Данные: TMDB • Алгоритмы: Popularity, TF-IDF, SVD
    </div>
""", unsafe_allow_html=True)