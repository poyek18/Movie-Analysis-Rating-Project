import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import MinMaxScaler

# === Load & Clean Data ===
df = pd.read_csv("CombinedRatings.csv")

required_columns = ['movie_title', 'year', 'total_popularity', 'combined_rating', 'genre']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

df['movie_title'] = df['movie_title'].astype(str).str.strip().str.title()
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'total_popularity', 'combined_rating'])
df['year'] = df['year'].astype(int)

df['genre'] = df['genre'].astype(str).str.split(',')
df = df.explode('genre')
df['genre'] = df['genre'].str.strip().str.title()

scaler = MinMaxScaler()
df['total_popularity_norm'] = scaler.fit_transform(df[['total_popularity']])

# === App Title ===
st.title("🎬 Movie Ratings Dashboard ")

# === Sidebar Filters ===
st.sidebar.header("📊 Filters")

all_genres = sorted(df['genre'].dropna().unique())
select_all = st.sidebar.checkbox("✅ Select All Genres", value=True)

selected_genres = st.sidebar.multiselect(
    "Choose Genre",
    all_genres,
    default=all_genres if select_all else []
)

min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("🎞️ Year Range", min_year, max_year, (min_year, max_year))

min_popularity = st.sidebar.slider("🔥 Minimum Popularity", 0, int(df['total_popularity'].max()), 0)

search = st.sidebar.text_input("🔍 Search Movie Title")

# 💡 Enhanced sort options
sort_by = st.sidebar.selectbox("📈 Sort By", [
    "Popularity (High to Low)",
    "Popularity (Low to High)",
    "Year (Newest First)",
    "Year (Oldest First)",
    "Title (A-Z)",
    "Title (Z-A)"
])

# === Apply Filters ===
filtered_df = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['total_popularity'] >= min_popularity)
]

if selected_genres:
    filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]

if search:
    filtered_df = filtered_df[filtered_df['movie_title'].str.contains(search, case=False)]

# === Apply Sorting ===
if sort_by == "Popularity (High to Low)":
    filtered_df = filtered_df.sort_values(by="total_popularity", ascending=False)
elif sort_by == "Popularity (Low to High)":
    filtered_df = filtered_df.sort_values(by="total_popularity", ascending=True)
elif sort_by == "Year (Newest First)":
    filtered_df = filtered_df.sort_values(by="year", ascending=False)
elif sort_by == "Year (Oldest First)":
    filtered_df = filtered_df.sort_values(by="year", ascending=True)
elif sort_by == "Title (A-Z)":
    filtered_df = filtered_df.sort_values(by="movie_title", ascending=True)
elif sort_by == "Title (Z-A)":
    filtered_df = filtered_df.sort_values(by="movie_title", ascending=False)

# ✅ Remove duplicate movies caused by multiple genres
filtered_df = filtered_df.drop_duplicates(subset=["movie_title", "year"])

# === Display Filtered Movies ===
st.subheader(f"🎥 {len(filtered_df)} Movies Matched")
st.dataframe(filtered_df[["movie_title", "year", "genre", "total_popularity", "combined_rating"]])

# === Top N Movies by Popularity ===
top_n = st.slider("🏆 Show Top N Movies", 3, 30, 10)
top_movies = filtered_df.head(top_n)

if not top_movies.empty:
    st.subheader("💯 Top Movies by Popularity")
    top_chart = alt.Chart(top_movies).mark_bar().encode(
        x=alt.X("total_popularity:Q", title="Total Popularity"),
        y=alt.Y("movie_title:N", sort='-x', title="Movie Title"),
        tooltip=["movie_title", "total_popularity", "combined_rating"]
    ).properties(height=400)
    st.altair_chart(top_chart, use_container_width=True)
else:
    st.warning("👀 No movies hit those filters... might wanna widen the net.")

# === Most Popular Genres ===
if not filtered_df.empty:
    st.subheader("📚 Genre Popularity")
    genre_popularity = (
        filtered_df.groupby("genre")["total_popularity"]
        .sum().sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    genre_chart = alt.Chart(genre_popularity).mark_bar().encode(
        x=alt.X("total_popularity:Q", title="Total Popularity"),
        y=alt.Y("genre:N", sort='-x', title="Genre"),
        tooltip=["genre", "total_popularity"]
    ).properties(height=400)
    st.altair_chart(genre_chart, use_container_width=True)

# === Average Rating by Genre (Filtered) ===
if not filtered_df.empty:
    st.subheader("⭐ Average Rating by Genre ")

    avg_rating_by_genre = (
        filtered_df.groupby("genre")["combined_rating"]
        .mean()
        .reset_index()
        .sort_values(by="combined_rating", ascending=False)
    )

    rating_chart = alt.Chart(avg_rating_by_genre).mark_bar().encode(
        x=alt.X("combined_rating:Q", title="Average Rating"),
        y=alt.Y("genre:N", sort='-x', title="Genre"),
        tooltip=["genre", "combined_rating"]
    ).properties(height=400)

    st.altair_chart(rating_chart, use_container_width=True)

# === Movie Comparison Mode ===
st.subheader("🎯 Compare Your Picks")
compare_options = filtered_df['movie_title'].unique()
selected_movies = st.multiselect("Select movies to compare (up to 5)", compare_options)

if selected_movies:
    compare_df = filtered_df[filtered_df['movie_title'].isin(selected_movies)].drop_duplicates(subset=['movie_title'])

    if len(compare_df) > 5:
        st.warning("⚠️ Max 5 movies at a time, no cap.")
    else:
        st.dataframe(compare_df[["movie_title", "year", "genre", "total_popularity", "combined_rating"]].set_index("movie_title"))

        # === Popularity Pie Chart ===
        st.subheader("🥧 Popularity Breakdown (Pie Chart)")
        popularity_pie = alt.Chart(compare_df).mark_arc().encode(
            theta=alt.Theta(field="total_popularity", type="quantitative"),
            color=alt.Color(field="movie_title", type="nominal", title="Movie"),
            tooltip=["movie_title", "total_popularity"]
        ).properties(height=400)
        st.altair_chart(popularity_pie, use_container_width=True)

        # === Rating Pie Chart ===
        st.subheader("⭐ Rating Breakdown (Pie Chart)")
        rating_pie = alt.Chart(compare_df).mark_arc().encode(
            theta=alt.Theta(field="combined_rating", type="quantitative"),
            color=alt.Color(field="movie_title", type="nominal", title="Movie"),
            tooltip=["movie_title", "combined_rating"]
        ).properties(height=400)
        st.altair_chart(rating_pie, use_container_width=True)
