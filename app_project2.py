import streamlit as st
import pandas as pd
import joblib
import os, subprocess
from gensim import corpora, models, similarities
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import numpy as np
import pickle
import plotly.express as px
import java_bootstrap
java_bootstrap.ensure_java()

# Đặt JAVA_HOME thủ công trước khi import pyspark
try:
    java_real_path = subprocess.check_output(
        ["readlink", "-f", "/usr/bin/java"]
    ).decode().strip()
    java_home = "/".join(java_real_path.split("/")[:-2])
    os.environ["JAVA_HOME"] = java_home
    print(f"✅ JAVA_HOME set to: {java_home}")
except Exception as e:
    print("❌ Could not set JAVA_HOME:", e)
# ==========================
# INIT SPARK
# ==========================
@st.cache_resource
def init_spark():
    return SparkSession.builder.appName("TestApp").getOrCreate()

spark = init_spark()
print("✅ Spark session created!")

# ==========================
# LOAD DATA & MODELS
# ==========================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))    
    # Load bằng pandas
    hotel_info = pd.read_csv(os.path.join(BASE_DIR, "data_clean", "hotel_info.csv"))
    hotel_comments = pd.read_csv(os.path.join(BASE_DIR, "data_clean", "hotel_comments.csv"))
    hotel_comments["Review_Date"] = pd.to_datetime(hotel_comments["Review_Date"], errors="coerce")
    hotel_corpus_cosine = pd.read_csv(os.path.join(BASE_DIR, "data_clean", "hotel_corpus_cosine.csv"))
    return hotel_info, hotel_comments, hotel_corpus_cosine
BASE_DIR = os.getcwd()

@st.cache_resource
def load_models():
    # TF-IDF (sklearn)
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
    cosine_similarity_matrix = joblib.load("models/cosine_similarity.pkl")
    # Gensim
    dictionary = corpora.Dictionary.load("models/dictionary.dict")
    corpus_gensim = list(corpora.MmCorpus("models/corpus_gensim.mm"))
    tfidf_gensim = models.TfidfModel.load("models/tfidf_gensim.model")
    similarity_index = similarities.Similarity.load("models/similarity_index.index")
    # ALS
    als_model = ALSModel.load(r"D:/7 KHOA/DL07/Project3/models/als_model") 
    return vectorizer, tfidf_matrix, dictionary, tfidf_gensim, als_model, corpus_gensim,similarity_index, cosine_similarity_matrix

# ==========================
# BUSINESS INSIGHT FUNCTIONS
# ==========================
# Hàm tìm hotel theo id hoặc key word trả ra thông tin dạng bảng 
def get_hotel_overview(hotels_df, keyword=None, hotel_id=None):
    cols = ["Hotel_ID", "Hotel_Name", "Hotel_Rank_Num", "Hotel_Address", "Total_Score","Location", "Cleanliness", "Service", "Facilities", "Value_for_money",
        "Comfort_and_room_quality", "comments_count"]
    # Truy vấn theo Hotel_ID
    if hotel_id is not None:
        result = hotels_df[hotels_df["Hotel_ID"] == hotel_id][cols]
        if result.empty:
            return f"❌ Không tìm thấy khách sạn với ID: {hotel_id}"
        return result.reset_index(drop=True)
    # Truy vấn theo keyword (hotel name)
    if keyword is not None:
        matched = hotels_df[hotels_df["Hotel_Name"].str.contains(keyword, case=False, na=False)]
        if matched.empty:
            return f"❌ Không tìm thấy khách sạn với từ khóa: {keyword}"
        return matched[cols].reset_index(drop=True)
    return "⚠️ Cần nhập ít nhất một trong hai: keyword hoặc hotel_id"

# Hàm tìm khách sạn theo ID hoặc key word trả ra biểu đồ phân tích
def analyze_strengths_weaknesses(hotels_df, keyword=None, hotel_id=None):
    # Các cột cần so sánh
    cols = ["Hotel_Rank_Num","Total_Score", "Location", "Cleanliness", "Service", "Facilities", "Value_for_money", "Comfort_and_room_quality"]
    
    # --- Tìm khách sạn ---
    if hotel_id is not None:
        hotel = hotels_df[hotels_df["Hotel_ID"] == hotel_id]
    elif keyword is not None:
        hotel = hotels_df[hotels_df["Hotel_Name"].str.contains(keyword, case=False, na=False)]
    else:
        return "⚠️ Cần nhập keyword hoặc hotel_id"
    
    if hotel.empty:
        return "❌ Không tìm thấy khách sạn"
    hotel = hotel.iloc[0]   # lấy record đầu tiên
    # --- Tính trung bình toàn hệ thống ---
    system_avg = hotels_df[cols].mean()
    # --- Điểm của khách sạn ---
    hotel_scores = hotel[cols]
    # --- Ghép dữ liệu cho vẽ ---
    compare_df = (pd.DataFrame({"Hotel": hotel_scores, "System_Avg": system_avg}).reset_index().rename(columns={"index": "Criteria"}))
    # --- Vẽ biểu đồ ---
    fig, ax = plt.subplots(figsize=(10,5))
    compare_df.plot(x="Criteria", kind="bar", ax=ax)
    ax.set_title(f"So sánh điểm khách sạn '{hotel['Hotel_Name']}' với trung bình hệ thống")
    ax.set_ylabel("Điểm")
    plt.xticks(rotation=45)
    st.pyplot(fig)  
    
    # --- Nhận xét điểm mạnh & yếu ---
    strengths = compare_df[compare_df["Hotel"] > compare_df["System_Avg"]]["Criteria"].tolist()
    weaknesses = compare_df[compare_df["Hotel"] < compare_df["System_Avg"]]["Criteria"].tolist()
    
    return {"Hotel_Name": hotel["Hotel_Name"],"Strengths": strengths,"Weaknesses": weaknesses}
# Hàm tìm theo ID hoặc key word cho chủ khách sạn, trả các biểu đồ thống kê cho khách sạn đó Quốc tịch, nhóm khách, xu hướng theo thời gian
def customer_statistics(reviews_df, keyword=None, hotel_id=None):
    # --- lọc review theo hotel ---
    if hotel_id is not None:
        data = reviews_df[reviews_df["Hotel_ID"] == hotel_id]
    elif keyword is not None:
        data = reviews_df[reviews_df["Hotel_Name"].str.contains(keyword, case=False, na=False)]
    else:
        return "⚠️ Cần nhập keyword hoặc hotel_id"
    if data.empty:
        return "❌ Không có review cho khách sạn này"
    hotel_name = data["Hotel_Name"].iloc[0]
    print(f"📊 Thống kê khách hàng cho khách sạn: {hotel_name}\n")
    
    # --- Quốc tịch ---
    nationality_count = data["Nationality"].value_counts().head(10).reset_index()
    nationality_count.columns = ["Nationality", "Count"]
    fig1 = px.bar(
        nationality_count,
        x="Nationality",
        y="Count",
        labels={"Nationality": "Quốc tịch", "Count": "Số lượng khách hàng"},
        title="Top 10 quốc tịch khách hàng"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- Nhóm khách ---
    group_count = data["Group_Name"].value_counts().reset_index()
    group_count.columns = ["Group_Name", "Count"]
    fig2 = px.bar(
        group_count,
        x="Group_Name",
        y="Count",
        labels={"Group_Name": "Nhóm khách", "Count": "Số lượng khách hàng"},
        title="Phân bố nhóm khách"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- Xu hướng theo thời gian ---
    trend = data.groupby(data["Review_Date"].dt.to_period("M")).size()
    trend.index = trend.index.to_timestamp()
    fig3 = px.line(
        x=trend.index,
        y=trend.values,
        labels={"x": "Thời gian", "y": "Số lượng khách hàng"},
        title="Xu hướng review theo thời gian"
    )
    st.plotly_chart(fig3, use_container_width=True) 

    # --- Phân bố số ngày ở ---
    days_dist = data["Days"].value_counts().sort_index()
    fig4 = px.bar(
        x=days_dist.index,
        y=days_dist.values,
        labels={"x": "Số ngày ở", "y": "Số lượng khách hàng"},
        title="Phân bố số ngày khách ở (Days)"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # --- Xu hướng theo tháng ---
    month_dist = data["Month_Stay"].value_counts().sort_index()
    full_months = pd.Series(0, index=np.arange(1, 13))
    month_dist = full_months.add(month_dist, fill_value=0).astype(int)

    fig5 = px.line(
        x=month_dist.index,
        y=month_dist.values,
        labels={"x": "Tháng", "y": "Số lượng khách"},
        title="Xu hướng khách ở theo tháng"
    )
    # Hiện đầy đủ tháng 1-12
    fig5.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=[str(i) for i in range(1, 13)]
    )
    fig5.update_traces(
        mode="lines+markers+text",
        text=month_dist.values,
        textposition="top center"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # --- Room type ---
    room_dist = data["Room_Type"].value_counts().head(5)
    fig6 = px.pie(
        values=room_dist.values,
        names=room_dist.index,
        title="Tỷ lệ top 5 loại phòng được đặt",
        hole=0.3
    )
    st.plotly_chart(fig6, use_container_width=True)

    # --- Điểm đánh giá ---
    fig7 = px.histogram(
        data, 
        x="Score", 
        nbins=10, 
        title="Phân bổ điểm đánh giá (Score)",
        labels={"Score": "Điểm khách hàng chấm", "count": "Số lượng khách hàng"}
    )
    hotel_score = data["Mean_Reviewer_Score"].mean()
    fig7.add_vline(
        x=hotel_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Hotel Mean = {hotel_score:.2f}",
        annotation_position="top right"
    )
    st.plotly_chart(fig7, use_container_width=True)

# Hàm tìm word_cloud
def hotel_wordcloud(df, keyword=None, hotel_id=None, body_col='Body_clean', hotel_name_col='Hotel_Name', hotel_id_col='Hotel_ID'):
    # Lọc dữ liệu theo hotel_id hoặc keyword
    if hotel_id is not None:
        hotel_df = df[df[hotel_id_col] == hotel_id]
    elif keyword is not None:
        hotel_df = df[df[hotel_name_col].str.contains(keyword, case=False, na=False)]
    else:
        print("❌ Bạn cần nhập keyword hoặc hotel_id")
        return
    if hotel_df.empty:
        print("❌ Không tìm thấy khách sạn phù hợp")
        return
    
    # Ghép toàn bộ review body lại
    text = " ".join(hotel_df[body_col].dropna().astype(str).tolist())
    if not text.strip():
        print("❌ Không có review text để tạo wordcloud")
        return
    
    # Tạo wordcloud
    wc = WordCloud(width=800, height=400, background_color="white", max_words=200, collocations=False).generate(text)
    
    # Vẽ
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    title = hotel_df[hotel_name_col].iloc[0] if not hotel_df.empty else "Hotel"
    ax.set_title(f"WordCloud - {title}", fontsize=16)
    st.pyplot(fig)

# BUSINESS INSIGHT WRAPPER
# ==========================
def business_insight(hotel_info, hotel_comments, keyword=None, hotel_id=None):
    """Tổng hợp insight cho 1 khách sạn"""
    # 1. Tổng quan khách sạn
    st.subheader("📋 Tổng quan khách sạn")
    overview = get_hotel_overview(hotel_info, keyword, hotel_id)

    # 2. Điểm mạnh & điểm yếu
    st.subheader("💡 Điểm mạnh & yếu")
    strengths_weaknesses = analyze_strengths_weaknesses(hotel_info, keyword, hotel_id)

    # 3. Thống kê khách hàng
    st.subheader("👥 Thống kê khách hàng")
    customer_stats = customer_statistics(hotel_comments, keyword, hotel_id)

    # 4. Wordcloud review
    st.subheader("☁️ WordCloud Review")
    wordcloud = hotel_wordcloud(hotel_comments, keyword, hotel_id)

    return {"Overview": overview,"Strengths_Weaknesses": strengths_weaknesses,"Customer_Statistics": customer_stats,"Word Cloud": wordcloud}
# ==========================
# RECOMMENDATION FUNCTIONS
# ==========================
# cosine
def recommend_hotels_by_keyword(hotel_corpus, cosine_similarity_matrix, keyword, top_k=5):
    hotel_corpus = hotel_corpus.reset_index(drop=True)
    # Tìm khách sạn theo keyword (chứa trong tên)
    matches = hotel_corpus[hotel_corpus["Hotel_Name"].str.contains(keyword, case=False, na=False)]
    if matches.empty:
        print(f"❌ Không tìm thấy khách sạn nào chứa từ khóa '{keyword}'")
        return pd.DataFrame()
    all_results = []
    for idx in matches.index:
        src_id = hotel_corpus.loc[idx, "Hotel_ID"]
        src_name = hotel_corpus.loc[idx, "Hotel_Name"]
        # Tính similarity
        sim_scores = list(enumerate(cosine_similarity_matrix[idx, :]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Loại chính nó
        sim_scores = [(i, score) for i, score in sim_scores if i != idx]
        # Lấy top-k và tránh trùng hotel_id
        seen_ids = set()
        count = 0
        for i, score in sim_scores:
            hid = hotel_corpus.loc[i, "Hotel_ID"]
            if hid not in seen_ids:
                seen_ids.add(hid)
                all_results.append({
                    "Source_Hotel_ID": src_id,
                    "Source_Hotel_Name": src_name,
                    "Recommended_Hotel_ID": hid,
                    "Recommended_Hotel_Name": hotel_corpus.loc[i, "Hotel_Name"],
                    "Recommended_Hotel_Address": hotel_corpus.loc[i, "Hotel_Address"],
                    "Recommended_Hotel_Description": hotel_corpus.loc[i, "Hotel_Description"],
                    "Similarity": round(score, 3)
                })
    # Chuyển sang DataFrame
    df = pd.DataFrame(all_results)
    if df.empty:
        return df
    # Giữ lại top 10 khách sạn không trùng Recommended_Hotel_ID
    df = df.sort_values("Similarity", ascending=False)
    df = df.drop_duplicates(subset=["Recommended_Hotel_ID"], keep="first")
    df = df.head(top_k)
    return df.reset_index(drop=True)

# gensim
def find_hotels_by_keyword(hotel_corpus2, keyword):
    matches = hotel_corpus2[hotel_corpus2["Hotel_Name"].str.contains(keyword, case=False, na=False)]
    if matches.empty:
        print(f"❌ Không tìm thấy khách sạn nào chứa từ khóa '{keyword}'")
        return pd.DataFrame()
    return matches
def get_topk_recommendations(hotel_corpus2, matches, corpus_gensim, tfidf, similarity_index, top_k=5):
    results = []
    for _, row in matches.iterrows():
        corpus_pos = row.name
        query_bow = corpus_gensim[corpus_pos]
        sims = similarity_index[tfidf[query_bow]]
        # Sắp xếp similarity, loại chính nó
        sims_sorted = sorted(list(enumerate(sims)), key=lambda x: -x[1])
        topk = [(i, score) for i, score in sims_sorted if i != corpus_pos][:top_k]
        for i, score in topk:
            # Lấy thông tin khách sạn recommended
            hotel_info = hotel_corpus2.iloc[i]
            results.append({
                "Source_Hotel": row["Hotel_Name"],
                "Recommended_Hotel": hotel_info["Hotel_Name"],
                "Address": hotel_info.get("Hotel_Address", ""),
                "Description": hotel_info.get("Hotel_Description", ""),
                "Score": score
            })
    return pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df

# ALS
def recommend_hotels_by_ALS(als_model, hotel_info_pyspark, nationality_id, top_k=10):
    # Lấy danh sách khách sạn (distinct)
    hotels_df = hotel_info_pyspark.select( "Hotel_ID", "Hotel_Name", "Hotel_Address", "hotel_numeric_id").distinct()
    
    # Gắn user_id vào để predict
    user_df = hotels_df.withColumn("nationality_id", F.lit(nationality_id))
    
    # Dự đoán bằng ALS
    predictions = als_model.transform(user_df)
    
    # Lọc prediction != null, lấy top-k
    result = predictions.filter(col("prediction").isNotNull()) \
                        .orderBy(col("prediction").desc()) \
                        .limit(top_k)
    
    return result.select("Hotel_Name", "Hotel_Address", "prediction")

# ==========================
# STREAMLIT APP
# ==========================
st.set_page_config(page_title="Hotel Recommendation System", layout="wide")

# Sidebar menu
menu = st.sidebar.radio(
    "Menu",
    ["Business Problem", "Evaluation & Report", "New Prediction/Analysis/Recommendation","Business Insight", "Team Info"]
)

# Load data & models
hotel_info, hotel_comments, hotel_corpus_cosine= load_data()
vectorizer, tfidf_matrix, dictionary, tfidf_gensim, als_model, corpus_gensim,similarity_index, cosine_similarity_matrix = load_models()
hotel_info_pyspark = spark.read.parquet(os.path.join(BASE_DIR, "data_clean", "hotel_info_pyspark.parquet"))

# --------------------------
# BUSINESS PROBLEM
# --------------------------
if menu == "Business Problem":
    st.title("🏨 Business Problem")
    st.write("""
    Hệ thống gợi ý khách sạn dựa trên dữ liệu đánh giá của khách hàng.
    
    - **Content-based Filtering**: TF-IDF + Cosine Similarity (sklearn, Gensim)  
    - **Collaborative Filtering**: ALS (Spark ML)  
    - **Hybrid Model**: Kết hợp thông tin khách sạn và phản hồi của khách  
    """)

# --------------------------
# EVALUATION & REPORT
# --------------------------
elif menu == "Evaluation & Report":
    st.title("📊 Evaluation & Report")
    st.write("So sánh RMSE giữa ALS và Content-based filtering:")

    # Demo số liệu RMSE
    rmse_als = 0.97
    rmse_content = 0.92

    st.metric("RMSE ALS", rmse_als)
    st.metric("RMSE Content-based", rmse_content)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(x=["ALS", "Content-based"], y=[rmse_als, rmse_content], ax=ax)
    ax.set_title("So sánh RMSE")
    st.pyplot(fig)

# --------------------------
# NEW PREDICTION / RECOMMENDATION
# --------------------------
elif menu == "New Prediction/Analysis/Recommendation":
    st.title("🔮 New Prediction / Analysis / Recommendation")

    option = st.selectbox("Chọn phương pháp:", ["Cosine TF-IDF", "Gensim", "ALS"])
    
    if option in ["Cosine TF-IDF", "Gensim"]:
        keyword = st.text_input("Nhập từ khóa (VD: Nha Trang, Da Nang, Beach...)", "")
        if st.button("Tìm kiếm"):
            if option == "Cosine TF-IDF":
                results = recommend_hotels_by_keyword(hotel_corpus_cosine, cosine_similarity_matrix, keyword, top_k=10)
                st.dataframe(results)
            elif option == "Gensim":
                st.write("⚡ Gensim")
                matches = find_hotels_by_keyword(hotel_corpus_cosine, keyword)
                if not matches.empty:
                    results = get_topk_recommendations(hotel_corpus_cosine,matches,corpus_gensim,tfidf_gensim,similarity_index,top_k=5)
                    st.dataframe(results)
                else:
                    st.warning(f"❌ Không tìm thấy khách sạn nào chứa từ khóa '{keyword}'")

    elif option == "ALS":
        nationality_id = st.number_input("Nhập Nationality_ID:", min_value=1, step=1)
        if st.button("Gợi ý khách sạn"):
            results = recommend_hotels_by_ALS(
                als_model, hotel_info_pyspark, nationality_id, top_k=10)
            if results.head(1):  # head(1) trả về danh sách rỗng nếu trống
                st.dataframe(results.toPandas())  # chuyển sang Pandas để hiển thị
            else:
                st.warning("Không tìm thấy gợi ý cho user này.")
# --------------------------
# BUSINESS INSIGHT
# --------------------------
elif menu == "Business Insight":
    st.title("📈 Business Insight")
    keyword = st.text_input("Nhập tên khách sạn hoặc từ khóa:")
    hotel_id = st.text_input("Hoặc nhập Hotel_ID:")

    if st.button("Phân tích"):
        insights = business_insight(hotel_info, hotel_comments, keyword=keyword if keyword else None,
                                    hotel_id=int(hotel_id) if hotel_id else None)
# --------------------------
# TEAM INFO
# --------------------------
elif menu == "Team Info":
    st.title("👥 Team Info")
    st.write("""
    **Thành viên nhóm**   
    - Nguyễn Lê Ngọc Bích - ngocbich.892k1@gmail.com  
    """)
