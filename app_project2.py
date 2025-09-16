import streamlit as st
import pandas as pd
import joblib
import os, subprocess
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


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

    return vectorizer, tfidf_matrix, cosine_similarity_matrix

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
# ===============================
# FINAL REPORT
def generate_pdf_report(df, filename="Final_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("🏨 Hotel Recommendation System - Final Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Summary
    elements.append(Paragraph("📊 Dataset Summary", styles['Heading2']))
    elements.append(Paragraph(f"• Tổng số đánh giá: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"• Số khách sạn duy nhất: {df['Hotel_Name'].nunique()}", styles['Normal']))
    elements.append(Paragraph(f"• Trung bình điểm số: {df['Score'].mean():.2f}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Basic Stats Table
    desc = df[['Score','Total_Score','Location','Cleanliness','Service','Facilities',
               'Value_for_money','Comfort_and_room_quality']].describe().round(2)

    table_data = [desc.columns.tolist()] + desc.reset_index().values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.black),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Conclusion
    elements.append(Paragraph("✅ Kết luận", styles['Heading2']))
    elements.append(Paragraph(
        "Hệ thống gợi ý khách sạn đã phân tích dữ liệu đánh giá từ khách hàng "
        "để cung cấp các gợi ý phù hợp. Báo cáo này tóm tắt đặc điểm dữ liệu, "
        "đưa ra thống kê mô tả và làm cơ sở cho các phân tích, trực quan hóa "
        "và mô hình gợi ý sau này.", styles['Normal']
    ))

    doc.build(elements)
    return filename

# ==========================
# STREAMLIT APP
# ==========================
st.set_page_config(page_title="Hotel Recommendation System", layout="wide")
# HEADER
# ------------------------
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #4e73df, #1cc88a); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 36px; margin: 0;">🏨 Hotel Recommendation System</h1>
        <p style="color: #f8f9fc; font-size: 18px; margin-top: 5px;">Tìm kiếm, phân tích & gợi ý khách sạn thông minh</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Horizontal menu bar
menu = option_menu(
    menu_title=None,  # không hiển thị tiêu đề
    options=["Business Problem", "New Prediction", "Business Insight","Final Report", "Team Info"],
    icons=["house", "bar-chart", "search", "lightbulb", "people", "file-earmark-text"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",  # menu nằm ngang
)
# Load data & models
hotel_info, hotel_comments, hotel_corpus_cosine= load_data()
vectorizer, tfidf_matrix,cosine_similarity_matrix = load_models()

# --------------------------
# BUSINESS PROBLEM
# --------------------------
if menu == "Business Problem":
    st.title("🏨 Hotel Recommendation System")
    st.write("""
    Ứng dụng này xây dựng hệ thống **gợi ý khách sạn** thông minh dựa trên dữ liệu đánh giá và mô tả khách sạn.  
    Mục tiêu là giúp khách du lịch tìm được khách sạn phù hợp nhanh chóng, đồng thời hỗ trợ doanh nghiệp nâng cao trải nghiệm khách hàng.  

    🔎 **Các phương pháp sử dụng**:
    - **Content-based Filtering**: Phân tích nội dung (TF-IDF + Cosine Similarity) để tìm khách sạn có đặc điểm tương tự.
    - **Hybrid Model**: Kết hợp thông tin khách sạn với phản hồi của khách hàng nhằm cải thiện chất lượng gợi ý.  

    💡 Với hệ thống này, người dùng có thể:
    - Tìm khách sạn theo từ khóa (ví dụ: "Da Nang", "Beach", "Resort").
    - So sánh các khách sạn theo nhiều tiêu chí.
    - Khai thác dữ liệu để hiểu rõ xu hướng và nhu cầu của khách hàng.
    """)

# --------------------------
# NEW PREDICTION / RECOMMENDATION
# --------------------------
elif menu == "New Prediction":
    st.title("🔮 New Prediction")

    option = st.selectbox("Chọn phương pháp:", ["Cosine TF-IDF"])
    
    if option in ["Cosine TF-IDF"]:
        keyword = st.text_input("Nhập từ khóa (VD: Nha Trang, Da Nang, Beach...)", "")
        if st.button("Tìm kiếm"):
            results = recommend_hotels_by_keyword(hotel_corpus_cosine, cosine_similarity_matrix, keyword, top_k=10)
            if not results.empty:
                st.dataframe(results)

# --------------------------
# BUSINESS INSIGHT
# --------------------------
elif menu == "Business Insight":
    st.title("📈 Business Insight")

    # Dropdown chọn khách sạn
    hotel_options = sorted(hotel_info["Hotel_Name"].unique())
    selected_hotel = st.selectbox("🏨 Chọn khách sạn:", [""] + hotel_options)

    # Hoặc nhập keyword / Hotel_ID
    keyword = st.text_input("🔎 Nhập từ khóa (VD: Da Nang, Beach...):")

    if st.button("Phân tích"):
        if selected_hotel:
            insights = business_insight(
                hotel_info, hotel_comments,
                keyword=selected_hotel
            )
        elif keyword:
            insights = business_insight(
                hotel_info, hotel_comments,
                keyword=keyword
            )
        elif hotel_id:
            insights = business_insight(
                hotel_info, hotel_comments,
                hotel_id=int(hotel_id)
            )
        else:
            st.warning("⚠️ Vui lòng chọn khách sạn, nhập keyword hoặc Hotel_ID.")
# --------------------------
# FINAL REPORT
# --------------------------
if menu == "Final Report":
    st.title("📑 Final Report")
    if "df" in st.session_state:
        if st.button("📑 Generate PDF Report"):
            filename = generate_pdf_report(st.session_state["df"])
            st.success(f"✅ Report generated: {filename}")
            with open(filename, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=filename)
    else:
        st.warning("⚠️ Please upload data first.")
# --------------------------
# TEAM INFO
# --------------------------
elif menu == "Team Info":
    st.title("👥 Team Info")
    st.write("""
    **Thành viên nhóm**   
    - Nguyễn Lê Ngọc Bích - ngocbich.892k1@gmail.com  
    """)
