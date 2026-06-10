# ==============================================================================
# [웹 배포용 프로그램 - app.py]
# 주제: 코사인 유사도 추천 시스템의 필터버블 제어 웹 시뮬레이터
# 프레임워크: Streamlit (웹 UI 자동 빌드 및 클라우드 배포용)
# 최종 보정: st.separator() 오류를 공식 명칭인 st.divider()로 수정 완료
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. 웹페이지 상단 레이아웃 및 설명글 설정
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Filter Bubble Simulator", layout="wide")

st.title("🎥 AI Recommendation & Filter Bubble Simulator")
st.markdown("""
이 시뮬레이터는 기하 과목의 '코사인 유사도(벡터의 내적)'와 확률과 통계의 '가중평균' 개념을 활용한 인공지능 추천 시스템입니다.
평소 선호하는 장르 점수를 입력하면, 기존 알고리즘이 유도하는 필터버블(편향) 현상과 이를 기술적으로 보정한 완화 알고리즘(다양성 확보)의 결과를 실시간 그래프로 대조해 볼 수 있습니다.
""")

st.divider()

# ------------------------------------------------------------------------------
# 2. 웹 왼쪽 사이드바: 사용자 데이터 및 10대 장르 평점 입력 UI 구축
# ------------------------------------------------------------------------------
st.sidebar.header("👤 User Information & Ratings")

# 텍스트 입력창 (영문 이름 입력 안내)
user_name = st.sidebar.text_input("Enter your English Name", "").strip()
if not user_name:
    user_name = "User"

genres = ["SF", "Horror", "Romance", "Comedy", "Documentary", "Humanities", "Arts", "Action", "Thriller", "Fantasy"]

st.sidebar.subheader("📌 Rate 10 Genres (0 ~ 5)")
st.sidebar.caption("5점: 최선호 장르 / 0점: 소비하지 않는 장르")

# 마우스 슬라이더 바로 점수 입력받기
genre_ratings = {}
for g in genres:
    genre_ratings[g] = st.sidebar.slider(f"[{g}] Score", min_value=0, max_value=5, value=0)

# ------------------------------------------------------------------------------
# 3. 데이터 행렬 구성 및 알고리즘 연산부 (수학 모델 적용)
# ------------------------------------------------------------------------------
content_names = [
    "Interstellar (SF)", "The Conjuring (Horror)", "About Time (Romance)", 
    "Extreme Job (Comedy)", "Cosmos (Documentary)", "Sapiens (Humanities)", 
    "Van Gogh (Arts)", "Mad Max (Action)", "Parasite (Thriller)", "Harry Potter (Fantasy)",
    "The Martian (SF)*", "A Quiet Place (Horror)*", "La La Land (Romance)*", 
    "Hangover (Comedy)*", "My Octopus Teacher (Doc)*", "Guns Germs Steel (Hum)*", 
    "Monet Exhibition (Arts)*", "John Wick (Action)*", "Inception (Thriller)*", "Lord of Rings (Fantasy)*"
]

categories = genres + genres 
target_ratings = np.array([genre_ratings[g] for g in genres] + [0] * 10)

# 가상 유저 데이터베이스 (User A~D)
ratings_matrix = np.array([
    target_ratings,
    [5, 1, 2, 4, 1, 1, 5, 5, 4, 3, 5, 2, 1, 3, 1, 2, 5, 5, 4, 4],  # User A
    [5, 1, 3, 2, 5, 5, 5, 1, 3, 2, 5, 2, 4, 1, 5, 4, 5, 1, 3, 2],  # User B
    [3, 5, 1, 2, 2, 2, 1, 4, 5, 5, 2, 5, 2, 1, 3, 1, 2, 5, 4, 5],  # User C
    [2, 1, 5, 5, 3, 2, 3, 2, 2, 4, 1, 1, 5, 4, 3, 2, 3, 3, 2, 4]   # User D
])

# 코사인 유사도 연산 함수 (벡터의 내적 활용)
def get_cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1[:10]) * np.linalg.norm(v2[:10])
    return np.dot(v1[:10], v2[:10]) / norm if norm > 0 else 0.0

user_sim = [get_cosine_similarity(ratings_matrix[0], ratings_matrix[i]) for i in range(5)]

# 아직 보지 않은 신작 후보군(인덱스 10~19) 협업 필터링 연산
unseen_indices = np.arange(10, 20)
base_scores = {}
for item_idx in unseen_indices:
    weighted_sum = sum(user_sim[i] * ratings_matrix[i, item_idx] for i in range(1, 5) if ratings_matrix[i, item_idx] > 0)
    sim_sum = sum(user_sim[i] for i in range(1, 5) if ratings_matrix[i, item_idx] > 0)
    base_scores[item_idx] = weighted_sum / sim_sum if sim_sum > 0 else 0.0

# 다양성 페널티 연산 (필터버블 차단 제어 수식)
lambda_param = 1.5
favorite_genres = [g for g, score in genre_ratings.items() if score >= 4]
total_score_sum = sum(genre_ratings.values())
bias_ratio = sum(genre_ratings[g] for g in favorite_genres) / total_score_sum if total_score_sum > 0 else 0.0

enhanced_scores = {}
penalties = {}
for item_idx in unseen_indices:
    genre = categories[item_idx]
    if genre in favorite_genres:
        penalty = (genre_ratings[genre] / 5.0) * bias_ratio * 1.5
    else:
        penalty = 0.0
    penalties[item_idx] = penalty
    enhanced_scores[item_idx] = max(0.0, base_scores[item_idx] - (lambda_param * penalty))

# 판다스 데이터프레임 구조화 및 정렬
results = []
for item_idx in unseen_indices:
    results.append({
        "Content": content_names[item_idx],
        "Genre": categories[item_idx],
        "Base Score (Bias)": round(base_scores[item_idx], 3),
        "Penalty Size": round(penalties[item_idx] * lambda_param, 3),
        "Adjusted Score (Diverse)": round(enhanced_scores[item_idx], 3)
    })
df_result = pd.DataFrame(results).sort_values(by="Base Score (Bias)", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------------------
# 4. 웹 우측 메인 화면: 정량 지표 대시보드 및 결과 출력
# ------------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Detected High Bias Genres", value=str(favorite_genres) if favorite_genres else "None")
with col2:
    st.metric(label="Genre Bias Ratio", value=f"{bias_ratio * 100:.1f}%")
with col3:
    st.metric(label="Filter Strength (lambda)", value=str(lambda_param))

st.subheader("📊 Simulation Result Chart")

# 안전한 웹 렌더링을 위해 객체 지향형 구조로 명시적 분리하여 그래프 생성
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)

x = np.arange(len(df_result))
width = 0.35

# 막대그래프 드로잉
bars1 = ax.bar(x - width/2, df_result["Base Score (Bias)"], width, label='1. Standard AI (Bias)', color='#3498db', alpha=0.9)
bars2 = ax.bar(x + width/2, df_result["Adjusted Score (Diverse)"], width, label='2. Safe Recommendation (Diversity)', color='#2ecc71', alpha=0.9)

# 페널티 화살표 및 상태 텍스트 시각화
for i in range(len(df_result)):
    base_val = df_result["Base Score (Bias)"][i]
    adj_val = df_result["Adjusted Score (Diverse)"][i]
    diff = base_val - adj_val
    
    if diff > 0.05:
        ax.annotate('', xy=(i + width/2, adj_val), xytext=(i + width/2, base_val),
                    arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.5))
        ax.text(i + width/2, (base_val + adj_val)/2 + 0.05, f'-{diff:.1f}', 
                color='#e74c3c', ha='left', va='center', fontsize=9, fontweight='bold')
        ax.text(i - width/2, base_val + 0.1, '[Biased!]', color='#c0392b', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax.text(i + width/2, adj_val + 0.1, '★Fresh', color='#27ae60', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title(f'Recommendation Score Change for {user_name}', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Unseen Movies (* representing unseen movies in database)', fontsize=11, labelpad=10)
ax.set_ylabel('Score (0.0 to 5.0)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(df_result["Content"], rotation=20, ha='right')
ax.set_ylim(0, 6)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(loc='upper right')

# 완성된 도화지(fig) 객체를 스트림릿에 안전하게 전달
st.pyplot(fig)

# 하단 데이터 표 추가 노출
st.subheader("📋 Raw Data Table")
st.dataframe(df_result, use_container_width=True)
