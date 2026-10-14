import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date

# ---------------- CẤU HÌNH TRANG ----------------
st.set_page_config(
    page_title="Stock AI Predictor Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cố định Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# ---------------- SIDEBAR (MENU BÊN TRÁI) ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3075/3075908.png", width=80)
    st.title("Bảng Điều Khiển")
    st.markdown("Cấu hình thông số dự báo.")
    
    ticker = st.text_input("🔍 Mã chứng khoán:", "VNM.VN", help="Nhập mã chứng khoán bạn muốn dự báo, ví dụ: VNM.VN, FPT.VN, AAPL...")
    predict_days = st.slider("📅 Số ngày dự báo:", min_value=1, max_value=10, value=5)
    
    predict_button = st.button("Tạo Dự Báo Mới 🚀", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.caption("Ứng dụng sử dụng mô hình Mạng Nơ-ron nhân tạo LSTM để phân tích chuỗi thời gian.")

# ---------------- GIAO DIỆN CHÍNH ----------------
st.title("🤖 Trợ lý AI Dự Báo Giá Cổ Phiếu (Bản Pro)")

# Nếu người dùng chưa bấm nút, hiển thị thông báo chào mừng
if not predict_button:
    st.info("👈 Vui lòng nhập mã chứng khoán và bấm **'Tạo Dự Báo Mới'** ở cột bên trái để bắt đầu.")
    st.stop()

# Bắt đầu xử lý khi bấm nút
with st.spinner(f'Đang tải dữ liệu và phân tích mã {ticker.upper()}...'):
    try:
        # Tải dữ liệu
        df = yf.download(ticker, start="2018-01-01", end=date.today().strftime("%Y-%m-%d"))
        if df.empty:
            st.error("❌ Không tìm thấy dữ liệu. Kiểm tra lại mã chứng khoán.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi tải dữ liệu: {e}")
        st.stop()

    # Xử lý format của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        data = df['Close'][ticker].values.reshape(-1, 1)
        df_display = df['Close'][[ticker]].copy() # Cho tab lịch sử
    else:
        data = df[['Close']].values
        df_display = df[['Close']].copy()

    if len(data) < 100:
        st.error("❌ Dữ liệu quá ngắn để huấn luyện AI.")
        st.stop()

    current_price = float(data[-1][0])
    prev_price = float(data[-2][0])
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100

    # Tiền xử lý & Huấn luyện (Giữ nguyên logic logic AI)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    def create_sequences(data_seq, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data_seq)):
            X.append(data_seq[i-window_size:i])
            y.append(data_seq[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-60:], test_scaled))
    X_test, y_test = create_sequences(test_combined)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), epochs=10, batch_size=32, shuffle=False, callbacks=[early_stop], verbose=0)

    # Dự báo tương lai
    last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)
    future_predictions = []

    for i in range(predict_days):
        next_pred = model.predict(last_60_days, verbose=0)
        future_predictions.append(next_pred[0,0])
        last_60_days = np.append(last_60_days[:,1:,:], next_pred.reshape(1,1,1), axis=1)

    future_predictions = np.array(future_predictions).reshape(-1,1)
    future_predictions = scaler.inverse_transform(future_predictions)

    final_predicted_price = float(future_predictions[-1][0])
    profit_loss_value = final_predicted_price - current_price
    profit_loss_pct = (profit_loss_value / current_price) * 100

# ---------------- GIAO DIỆN KẾT QUẢ TỪNG TAB ----------------
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan & Dự báo", "📈 Dữ liệu Lịch sử", "🧠 Thông tin Mô hình"])

with tab1:
    st.subheader(f"Chỉ số cho {ticker.upper()}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá đóng cửa hiện tại", f"{current_price:,.2f}", f"{daily_change:,.2f} ({daily_change_pct:.2f}%) so với phiên trước")
    with col2:
        st.metric(f"Dự báo sau {predict_days} ngày", f"{final_predicted_price:,.2f}")
    with col3:
        st.metric("Dự kiến Lãi/Lỗ (%)", f"{profit_loss_pct:.2f}%", f"{profit_loss_value:,.2f}")
        
    st.markdown("### 📈 Biểu đồ biến động (Interactive)")
    
    # Vẽ bằng Plotly
    show_history = 120 # Hiện 120 ngày qua
    hist_data = data[-show_history:].flatten()
    
    # Tạo chuỗi ngày
    dates_hist = pd.date_range(end=date.today(), periods=show_history, freq='B')
    dates_future = pd.date_range(start=date.today() + pd.Timedelta(days=1), periods=predict_days, freq='B')
    
    fig = go.Figure()
    
    # Đường giá thật
    fig.add_trace(go.Scatter(x=dates_hist, y=hist_data, mode='lines', name='Giá thực tế', line=dict(color='#1f77b4', width=2)))
    
    # Đường giá dự báo (nối điểm cuối cùng của giá thật với điểm đầu của giá dự báo)
    pred_dates = [dates_hist[-1]] + list(dates_future)
    pred_values = [hist_data[-1]] + list(future_predictions.flatten())
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_values, mode='lines+markers', name='Giá dự báo', line=dict(color='#ff7f0e', width=2, dash='dot'), marker=dict(size=8)))
    
    fig.update_layout(
        title=f"Biểu đồ giá cổ phiếu {ticker.upper()}",
        xaxis_title="Thời gian",
        yaxis_title="Giá đóng cửa",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Dữ liệu thô gần đây của {ticker.upper()}")
    df_display.index = df_display.index.strftime('%Y-%m-%d')
    st.dataframe(df_display.tail(30), use_container_width=True)
    
    with st.expander("Tải dữ liệu xuống file CSV"):
        csv = df_display.to_csv()
        st.download_button(label="📥 Download CSV", data=csv, file_name=f"{ticker}_history.csv", mime='text/csv')

with tab3:
    st.subheader("Kiến trúc Mạng Nơ-ron")
    st.write("Mô hình được huấn luyện trực tiếp (On-the-fly) sử dụng kiến trúc Long Short-Term Memory (LSTM) - chuyên trị các bài toán chuỗi thời gian.")
    st.code("""
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
    """, language='python')
    st.info("Kích thước cửa sổ (Window Size) nhìn về quá khứ là: 60 ngày. Tỷ lệ tập Train/Test là 80/20. Tối đa chạy 10 Epochs với EarlyStopping.")
