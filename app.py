import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator

# 1. 設定網頁排版 (改為寬螢幕模式)
st.set_page_config(page_title="我的終極選股 APP", layout="wide")

# ==========================================
# --- 這裡輸入你的持股與成本 (你的專屬資料庫) ---
# 格式： '股票代號': {'cost': 成本價, 'shares': 股數}
# ==========================================
MY_PORTFOLIO = {
    '2330.TW': {'cost': 650.0, 'shares': 1000},  # 台積電: 成本650，持有1000股(1張)
    '2317.TW': {'cost': 105.5, 'shares': 2000},  # 鴻海: 成本105.5，持有2000股(2張)
    '2454.TW': {'cost': 900.0, 'shares': 500}    # 聯發科: 成本900，持有500股(零股)
}

st.title("🚀 專屬股市分析與資產追蹤")

# 建立兩個分頁：一個看技術分析，一個看持股損益
tab1, tab2 = st.tabs(["📈 個股技術分析", "💰 我的投資組合"])

# ----------------------------------------
# 分頁 1：個股技術分析與警示
# ----------------------------------------
with tab1:
    # 自選股下拉選單 (結合你的持股 + 手動輸入選項)
    stock_options = list(MY_PORTFOLIO.keys()) + ["手動輸入其他代號..."]
    selected_option = st.selectbox("請選擇要分析的自選股 (或選擇手動輸入)", stock_options)

    if selected_option == "手動輸入其他代號...":
        ticker_symbol = st.text_input("請輸入股票代號 (台股請加 .TW)", "2603.TW")
    else:
        ticker_symbol = selected_option

    if ticker_symbol:
        st.subheader(f"正在分析： **{ticker_symbol}**")
        
        # 抓取過去一年的資料來計算長天期指標
        ticker_data = yf.Ticker(ticker_symbol)
        df = ticker_data.history(period="1y")
        
        if not df.empty:
            # --- 計算技術指標 ---
            # 均線 (MA5, MA20, MA60)
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # KD 指標
            kd = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=9, smooth_window=3)
            df['K'] = kd.stoch()
            df['D'] = kd.stoch_signal()
            
            # RSI 指標 (14日)
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()

            # --- 自動化條件判斷 (警示系統) ---
            latest_price = df['Close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            kd_k = df['K'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("最新收盤價", f"{latest_price:.2f}")
            
            # 判斷多空趨勢
            if latest_price > ma20:
                col2.success(f"🟢 多頭格局 (站上月線 {ma20:.2f})")
            else:
                col2.error(f"🔴 空頭警訊 (跌破月線 {ma20:.2f})")
                
            # 判斷超買超賣
            if kd_k > 80:
                col3.warning(f"⚠️ KD過熱 (K值: {kd_k:.1f}，有回檔風險)")
            elif kd_k < 20:
                col3.info(f"💡 KD超賣 (K值: {kd_k:.1f}，可能出現反彈)")
            else:
                col3.metric("KD - K值", f"{kd_k:.1f}")

            # --- 繪製專業 K 線圖 (符合台灣紅綠習慣) ---
            # 只取近 120 天畫圖比較清楚
            df_plot = df.tail(120)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # 上方 K 線與均線
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
                increasing_line_color='red', decreasing_line_color='green', name='K線'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'], line=dict(color='blue', width=1), name='5日線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='20日線(月)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA60'], line=dict(color='purple', width=1.5), name='60日線(季)'), row=1, col=1)
            
            # 下方成交量 (根據漲跌決定紅綠)
            colors = ['red' if row['Close'] >= row['Open'] else 'green' for index, row in df_plot.iterrows()]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
            
            fig.update_layout(title="進階技術分析圖表 (可滑動縮放)", xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("找不到該股票資料。")

# ----------------------------------------
# 分頁 2：我的投資組合 (損益追蹤)
# ----------------------------------------
with tab2:
    st.subheader("💼 持股即時損益狀態")
    
    if MY_PORTFOLIO:
        portfolio_data = []
        total_cost = 0
        total_value = 0
        
        # 加上進度條讓等待抓資料的過程更好看
        progress_text = "正在為您結算持股最新報價..."
        my_bar = st.progress(0, text=progress_text)
        
        items = list(MY_PORTFOLIO.items())
        for i, (symbol, info) in enumerate(items):
            tick = yf.Ticker(symbol)
            hist = tick.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                cost = info['cost']
                shares = info['shares']
                
                stock_cost = cost * shares
                stock_value = current_price * shares
                profit = stock_value - stock_cost
                roi = (profit / stock_cost) * 100 if stock_cost > 0 else 0
                
                total_cost += stock_cost
                total_value += stock_value
                
                portfolio_data.append({
                    "股票代號": symbol,
                    "持股數": shares,
                    "平均成本": cost,
                    "最新股價": round(current_price, 2),
                    "總成本": stock_cost,
                    "目前市值": round(stock_value, 2),
                    "未實現損益": round(profit, 0),
                    "報酬率 (%)": round(roi, 2)
                })
            # 更新進度條
            my_bar.progress((i + 1) / len(items), text=progress_text)
            
        my_bar.empty() # 隱藏進度條
        
        # 顯示總結
        total_profit = total_value - total_cost
        total_roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("投資總成本", f"${total_cost:,.0f}")
        col2.metric("目前總市值", f"${total_value:,.0f}")
        col3.metric("總未實現損益", f"${total_profit:,.0f}", f"{total_roi:.2f}%")
        
        # 顯示表格
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio.style.format({
            "持股數": "{:,.0f}",
            "總成本": "${:,.0f}",
            "目前市值": "${:,.0f}",
            "未實現損益": "${:,.0f}"
        }), use_container_width=True)
    else:
        st.info("您目前沒有設定任何持股。請至程式碼 `MY_PORTFOLIO` 區塊新增。")
