import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
import twstock

# 把網頁標籤也改成帥順的專屬名稱
st.set_page_config(page_title="帥順股市分析與資產管理神器", layout="wide")

# ==========================================
# 🔒 隱私防護系統：請在這裡設定你的專屬密碼
# ==========================================
APP_PASSWORD = "8888" 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 帥順專屬系統已上鎖")
    st.info("此為私人財務追蹤系統，請輸入密碼以進行解鎖。")
    
    pwd_input = st.text_input("🔑 請輸入密碼：", type="password")
    
    if st.button("解鎖登入"):
        if pwd_input == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun() 
        else:
            st.error("❌ 密碼錯誤，請重新輸入。")
            
    st.stop() 

# ==========================================
# 🔓 以下為密碼正確後，才會顯示的正式內容
# ==========================================
st.title("🚀 帥順股市分析與資產管理神器")

# 你的 Google 試算表 CSV 專屬網址
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ4j2F1BSeWfRyA748KJh4hkU3KB26odS4uTfP7AZQgNcR0zvQVvjjYOfIvku-5vi8FcyW2BxNBDtq/pub?output=csv"

@st.cache_data(ttl=60)
def load_portfolio(url):
    try:
        df = pd.read_csv(url)
        portfolio = {}
        for index, row in df.iterrows():
            if pd.notna(row['代號']):
                symbol = str(row['代號']).strip()
                pure_code = symbol.split('.')[0]
                
                if pure_code in twstock.codes:
                    stock_name = twstock.codes[pure_code].name
                else:
                    stock_name = str(row['股票名稱']).strip() if '股票名稱' in df.columns and pd.notna(row['股票名稱']) else "未知"
                    
                portfolio[symbol] = {
                    'cost': float(row['成本']), 
                    'shares': int(row['股數']),
                    'name': stock_name
                }
        return portfolio
    except Exception as e:
        st.error("讀取試算表失敗，請確認網址是否正確且已設定為 CSV 發布。")
        return {}

MY_PORTFOLIO = load_portfolio(SHEET_URL)

tab1, tab2 = st.tabs(["📈 個股技術分析", "💰 我的投資組合"])

# ----------------------------------------
# 分頁 1：個股技術分析與警示
# ----------------------------------------
with tab1:
    def display_stock(symbol):
        if symbol in MY_PORTFOLIO and MY_PORTFOLIO[symbol]['name']:
            return f"{symbol} ({MY_PORTFOLIO[symbol]['name']})"
        return symbol

    stock_options = list(MY_PORTFOLIO.keys()) + ["手動輸入其他代號..."]
    selected_option = st.selectbox("請選擇要分析的自選股 (或選擇手動輸入)", stock_options, format_func=display_stock)

    if selected_option == "手動輸入其他代號...":
        ticker_symbol = st.text_input("請輸入股票代號 (台股請加 .TW 或 .TWO)", "00878.TW")
        pure_code = ticker_symbol.split('.')[0]
        if pure_code in twstock.codes:
            display_name = f"{ticker_symbol} ({twstock.codes[pure_code].name})"
        else:
            display_name = ticker_symbol
    else:
        ticker_symbol = selected_option
        display_name = display_stock(ticker_symbol)

    if ticker_symbol:
        st.subheader(f"正在分析： **{display_name}**")
        
        # 🌟 修改點：移除 session，讓 yfinance 自己用最高級的方式抓資料
        ticker_data = yf.Ticker(ticker_symbol)
        df = ticker_data.history(period="1y")
        
        if not df.empty:
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            kd = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=9, smooth_window=3)
            df['K'] = kd.stoch()
            
            latest_price = df['Close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            kd_k = df['K'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("最新收盤價", f"{latest_price:.2f}")
            
            if latest_price > ma20:
                col2.success(f"🟢 多頭格局 (站上月線 {ma20:.2f})")
            else:
                col2.error(f"🔴 空頭警訊 (跌破月線 {ma20:.2f})")
                
            if kd_k > 80:
                col3.warning(f"⚠️ KD過熱 (K值: {kd_k:.1f})")
            elif kd_k < 20:
                col3.info(f"💡 KD超賣 (K值: {kd_k:.1f})")
            else:
                col3.metric("KD - K值", f"{kd_k:.1f}")

            df_plot = df.tail(120)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
                increasing_line_color='red', decreasing_line_color='green', name='K線'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], line=dict(color='orange', width=1.5), name='20日線(月)'), row=1, col=1)
            
            colors = ['red' if row['Close'] >= row['Open'] else 'green' for index, row in df_plot.iterrows()]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
            
            fig.update_layout(title="技術分析圖表 (可滑動縮放)", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("找不到該股票資料，可能是代號錯誤或系統連線異常。")

# ----------------------------------------
# 分頁 2：我的投資組合 (損益追蹤)
# ----------------------------------------
with tab2:
    st.subheader("💼 持股即時淨損益狀態 (已自動判斷 ETF 優惠稅率)")
    
    if MY_PORTFOLIO:
        portfolio_data = []
        total_cost = 0
        total_value = 0
        
        my_bar = st.progress(0, text="正在為您結算持股最新報價...")
        items = list(MY_PORTFOLIO.items())
        
        for i, (symbol, info) in enumerate(items):
            # 🌟 修改點：同樣移除 session
            tick = yf.Ticker(symbol)
            hist = tick.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                cost = info['cost']
                shares = info['shares']
                stock_name = info['name']
                
                stock_cost_raw = cost * shares
                stock_value_raw = current_price * shares
                
                discount = 0.6
                buy_fee = max(20, stock_cost_raw * 0.001425 * discount)
                sell_fee = max(20, stock_value_raw * 0.001425 * discount)
                
                if symbol.startswith("00"):
                    tax = stock_value_raw * 0.001
                    type_label = "ETF"
                else:
                    tax = stock_value_raw * 0.003
                    type_label = "個股"
                
                true_stock_cost = stock_cost_raw + buy_fee
                true_profit = stock_value_raw - stock_cost_raw - buy_fee - sell_fee - tax
                roi = (true_profit / true_stock_cost) * 100 if true_stock_cost > 0 else 0
                
                total_cost += true_stock_cost
                total_value += stock_value_raw
                
                portfolio_data.append({
                    "股票名稱": stock_name,
                    "股票代號": f"{symbol} ({type_label})",
                    "持股數": shares,
                    "平均成本": cost,
                    "最新股價": round(current_price, 2),
                    "總成本(含息)": true_stock_cost,
                    "目前市值": round(stock_value_raw, 2),
                    "淨損益": round(true_profit, 0),
                    "報酬率 (%)": round(roi, 2)
                })
            my_bar.progress((i + 1) / len(items), text="正在為您結算持股最新報價...")
            
        my_bar.empty()
        
        total_profit = sum([p["淨損益"] for p in portfolio_data])
        total_roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("投資總成本 (含手續費)", f"${total_cost:,.0f}")
        col2.metric("目前總市值", f"${total_value:,.0f}")
        col3.metric("總未實現淨利", f"${total_profit:,.0f}", f"{total_roi:.2f}%")
        
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio.style.format({
            "持股數": "{:,.0f}",
            "平均成本": "{:.2f}",
            "最新股價": "{:.2f}",
            "總成本(含息)": "${:,.0f}",
            "目前市值": "${:,.0f}",
            "淨損益": "${:,.0f}"
        }), use_container_width=True)
        
        st.caption("💡 想要修改持股？請直接在手機上開啟您的 Google 試算表更新資料，APP 會在 60 秒內自動同步。")
    else:
        st.info("尚未從試算表讀取到持股資料。請確認您的試算表 A、B、C 欄有正確輸入內容。")
