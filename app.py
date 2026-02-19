import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
import twstock
import requests

# 把網頁標籤也改成帥順的專屬名稱
st.set_page_config(page_title="帥順股市分析與資產管理神器", layout="wide")

# ==========================================
# 🛡️ 破解 Yahoo 阻擋機制：偽裝成真人瀏覽器
# ==========================================
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

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
        
        # 傳入偽裝通道去抓資料
        ticker_data = yf.Ticker(ticker_symbol, session=yf_session)
        df = ticker_data.history(period="1y
