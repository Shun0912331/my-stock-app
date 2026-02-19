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

# 檢查使用者是否已經登入
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 如果還沒登入，就顯示輸入密碼的畫面
if not st.session_state["authenticated"]:
    st.title("🔒 帥順專屬系統已上鎖")
    st.info("此為私人財務追蹤系統，請輸入密碼以進行解鎖。")
    
    # type="password" 會讓輸入的字變成黑點，保護隱私
    pwd_input = st.text_input("🔑 請輸入密碼：", type="password")
    
    if st.button("解鎖登入"):
        if pwd_input == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun() # 密碼正確，重新載入頁面
        else:
            st.error("❌ 密碼錯誤，請重新輸入。")
            
    # st.stop() 非常重要！這會阻止系統繼續往下執行，保護底下的資料不被偷看
    st.stop() 

# ==========================================
# 🔓 以下為密碼正確後，才會顯示的正式內容
# ==========================================
# 更新為帥順的專屬大標題
st.title("🚀 帥順股市分析與資產管理神器")

# 你的 Google 試算表 CSV 專屬網址
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ4j2F1BSeWfRyA748KJh4hkU3KB26odS4uTfP7AZQgNcR0zvQVvjjYOfIvku-5vi8FcyW2BxNBDtq/pub?output=csv"

# 建立自動讀取試算表的函數 (設定快取，每 60 秒更新一次)
@st.cache_data(ttl=60)
def load_portfolio(url):
    try:
        df = pd.read_csv(url)
        portfolio = {}
        for index, row in df.iterrows():
            if pd.notna(row['代號']):
                symbol = str(row['代號']).strip()
                pure_code = symbol.split('.')[0]
                
                # 透過 twstock 查詢正統中文名稱
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

# 讀取持股資料
MY_PORTFOLIO = load_portfolio(SHEET_URL)

# 建立兩個分頁
tab1, tab2 = st.tabs(["📈 個股技術分析", "💰 我的投資組合"])

# ----------------------------------------
# 分頁 1：個股技術分析與警示
# ----------------------------------------
with tab1:
    def display_stock(symbol):
        if symbol in MY_PORTFOLIO and MY
