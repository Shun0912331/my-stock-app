import streamlit as st
import yfinance as yf
import pandas as pd

# 設定網頁標題與排版
st.set_page_config(page_title="我的專屬選股 APP", layout="centered")
st.title("📈 股票即時分析儀表板")

# 讓使用者輸入股票代號
ticker_symbol = st.text_input("請輸入股票代號 (台股請加 .TW，例如 2330.TW)", "2330.TW")

if ticker_symbol:
    st.write(f"正在分析： **{ticker_symbol}**")
    
    # 抓取過去一年的資料 (為了計算均線需要多一點資料)
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period="1y")
    
    if not df.empty:
        # 自動計算 20 日均線 (月線)
        df['20MA'] = df['Close'].rolling(window=20).mean()
        
        # 顯示最新收盤價
        latest_price = df['Close'].iloc[-1]
        st.metric(label="最新收盤價", value=f"{latest_price:.2f}")
        
        # 繪製走勢圖 (同時顯示收盤價與月線)
        st.subheader("近半年股價與月線走勢")
        # 只取最近 120 天的資料來畫圖，畫面比較好看
        chart_data = df[['Close', '20MA']].tail(120)
        st.line_chart(chart_data)
        
        # 顯示原始數據表格
        with st.expander("查看詳細歷史數據"):
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
    else:
        st.error("找不到該股票資料，請確認代號是否正確。")
