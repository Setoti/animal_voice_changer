import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import io
import resampy
import pyworld as pw

# ページ設定
st.set_page_config(page_title="どうぶつ語ボイスチェンジャー", layout="centered")

st.title("どうぶつ語ボイスチェンジャー")
st.write("音声をアップロードして、どうぶつ語に加工しましょう。")

# 重い処理をキャッシュする関数
@st.cache_data
def animal_voice_effect( x, fs, speed, f0_base):
    # 基本周波数、スペクトル包絡、非周期指標の抽出
    _f0, t = pw.dio(x, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(x, _f0, t, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(x, f0, t, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(x, f0, t, fs)  # 非周期性指標の抽出

    # 基本周波数の変化の抑制
    f0[f0>0] = (f0[f0>0] - f0_base) * 0.1 + f0_base
    synthesized = pw.synthesize(f0, sp, ap, fs)

    # 早送り再生（リサンプリング）
    fs_new = int(fs/speed)
    y = resampy.resample(synthesized, sr_orig=fs, sr_new=fs_new)

    return y

uploaded_file = st.file_uploader("音声ファイルを選択", type=['mp3', 'wav', 'ogg'])

if uploaded_file is not None:

    st.subheader("加工前の音声")
    # アップロードファイルの表示
    st.audio(uploaded_file)

    # データの読み込み
    with st.spinner('読み込み中...'):
        y, sr = librosa.load(uploaded_file)
        y = y.astype(np.float64)
    
    st.subheader("加工後の音声")
    
    # サイドバーやカラムを使って設定UIを作成
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("音声速度 [倍]", 1.5, 3.0, 2.0)
    with col2:
        f0 = st.slider("声の高さ [Hz]（基本周波数：F0）", 50, 400, 200)

    # 実行ボタン（スライダーを動かしただけでは重い処理をさせない）
    with st.spinner("エフェクトを計算中..."):
        # エフェクト処理
        y_processed = animal_voice_effect( y, sr, speed, f0 )

        # プレビュー再生
        preview_io = io.BytesIO()
        sf.write(preview_io, y_processed, sr, format='WAV')
        st.audio(preview_io.getvalue(), format='audio/wav')
        
else:
    st.info("音声をアップロードしてください。")