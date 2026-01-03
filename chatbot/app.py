import streamlit as st
import numpy as np
import pickle
import time

# --- CONFIG & LOAD MODEL ---
st.set_page_config(page_title="Chat Bot Rekomendasi Jurusan", page_icon="🤖")

@st.cache_resource
def load_model():
    with open('model_k-means.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()
kmeans = data["model"]
scaler = data["scaler"]
le_sekolah = data["le_sekolah"]
le_jurusan = data["le_jurusan"]
cluster_jurusan_map = data["cluster_map"]

# --- INISIALISASI SESSION STATE ---
# Untuk menyimpan riwayat percakapan
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo!!! Saya Bot Rekomendasi Jurusan Kuliah. Siapa Namamu dan Dari Mana Asal Sekolahmu?"}
    ]
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "step" not in st.session_state:
    st.session_state.step = 1

# --- FUNGSI PREDIKSI ---
def get_prediction(d):
    js_enc = le_sekolah.transform([d['asal'].lower()])[0]
    jur_enc = le_jurusan.transform([d['jurusan'].lower()])[0]
    
    input_array = np.array([[
        js_enc, jur_enc, d['pkn'], d['mtk'], d['indo'], d['ing'], d['lainnya'],
        d['olr'], d['mus'], d['edit'], d['game']
    ]])
    
    scaled = scaler.transform(input_array)
    cluster = kmeans.predict(scaled)[0]
    return cluster_jurusan_map.get(cluster, ["Umum"])

# --- HEADER ---
st.title("Chat Bot Rekomendasi :green[Jurusan Kuliah]")
st.write("Ngobrol dengan AI untuk menentukan masa depanmu.")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt := st.chat_input("Ketik jawabanmu di sini..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logika Alur Chat (State Machine)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # --- STEP 1: NAMA & ASAL SEKOLAH ---
        if st.session_state.step == 1:
            input_text = prompt.lower()
            
            # Mencari kata kunci SMA atau SMK di dalam kalimat
            if 'sma' in input_text:
                sekolah_input = 'sma'
            elif 'smk' in input_text:
                sekolah_input = 'smk'
            else:
                sekolah_input = None

            if sekolah_input is None:
                full_response = "Maaf, bot ini khusus untuk lulusan **SMA** atau **SMK**. Mohon sebutkan sekolahmu dengan jelas (Contoh: *Dede dari SMA*)."
            else:
                # Mengambil nama: kita anggap kata pertama adalah nama jika tidak pakai koma
                nama_input = prompt.split()[0].capitalize() 
                
                st.session_state.user_data['nama'] = nama_input
                st.session_state.user_data['asal'] = sekolah_input
                
                full_response = f"Salam kenal {nama_input}! Karena kamu dari {sekolah_input.upper()}, apa jurusanmu?"
                st.session_state.step = 2

        elif st.session_state.step == 2:
            input_jurusan = prompt.strip().lower()
            asal = st.session_state.user_data['asal']
            
            # Filter berdasarkan pilihan di Step 1
            if asal == 'sma':
                pilihan_valid = ['ipa', 'ips']
            else:
                pilihan_valid = ['rpl', 'tkj', 'tbsm', 'Multimedia']
            if input_jurusan not in pilihan_valid:
                full_response = f"Maaf, untuk {asal.upper()} pilihan yang tersedia hanya: **{', '.join(pilihan_valid).upper()}**. Silakan pilih salah satu."
            else:
                st.session_state.user_data['jurusan'] = input_jurusan
                full_response = f"Bagus! Jurusan {input_jurusan.upper()}. Sekarang Masukkan Nilai Kewarganegaraan Terakhir Kamu ?"
                st.session_state.step = 3.1

        # --- STEP 3: NILAI UMUM SATU PER SATU ---
        elif st.session_state.step == 3.1:
            try:
                # 1. Simpan nilai yang baru dimasukkan (90)
                st.session_state.user_data['pkn'] = float(prompt)
                
                # 2. Siapkan pertanyaan berikutnya
                full_response = "Bagus! Sekarang berapa nilai **Matematika** kamu?"
                
                # 3. NAIKKAN STEP (Penting!)
                st.session_state.step = 3.2
            except ValueError:
                full_response = "⚠️ Masukkan angka saja ya. Berapa nilai PKN kamu?"

        # --- STEP 3.2: TERIMA MTK, TANYA INDO ---
        elif st.session_state.step == 3.2:
            try:
                st.session_state.user_data['mtk'] = float(prompt)
                full_response = "Oke. Lanjut ke nilai **Bahasa Indonesia**:"
                st.session_state.step = 3.3
            except ValueError:
                full_response = "⚠️ Tolong masukkan angka untuk nilai Matematika."

        elif st.session_state.step == 3.3:
            try:
                st.session_state.user_data['indo'] = float(prompt)
                full_response = "Berapa nilai **Bahasa Inggris** kamu?"
                st.session_state.step = 3.4
            except ValueError:
                full_response = "⚠️ Masukkan angka untuk Bahasa Indonesia:"

        elif st.session_state.step == 3.4:
            try:
                st.session_state.user_data['ing'] = float(prompt)
                
                # JIKA IPA -> TANYA FISIKA
                if st.session_state.user_data['jurusan'] == 'ipa':
                    full_response = "Lanjut ke nilai Sains. Berapa nilai **Fisika**?"
                    st.session_state.step = 4.1
                # JIKA IPS -> TANYA EKONOMI
                elif st.session_state.user_data['jurusan'] == 'ips':
                    full_response = "Lanjut ke nilai Sosial. Berapa nilai **Ekonomi**?"
                    st.session_state.step = 4.1
                # JIKA SMK -> TANYA KOMPETENSI (LANGSUNG SELESAI)
                else:
                    full_response = "Berapa nilai **Kompetensi Kejuruan** kamu?"
                    st.session_state.step = 4.4
            except ValueError:
                full_response = "⚠️ Masukkan angka untuk Bahasa Inggris:"

        # --- STEP 4: NILAI KHUSUS (FISIKA/KIMIA/BIO ATAU EKO/GEO/SOS) ---
        elif st.session_state.step == 4.1:
            st.session_state.user_data['n1'] = float(prompt) # Fisika atau Ekonomi
            map_label = "Kimia" if st.session_state.user_data['jurusan'] == 'ipa' else "Geografi"
            full_response = f"Berapa nilai **{map_label}**?"
            st.session_state.step = 4.2

        elif st.session_state.step == 4.2:
            st.session_state.user_data['n2'] = float(prompt) # Kimia atau Geografi
            map_label = "Biologi" if st.session_state.user_data['jurusan'] == 'ipa' else "Sosiologi"
            full_response = f"Berapa nilai **{map_label}**?"
            st.session_state.step = 4.3

        elif st.session_state.step == 4.3:
            st.session_state.user_data['n3'] = float(prompt) # Biologi atau Sosiologi
            # HITUNG RATA-RATA UNTUK MODEL
            avg = (st.session_state.user_data['n1'] + st.session_state.user_data['n2'] + st.session_state.user_data['n3']) / 3
            st.session_state.user_data['lainnya'] = avg
            
            full_response = "Terakhir, apa hobimu? Ketik angka: \n1. Olahraga\n2. Musik\n3. Editing\n4. Game\n(Contoh: 1, 3)\n"
            st.session_state.step = 5

        elif st.session_state.step == 4.4: # KHUSUS SMK
            st.session_state.user_data['lainnya'] = float(prompt)
            full_response = "Terakhir, apa hobimu? Ketik angka: \n1. Olahraga\n2. Musik\n3. Editing\n4. Game\n(Contoh: 1, 3)"
            st.session_state.step = 5

        elif st.session_state.step == 4:
            if ',' in prompt:
                vals = [float(v.strip()) for v in prompt.split(',')]
                st.session_state.user_data['lainnya'] = sum(vals) / len(vals)
            else:
                st.session_state.user_data['lainnya'] = float(prompt)
            
            full_response = "Terakhir, apa hobimu? Ketik angka: 1 jika Hobi Olahraga, 2 jika Musik, 3 jika Editing, 4 jika Game. (Bisa pilih lebih dari satu, contoh: 1, 3)"
            st.session_state.step = 5

        elif st.session_state.step == 5:
            # Reset hobi
            h = {'olr':0, 'mus':0, 'edit':0, 'game':0}
            if '1' in prompt: h['olr'] = 1
            if '2' in prompt: h['mus'] = 1
            if '3' in prompt: h['edit'] = 1
            if '4' in prompt: h['game'] = 1
            st.session_state.user_data.update(h)

            # Hitung Rekomendasi
            hasil = get_prediction(st.session_state.user_data)
            full_response = f"Analisis selesai! Berdasarkan profilmu, kamu sangat cocok masuk jurusan: \n\n" + "\n".join([f"✅ **{j}**" for j in hasil])
            full_response += "\n\n**Apakah kamu ingin mencoba rekomendasi jurusan lagi? (Ya/Tidak)**"
            
            # Pindah ke step konfirmasi
            st.session_state.step = 6

        # --- STEP 6: CEK JAWABAN REKOMENDASI LAGI ---
        elif st.session_state.step == 6:
            jawaban = prompt.strip().lower()
            
            if "ya" in jawaban:
                # RESET DATA TAPI LANGSUNG MULAI LAGI
                st.session_state.user_data = {} 
                st.session_state.step = 1
                full_response = "Oke, mari kita mulai lagi! Siapa namamu dan dari mana asal sekolahmu?"
            else:
                full_response = "Baik, terima kasih sudah berkonsultasi! Semoga sukses dengan pilihan jurusanmu. 👋"
                # Tetap di step 6 atau selesaikan percakapan
        # Animasi mengetik
        for chunk in full_response.split():
            response_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})