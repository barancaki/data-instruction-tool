import streamlit as st
import streamlit_authenticator as stauth
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Ana Sayfa",
    page_icon="🏠",
    layout="wide"
)

# Authentication konfigürasyonu
def load_config():
    """Authentication konfigürasyonunu .env dosyasından yükle"""
    
    # .env dosyasından şifreleri al
    passwords = [
        os.getenv('PASSWORD_ADMIN'),
        os.getenv('PASSWORD_KULLANICI'),
        os.getenv('PASSWORD_SEFA'),
        os.getenv('PASSWORD_DILARA'),
        os.getenv('PASSWORD_SERPIL')
    ]
    
    # Şifreleri hash'le
    hashed_passwords = stauth.Hasher(passwords).generate()
    
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': os.getenv('PREAUTHORIZED_EMAIL'),
                    'name': 'Baran Çakı (Admin)',
                    'password': hashed_passwords[0]
                },
                'kullanici': {
                    'email': 'kullanici@example.com',
                    'name': 'Genel Kullanıcı',
                    'password': hashed_passwords[1]
                },
                'sefa.hft': {
                    'email': 'sefa@hotmail.com',
                    'name': 'Sefa',
                    'password': hashed_passwords[2]
                },
                'dilara.hft': {
                    'email': 'dilara@hotmail.com',
                    'name': 'Dilara',
                    'password': hashed_passwords[3]
                },
                'serpil.hft': {
                    'email': 'serpil@hotmail.com',
                    'name': 'Serpil (Admin)',
                    'password': hashed_passwords[4]
                }
            }
        },
        'cookie': {
            'expiry_days': int(os.getenv('COOKIE_EXPIRY_DAYS', '30')),
            'key': os.getenv('COOKIE_KEY', 'some_signature_key_123456'),
            'name': os.getenv('COOKIE_NAME', 'streamlit_auth_cookie')
        },
        'preauthorized': {
            'emails': [os.getenv('PREAUTHORIZED_EMAIL')]
        }
    }
    return config

# Ana uygulama
def main():
    # Konfigürasyonu yükle
    config = load_config()
    
    # Authenticator oluştur
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    
    # Login widget'ı
    name, authentication_status, username = authenticator.login('🔐 Giriş Yap', 'main')
    
    # Authentication durumunu kontrol et
    if authentication_status == False:
        st.error('❌ Kullanıcı adı/şifre yanlış')
        
    elif authentication_status == None:
        st.warning('⚠️ Lütfen kullanıcı adı ve şifrenizi girin')
        
    elif authentication_status:
        # Başarılı giriş sonrası ana içerik
        
        # Logout butonu sidebar'da
        authenticator.logout('🚪 Çıkış Yap', 'sidebar', key='unique_key')
        
        # Hoş geldin mesajı
        st.success(f'🎉 Hoş geldiniz **{name}**!')
        st.title('🏠 Ana Sayfa Dashboard')
        
        # Session state'e login bilgisini kaydet
        st.session_state['authentication_status'] = True
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        # Kullanıcı rolüne göre farklı içerik
        if username == 'admin':
            st.info('👑 **Yönetici** yetkileriniz ile sisteme giriş yaptınız.')
        else:
            st.info(f'👤 **{name}** olarak sisteme giriş yaptınız.')
        
        # Dashboard metrikleri
        st.header('📊 Dashboard Özeti')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="👥 Toplam Kullanıcı", 
                value="5", 
                delta="Aktif",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="📁 Aktif Projeler", 
                value="4", 
                delta="Aktif",
                delta_color="normal"
            )
            
        with col3:
            st.metric(
                label="✅ Tamamlanan", 
                value="4", 
                delta="0",
                delta_color="normal"
            )
            
        with col4:
            st.metric(
                label="⏱️ Bekleyen", 
                value="3", 
                delta="-2",
                delta_color="inverse"
            )
        
        # Ana içerik alanları
        st.header('📈 Son Aktiviteler')
        
        tab1, tab2, tab3 = st.tabs(["📊 Grafikler", "📋 Tablolar", "⚙️ Ayarlar"])
        
        with tab1:
            st.subheader("Performans Grafikleri")
            
            # Örnek grafik verisi
            import pandas as pd
            import numpy as np
            
            # Rastgele veri oluştur
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            data = pd.DataFrame({
                'Tarih': dates,
                'Proje': np.random.randint(50, 200, 30),
                'Gelişim': np.random.randint(100, 500, 30)
            })
            
            st.line_chart(data.set_index('Tarih'))
            
        with tab2:
            st.subheader("Kullanıcı Tablosu")
            
            user_data = pd.DataFrame({
                'Kullanıcı': ['admin', 'kullanici', 'sefa', 'dilara', 'serpil'],
                'Tam İsim': ['Baran Çakı', 'Genel Kullanıcı', 'Sefa Uyar', 'Dilara Ay', 'Serpil Sözen'],
                'Rol': ['Yönetici', 'Kullanıcı', 'Kullanıcı', 'Kullanıcı', 'Yönetici'],
                'Durum': ['Aktif', 'Pasif', 'Aktif', 'Aktif', 'Aktif']
            })
            
            st.dataframe(user_data, use_container_width=True)
            
        with tab3:
            st.subheader("Sistem Ayarları")
            
            if username == 'admin':
                st.success("🔧 Yönetici olarak tüm ayarlara erişiminiz var.")
                
                with st.form("admin_settings"):
                    st.selectbox("Tema Seçimi", ["Light", "Dark", "Auto"])
                    st.slider("Session Timeout (dakika)", 5, 120, 30)
                    st.checkbox("E-posta bildirimleri")
                    st.checkbox("SMS bildirimleri")
                    
                    if st.form_submit_button("💾 Ayarları Kaydet"):
                        st.success("✅ Ayarlar kaydedildi!")
            else:
                st.warning("⚠️ Ayar değişiklikleri için yönetici yetkisi gereklidir.")
        
        # Sidebar'da kullanıcı bilgileri
        st.sidebar.header('👤 Kullanıcı Bilgileri')
        st.sidebar.info(f'''
        **İsim:** {name}  
        **Kullanıcı Adı:** {username}  
        **Rol:** {'Yönetici' if username == 'admin' else 'Kullanıcı'}
        ''')
        
        st.sidebar.header('🧭 Navigasyon')
        st.sidebar.success('Diğer sayfalara geçmek için soldaki menüyü kullanın.')
        
        # Sistem bilgileri (sadece admin için)
        if username == 'admin':
            st.sidebar.header('🔧 Sistem Bilgileri')
            st.sidebar.text('🔋 Sistem: Aktif')
            st.sidebar.text('💾 Veritabanı: Bağlı')
            st.sidebar.text('🌐 Bağlantı: Güvenli')

if __name__ == "__main__":
    main()