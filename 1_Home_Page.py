import streamlit as st
import streamlit_authenticator as stauth
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Home Page",
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
    name, authentication_status, username = authenticator.login('🔐 Login', 'main')
    
    # Authentication durumunu kontrol et
    if authentication_status == False:
        st.error('❌ Username or password is wrong')
        
    elif authentication_status == None:
        st.warning('⚠️ Please enter your username and password')
        
    elif authentication_status:
        # Başarılı giriş sonrası ana içerik
        
        # Logout butonu sidebar'da
        authenticator.logout('🚪 Log out', 'sidebar', key='unique_key')
        
        # Hoş geldin mesajı
        st.success(f'🎉 Welcome **{name}**!')
        st.title('🏠 Home Page Dashboard')
        
        # Session state'e login bilgisini kaydet
        st.session_state['authentication_status'] = True
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        # Kullanıcı rolüne göre farklı içerik
        if username == 'admin':
            st.info('👑 You have logged in with your **administrator** privileges.')
        elif username == 'serpil.hft':
            st.info('👑 You have logged in with your **administrator** privileges.')
        else:
            st.info(f'👤 You have logged into the system as **{name}**.')
        
        # Dashboard metrikleri
        st.header('📊 Dashboard Summary')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="👥 Total users", 
                value="5", 
                delta="Active",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="📁 Active projects", 
                value="4", 
                delta="Active",
                delta_color="normal"
            )
            
        with col3:
            st.metric(
                label="✅ Completed", 
                value="3", 
                delta="0",
                delta_color="normal"
            )
            
        with col4:
            st.metric(
                label="⏱️ Waiting", 
                value="1", 
                delta="-1",
                delta_color="inverse"
            )
        
        # Ana içerik alanları
        st.header('📈 Last Activities')
        
        tab1, tab2, tab3 = st.tabs(["📊 Graphics", "📋 User Info Table", "⚙️ Settings"])
        
        with tab1:
            st.subheader("Performance Graphics")
            
            # Örnek grafik verisi
            import pandas as pd
            import numpy as np
            
            # Rastgele veri oluştur
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            data = pd.DataFrame({
                'Date': dates,
                'Project': np.random.randint(50, 200, 30),
                'Improve': np.random.randint(100, 500, 30)
            })
            
            st.line_chart(data.set_index('Date'))
            
        with tab2:
            st.subheader("User Table")
            
            user_data = pd.DataFrame({
                'User': ['admin', 'kullanici', 'sefa', 'dilara', 'serpil'],
                'Full Name': ['Baran Çakı', 'User', 'Sefa Uyar', 'Dilara Ay', 'Serpil Sözen'],
                'Role': ['Admin', 'User', 'User', 'User', 'Admin'],
                'Status': ['Active', 'Disabled', 'Active', 'Active', 'Active']
            })
            
            st.dataframe(user_data, use_container_width=True)
            
        with tab3:
            st.subheader("System Settings")
            
            if username == 'admin' or username == 'serpil.hft':
                st.success("🔧 As an administrator, you have access to all settings.")
                
                with st.form("admin_settings"):
                    st.selectbox("Theme Selection", ["Light", "Dark", "Auto"])
                    st.slider("Session Timeout (minute)", 5, 120, 30)
                    st.checkbox("E-mail notifications")
                    st.checkbox("SMS notifications")
                    
                    if st.form_submit_button("💾 Save all changes"):
                        st.success("✅ Saved all changes.")
            else:
                st.warning("⚠️ Administrator privileges are required to make changes to settings.")
        
        # Sidebar'da kullanıcı bilgileri
        st.sidebar.header('👤 User Info')
        st.sidebar.info(f'''
        **Name:** {name}  
        **User Name:** {username}  
        **Role:** {'Admin' if username == 'admin' or username == 'serpil.hft' else 'User'}
        ''')
        
        st.sidebar.header('🧭 Navigation')
        st.sidebar.success('Use the menu on the left to navigate to other pages.')
        
        # Sistem bilgileri (sadece admin için)
        if username == 'admin':
            st.sidebar.header('🔧 System Informations')
            st.sidebar.text('🔋 System: Active')
            st.sidebar.text('💾 Database: Linked')
            st.sidebar.text('🌐 Web App: Secured')

if __name__ == "__main__":
    main()