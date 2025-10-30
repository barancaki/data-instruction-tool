import streamlit as st
import streamlit_authenticator as stauth
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Eğer CapRover veya başka bir platform PORT environment variable ile başlatıyorsa,
# konteyner loglarında kolay takip için kısa bir çıktı ekleyin.
if os.getenv('PORT'):
    print(f"Starting in container environment. PORT={os.getenv('PORT')}")

# Yeni: CapRover / WebSocket troubleshooting bilgisi (sidebar'ta)
def show_ws_troubleshooting():
    """Kısa CapRover + WebSocket kontrol listesi."""
    with st.sidebar.expander("🛠️ WebSocket / CapRover Troubleshooting", expanded=False):
        st.markdown(
            "- Ensure your CapRover app has HTTPS enabled (Force HTTPS) if you use a custom domain.\n"
            "- Make sure CapRover proxies WebSocket traffic (CapRover normally supports websockets).\n"
            "- In CapRover → App Configs add required env vars (PASSWORD_*, COOKIE_*, PREAUTHORIZED_EMAIL).\n"
            "- If you added custom Nginx rules, ensure these headers are forwarded:\n"
            "  proxy_set_header Upgrade $http_upgrade;\n"
            "  proxy_set_header Connection $connection_upgrade;\n"
        )
        st.markdown("Preferred Streamlit launch (use in Docker CMD so CapRover's $PORT is used):")
        st.code("streamlit run 1_Home_Page.py --server.port ${PORT:-8080} --server.address 0.0.0.0 "
                "--server.headless true --server.enableCORS false --server.enableXsrfProtection false")

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Home Page",
    page_icon="🏠",
    layout="wide"
)

# Gösterge panelindeki rehberi her çalışmada ekrana koy
show_ws_troubleshooting()

# Authentication konfigürasyonu
def load_config():
    """Authentication konfigürasyonunu .env dosyasından yükle"""
    
    # Gerekli environment değişkenleri listesi
    required_keys = [
        'PASSWORD_ADMIN',
        'PASSWORD_KULLANICI',
        'PASSWORD_SEFA',
        'PASSWORD_DILARA',
        'PASSWORD_SERPIL',
        'PASSWORD_CIHAN',
        'COOKIE_KEY',
        'COOKIE_NAME',
        'COOKIE_EXPIRY_DAYS',
        'PREAUTHORIZED_EMAIL'
    ]
    
    # Eksik olanları tespit et
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        # Kullanıcıya net bilgi ver ve deploy'da CapRover App Configs'a eklemelerini söyle
        st.error(
            "Eksik environment değişkenleri tespit edildi: "
            f"{', '.join(missing)}\n\n"
            "CapRover'da uygulamanızın Settings -> App Configs (Environment Variables) bölümüne "
            "bunları ekleyip tekrar deploy edin. Yerel test için .env kullanıyorsanız, "
            "üretime .env göndermeyin; CapRover env'lerini kullanın."
        )
        st.stop()
    
    # .env dosyasından şifreleri al
    passwords = [
        os.getenv('PASSWORD_ADMIN'),
        os.getenv('PASSWORD_KULLANICI'),
        os.getenv('PASSWORD_SEFA'),
        os.getenv('PASSWORD_DILARA'),
        os.getenv('PASSWORD_SERPIL'),
        os.getenv('PASSWORD_CIHAN')
    ]
    
    # Şifreleri hash'le (hata olursa net mesaj ver)
    try:
        hashed_passwords = stauth.Hasher(passwords).generate()
    except Exception as e:
        st.error(f"Şifreleri hash'lerken hata oluştu: {e}\n"
                 "Env değişkenlerinizin doğru set edildiğinden emin olun.")
        st.stop()
    
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
                },
                'cihan.hft': {
                    'email': 'cihan.keser@hf-turkey.com',
                    'name': 'Cihan (Admin)',
                    'password': hashed_passwords[5]
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

def show_caprover_guide():
    """Sidebar'ta CapRover deploy rehberini göster."""
    guide = """
    CapRover deployment quick checklist:
    1) Add Dockerfile, requirements.txt and captain-definition to repo (Dockerfile must run Streamlit on $PORT).
    2) In CapRover -> Apps -> <app> -> App Configs, add these env vars from your .env:
       - PASSWORD_ADMIN, PASSWORD_KULLANICI, PASSWORD_SEFA, PASSWORD_DILARA, PASSWORD_SERPIL, PASSWORD_CIHAN
       - COOKIE_KEY, COOKIE_NAME, COOKIE_EXPIRY_DAYS
       - PREAUTHORIZED_EMAIL
    3) Deploy via CapRover (GitHub/CLI or manual). Check build logs for pip errors.
    4) If deployment fails due to pip dependency errors, relax versions in requirements.txt (remove strict pins) and rebuild.
    5) Check CapRover -> Logs for runtime errors and ensure port mapping is correct.
    """
    with st.sidebar.expander("🚀 CapRover Deployment Guide", expanded=False):
        st.markdown(guide)

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
        
        # Show CapRover guide in sidebar
        show_caprover_guide()
        
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