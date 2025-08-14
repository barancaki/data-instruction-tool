import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Ana Sayfa",
    page_icon="🏠",
    layout="wide"
)

# Authentication konfigürasyonu
def load_config():
    """Authentication konfigürasyonunu yükle"""
    # Şifreleri hash'le
    hashed_passwords = stauth.Hasher(['baranisreluctant.123', 'password123' , 'sefa.hft@2025' , 'dilara.hft@2025' , 'serpil.hft@2025']).generate()
    
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': 'baran.caki@hotmail.com',
                    'name': 'Administrator (Baran)',
                    'password': hashed_passwords[0]
                },
                'kullanici': {
                    'email': 'kullanici@example.com', 
                    'name': 'Kullanıcı',
                    'password': hashed_passwords[1]
                },
                'sefa.hft': {
                    'email': 'sefa.uyar@external.hf-turkey.com', 
                    'name': 'Sefa Uyar',
                    'password': hashed_passwords[2]
                },
                'dilara.hft': {
                    'email': 'dilara.ay@external.hf-turkey.com', 
                    'name': 'Dilara Ay',
                    'password': hashed_passwords[3]
                },
                'serpil.hft': {
                    'email': 'serpil.sozen@hf-turkey.com', 
                    'name': 'Serpil Sözen (Admin)',
                    'password': hashed_passwords[4]
                },
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'some_signature_key_123456',  # Güvenli bir anahtar
            'name': 'streamlit_auth_cookie'
        },
        'preauthorized': {
            'emails': ['baran.caki@hotmail.com']
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
    name, authentication_status, username = authenticator.login('Giriş Yap', 'main')
    
    # Authentication durumunu kontrol et
    if authentication_status == False:
        st.error('Kullanıcı adı/şifre yanlış')
        
    elif authentication_status == None:
        st.warning('Lütfen kullanıcı adı ve şifrenizi girin')
        
    elif authentication_status:
        # Başarılı giriş sonrası ana içerik
        authenticator.logout('Çıkış Yap', 'sidebar')
        
        st.write(f'Hoş geldiniz *{name}*!')
        st.title('🏠 Ana Sayfa')
        
        # Session state'e login bilgisini kaydet
        st.session_state['authentication_status'] = True
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        # Ana sayfa içeriği
        st.header('Dashboard')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Kullanıcı", "2", "1")
        
        with col2:
            st.metric("Aktif Projeler", "4", "1")
            
        with col3:
            st.metric("Tamamlanan", "2", "1")
        
        st.header('Son Aktiviteler')
        st.info('Fuar Scraper = 3 Yeni Fuar Eklendi !\n' \
        'PDF Scraper alanı eklendi (Erişime kapalıdır.)')
        
        # Sidebar'da kullanıcı bilgileri
        st.sidebar.success(f'Giriş yapılan kullanıcı: {name}')
        st.sidebar.info('Diğer sayfalara geçmek için soldaki menüyü kullanın.')

if __name__ == "__main__":
    main()