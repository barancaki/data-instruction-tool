import streamlit as st
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

def check_authentication():
    """
    Kullanıcının giriş yapıp yapmadığını kontrol eder
    Returns: True eğer giriş yapılmışsa, False aksi halde
    """
    if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
        st.warning('⚠️ Bu sayfaya erişim için giriş yapmalısınız!')
        st.info('👈 Lütfen Ana Sayfa\'ya giderek giriş yapın.')
        
        if st.button("🏠 Ana Sayfaya Git"):
            st.switch_page("1_Ana_Sayfa.py")
            
        st.stop()  # Sayfanın geri kalanını göstermez
        return False
    return True

def get_user_info():
    """
    Giriş yapmış kullanıcının bilgilerini döndürür
    Returns: dict with name and username
    """
    if check_authentication():
        return {
            'name': st.session_state.get('name', ''),
            'username': st.session_state.get('username', '')
        }
    return None

def show_user_info_sidebar():
    """
    Sidebar'da kullanıcı bilgilerini gösterir
    """
    if 'name' in st.session_state and st.session_state['name']:
        username = st.session_state.get('username', '')
        name = st.session_state.get('name', '')
        
        st.sidebar.header('👤 Kullanıcı Bilgileri')
        st.sidebar.success(f'**{name}** olarak giriş yaptınız')
        st.sidebar.info(f'Kullanıcı adı: `{username}`')
        
        # Rol bilgisi
        if username == 'admin':
            st.sidebar.info('🔑 **Yönetici** yetkileriniz var')
        
        # Çıkış yap butonu
        if st.sidebar.button('🚪 Çıkış Yap', key='sidebar_logout'):
            # Session state'i temizle
            st.session_state['authentication_status'] = False
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.switch_page("1_Ana_Sayfa.py")

def is_admin():
    """
    Kullanıcının admin olup olmadığını kontrol eder
    """
    return (st.session_state.get('username', '') == 'admin' and 
            st.session_state.get('authentication_status', False))