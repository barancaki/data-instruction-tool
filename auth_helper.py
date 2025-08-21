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
        st.warning('⚠️ You must log in to access this page!')
        st.info('👈 Please go to the Home Page and log in.')
        
        if st.button("🏠 Go to Home Page"):
            st.switch_page("1_Home_Page.py")
            
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
        
        st.sidebar.header('👤 User Information')
        st.sidebar.success(f'You have logged in as **{name}**')
        st.sidebar.info(f'Username: `{username}`')
        
        # Rol bilgisi
        if username == 'admin':
            st.sidebar.info('🔑 You have **administrator** privileges.')
        
        # Çıkış yap butonu
        if st.sidebar.button('🚪 Log Out', key='sidebar_logout'):
            # Session state'i temizle
            st.session_state['authentication_status'] = False
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.switch_page("1_Home_Page.py")

def is_admin():
    """
    Kullanıcının admin olup olmadığını kontrol eder
    """
    return (st.session_state.get('username', '') == 'admin' and 
            st.session_state.get('authentication_status', False))