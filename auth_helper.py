import streamlit as st

def check_authentication():
    """
    Kullanıcının giriş yapıp yapmadığını kontrol eder
    Returns: True eğer giriş yapılmışsa, False aksi halde
    """
    if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
        st.warning('⚠️ Bu sayfaya erişim için giriş yapmalısınız!')
        st.info('👈 Lütfen Ana Sayfa\'ya giderek giriş yapın.')
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
    if 'name' in st.session_state:
        st.sidebar.success(f'Giriş yapılan: {st.session_state["name"]}')
        if st.sidebar.button('🚪 Çıkış Yap'):
            # Session state'i temizle
            st.session_state['authentication_status'] = False
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.rerun()