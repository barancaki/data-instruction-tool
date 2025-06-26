import streamlit as st
import pandas as pd
from Levenshtein import ratio
import tempfile
from itertools import combinations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
import gc

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedDataMatcher:
    def __init__(self, similarity_threshold: float = 0.7, batch_size: int = 1000):
        self.similarity_threshold = similarity_threshold / 100
        self.batch_size = batch_size
        self.use_tfidf_for_large_data = True
        self.large_data_threshold = 5000  # 5000+ kayıt için TF-IDF kullan

    def _clean_value(self, value) -> str:
        """Değerleri temizle ve standartlaştır"""
        if pd.isna(value):
            return ""
        return str(value).strip().lower()  # Küçük harfe çevir

    def _calculate_column_similarity(self, val1: str, val2: str) -> float:
        """İki değer arasındaki benzerliği hesapla"""
        if not val1 or not val2:
            return 0.0
        return ratio(val1, val2)

    def _calculate_tfidf_similarity(self, val1: str, val2: str) -> float:
        """TF-IDF ile benzerlik hesapla"""
        if not val1 or not val2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 3),
                min_df=1
            )
            vectors = vectorizer.fit_transform([val1, val2])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except:
            return ratio(val1, val2)

    def _find_matches_by_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                cols: List[str], file1_name: str, file2_name: str,
                                original_df1: pd.DataFrame, original_df2: pd.DataFrame,
                                use_tfidf: bool = False) -> List[Dict]:
        """Kolonları ayrı ayrı karşılaştırarak eşleşmeleri bul"""
        results = []
        
        # Şirket standart kolonları
        company_columns = ["Data Source", "CompanyName", "CompanyWebsite", "CompanyMail"]
        
        # Her df1 satırını df2'deki tüm satırlarla karşılaştır
        total_comparisons = len(df1) * len(df2)
        
        # Progress bar
        progress_bar = st.progress(0)
        processed = 0
        
        for idx1, row1 in df1.iterrows():
            # Satır1'in tüm kolonları boş mu kontrol et
            row1_has_data = any(self._clean_value(row1.get(col, "")) for col in cols)
            if not row1_has_data:
                processed += len(df2)
                continue
                
            best_matches = []
            
            for idx2, row2 in df2.iterrows():
                # Satır2'nin tüm kolonları boş mu kontrol et
                row2_has_data = any(self._clean_value(row2.get(col, "")) for col in cols)
                if not row2_has_data:
                    processed += 1
                    continue
                
                column_similarities = {}
                total_similarity = 0
                valid_columns = 0
                
                # Her kolon için benzerlik hesapla
                for col in cols:
                    val1 = self._clean_value(row1.get(col, ""))
                    val2 = self._clean_value(row2.get(col, ""))
                    
                    if val1 and val2:  # İkisi de dolu ise
                        if use_tfidf:
                            sim = self._calculate_tfidf_similarity(val1, val2)
                        else:
                            sim = self._calculate_column_similarity(val1, val2)
                        
                        column_similarities[col] = sim
                        total_similarity += sim
                        valid_columns += 1
                
                # Ortalama benzerlik hesapla
                if valid_columns > 0:
                    avg_similarity = total_similarity / valid_columns
                    
                    # En az bir kolon eşik değeri geçiyorsa veya ortalama eşik geçiyorsa
                    max_col_sim = max(column_similarities.values()) if column_similarities else 0
                    
                    if max_col_sim >= self.similarity_threshold or avg_similarity >= self.similarity_threshold:
                        best_matches.append({
                            'idx2': idx2,
                            'avg_similarity': avg_similarity,
                            'max_similarity': max_col_sim,
                            'column_similarities': column_similarities,
                            'row2': row2
                        })
                
                processed += 1
                if processed % 100 == 0:
                    progress_bar.progress(processed / total_comparisons)
            
            # En iyi eşleşmeleri sonuçlara ekle
            for match in best_matches:
                result_row = {}
                
                # Şirket standart kolonlarını ekle
                for col in company_columns:
                    if col in original_df1.columns:
                        result_row[f"{col}_{file1_name}"] = original_df1.loc[idx1, col]
                    else:
                        result_row[f"{col}_{file1_name}"] = ""
                        
                    if col in original_df2.columns:
                        result_row[f"{col}_{file2_name}"] = original_df2.loc[match['idx2'], col]
                    else:
                        result_row[f"{col}_{file2_name}"] = ""
                
                # Eşleşme detayları
                match_details = []
                for col, sim in match['column_similarities'].items():
                    if sim >= self.similarity_threshold:
                        match_details.append(f"{col}: {sim:.3f}")
                
                # Eşleşme tipi belirleme
                if match['max_similarity'] >= 0.95:
                    match_type = "Exact Match"
                elif match['max_similarity'] >= self.similarity_threshold:
                    match_type = f"Strong Match ({use_tfidf and 'TF-IDF' or 'Levenshtein'})"
                else:
                    match_type = f"Partial Match ({use_tfidf and 'TF-IDF' or 'Levenshtein'})"
                
                result_row["Match_Type"] = match_type
                result_row["Average_Similarity"] = round(match['avg_similarity'], 3)
                result_row["Max_Similarity"] = round(match['max_similarity'], 3)
                result_row["Matching_Columns"] = "; ".join(match_details)
                result_row["Source"] = f"{file1_name} vs {file2_name}"
                
                results.append(result_row)
        
        progress_bar.empty()
        return results

    def _exact_match_by_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, cols: List[str], 
                               file1_name: str, file2_name: str, 
                               original_df1: pd.DataFrame, original_df2: pd.DataFrame) -> List[Dict]:
        """Kolonları ayrı ayrı exact match kontrolü"""
        results = []
        company_columns = ["Data Source", "CompanyName", "CompanyWebsite", "CompanyMail"]
        
        # Her kolon için ayrı exact match
        for col in cols:
            if col not in df1.columns or col not in df2.columns:
                continue
                
            # Bu kolonda exact match olanları bul (boş değerleri hariç tut)
            df1_filtered = df1[df1[col].notna() & (df1[col].astype(str).str.strip() != "")]
            df2_filtered = df2[df2[col].notna() & (df2[col].astype(str).str.strip() != "")]
            
            merged = pd.merge(df1_filtered[[col]], df2_filtered[[col]], on=col, how="inner")
            
            if not merged.empty:
                # Eşleşen değerleri bul
                matching_values = merged[col].unique()
                
                for value in matching_values:
                    if not value or str(value).strip() == "":  # Boş değerleri atla
                        continue
                        
                    df1_matches = df1_filtered[df1_filtered[col] == value]
                    df2_matches = df2_filtered[df2_filtered[col] == value]
                    
                    # Kartezyen çarpım (her eşleşmeyi kaydet)
                    for _, row1 in df1_matches.iterrows():
                        for _, row2 in df2_matches.iterrows():
                            result_row = {}
                            
                            # Şirket standart kolonlarını ekle
                            for comp_col in company_columns:
                                if comp_col in original_df1.columns:
                                    result_row[f"{comp_col}_{file1_name}"] = original_df1.loc[row1.name, comp_col]
                                else:
                                    result_row[f"{comp_col}_{file1_name}"] = ""
                                    
                                if comp_col in original_df2.columns:
                                    result_row[f"{comp_col}_{file2_name}"] = original_df2.loc[row2.name, comp_col]
                                else:
                                    result_row[f"{comp_col}_{file2_name}"] = ""
                            
                            result_row["Match_Type"] = "Exact Match"
                            result_row["Average_Similarity"] = 1.0
                            result_row["Max_Similarity"] = 1.0
                            result_row["Matching_Columns"] = f"{col}: 1.000"
                            result_row["Source"] = f"{file1_name} vs {file2_name}"
                            
                            results.append(result_row)
        
        return results

    def _match_two_files(self, df1: pd.DataFrame, df2: pd.DataFrame, cols: List[str], 
                        file1_name: str, file2_name: str) -> pd.DataFrame:
        """İki dosyayı karşılaştır - Yeni algoritma"""
        try:
            start_time = time.time()
            
            # Kopya oluştur
            df1 = df1.copy()
            df2 = df2.copy()
            original_df1 = df1.copy()
            original_df2 = df2.copy()
            
            # Kolonları temizle
            for col in cols:
                if col in df1.columns:
                    df1[col] = df1[col].apply(self._clean_value)
                if col in df2.columns:
                    df2[col] = df2[col].apply(self._clean_value)
            
            if df1.empty or df2.empty:
                st.warning(f"Karşılaştırma için yeterli veri yok: {file1_name} vs {file2_name}")
                return pd.DataFrame()
            
            st.info(f"📊 Karşılaştırılıyor: {file1_name} ({len(df1)} kayıt) vs {file2_name} ({len(df2)} kayıt)")
            
            results = []
            
            # 1. Exact Match (kolon bazlı)
            with st.spinner("🎯 Exact matching (kolon bazlı)..."):
                exact_results = self._exact_match_by_columns(
                    df1, df2, cols, file1_name, file2_name, original_df1, original_df2
                )
                results.extend(exact_results)
                st.success(f"✅ Exact match tamamlandı: {len(exact_results)} eşleşme")
            
            # 2. Fuzzy Match (kolon bazlı)
            with st.spinner("🔍 Fuzzy matching (kolon bazlı)..."):
                # Büyük veri seti kontrolü
                total_comparisons = len(df1) * len(df2)
                use_tfidf = (self.use_tfidf_for_large_data and 
                           total_comparisons > self.large_data_threshold)
                
                if use_tfidf:
                    st.info("🚀 Büyük veri seti tespit edildi. TF-IDF algoritması kullanılıyor...")
                else:
                    st.info("📝 Levenshtein algoritması kullanılıyor...")
                
                fuzzy_results = self._find_matches_by_columns(
                    df1, df2, cols, file1_name, file2_name, 
                    original_df1, original_df2, use_tfidf
                )
                
                # Exact match'te bulunanları çıkar (duplikasyonu önle)
                exact_pairs = set()
                for exact_result in exact_results:
                    # Satır tanımlayıcısı oluştur
                    identifier = (
                        exact_result.get(f"CompanyName_{file1_name}", ""),
                        exact_result.get(f"CompanyName_{file2_name}", "")
                    )
                    exact_pairs.add(identifier)
                
                # Fuzzy results'tan exact match'leri çıkar
                filtered_fuzzy = []
                for fuzzy_result in fuzzy_results:
                    identifier = (
                        fuzzy_result.get(f"CompanyName_{file1_name}", ""),
                        fuzzy_result.get(f"CompanyName_{file2_name}", "")
                    )
                    if identifier not in exact_pairs:
                        filtered_fuzzy.append(fuzzy_result)
                
                results.extend(filtered_fuzzy)
                st.success(f"✅ Fuzzy match tamamlandı: {len(filtered_fuzzy)} eşleşme")
            
            elapsed_time = time.time() - start_time
            st.info(f"⏱️ Karşılaştırma süresi: {elapsed_time:.2f} saniye")
            
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            st.error(f"❌ Karşılaştırma hatası ({file1_name} vs {file2_name}): {str(e)}")
            logger.error(f"Match error: {e}")
            return pd.DataFrame()


# Streamlit Arayüzü
def main():
    st.set_page_config(page_title="Excel Karşılaştırıcı", layout="wide")
    st.header("📁 Optimize Edilmiş Çoklu Excel Dosyası Karşılaştırıcı (Kolon Bazlı)")
    
    # Yan panel - Ayarlar
    with st.sidebar:
        st.subheader("⚙️ Ayarlar")
        similarity = st.slider("Benzerlik Eşiği (%):", 0, 100, 70)
        batch_size = st.number_input("Batch Boyutu:", min_value=100, max_value=5000, value=1000)
        use_tfidf = st.checkbox("Büyük veri için TF-IDF kullan", value=True)
        
        st.subheader("📝 Yeni Özellikler")
        st.success("""
        **✨ Kolon Bazlı Eşleştirme:**
        - Her kolon ayrı ayrı karşılaştırılır
        - CompanyName eşleşen ama Website farklı olan kayıtlar da bulunur
        - Eşleşen kolonlar detaylı gösterilir
        - Partial match desteği
        """)
        
        st.subheader("📊 Optimizasyonlar")
        st.info("""
        - TF-IDF: Büyük veri setleri için hızlı
        - Batch Processing: Bellek verimli  
        - Progress Bar: İlerleme takibi
        - Hata Yönetimi: Güvenli işlem
        """)
    
    # Ana içerik
    uploaded_files = st.file_uploader(
        "Birden fazla Excel dosyasını buraya yükleyin", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) >= 2:
        st.subheader("📄 Dosya Önizlemeleri")

        dataframes = {}
        total_rows = 0
        
        # Dosyaları yükle
        for file in uploaded_files:
            try:
                with st.spinner(f"📂 {file.name} yükleniyor..."):
                    df = pd.read_excel(file)
                    if df.empty:
                        st.warning(f"⚠️ {file.name} boş!")
                        continue
                    
                    dataframes[file.name] = df
                    total_rows += len(df)
                    
                    # Önizleme
                    with st.expander(f"📋 {file.name} ({len(df):,} kayıt)"):
                        st.dataframe(df.head())
                        
            except Exception as e:
                st.error(f"❌ {file.name} dosyası okunamadı: {str(e)}")
                continue

        if not dataframes:
            st.error("❌ Hiçbir dosya başarıyla yüklenemedi!")
            return
            
        # Performans uyarısı
        if total_rows > 50000:
            st.warning(f"⚠️ Toplam {total_rows:,} kayıt tespit edildi. İşlem uzun sürebilir.")

        st.subheader("🔗 Karşılaştırılacak Ortak Sütunları Seçin")
        
        # Ortak sütunları bul
        common_cols = list(set.intersection(*(set(df.columns) for df in dataframes.values())))
        
        # Şirket standart kolonları kontrol et
        required_columns = ["Data Source", "CompanyName", "CompanyWebsite", "CompanyMail"]
        missing_columns = []
        
        for col in required_columns:
            if col not in common_cols:
                missing_columns.append(col)
        
        if missing_columns:
            st.error(f"❌ Şu gerekli kolonlar eksik: {', '.join(missing_columns)}")
            st.info("📋 Şirket standart kolonları: Data Source, CompanyName, CompanyWebsite, CompanyMail")
            return
            
        # Kullanıcı ek sütun seçebilir
        additional_cols = [col for col in common_cols if col not in required_columns]
        
        if additional_cols:
            st.info("✅ Şirket standart kolonları tespit edildi. İsteğe bağlı ek sütunlar seçebilirsiniz:")
            selected_additional = st.multiselect(
                "Ek sütunlar (opsiyonel):", 
                sorted(additional_cols),
                help="Karşılaştırmada kullanılacak ek sütunlar"
            )
            selected_columns = required_columns + selected_additional
        else:
            st.info("✅ Şirket standart kolonları tespit edildi.")
            selected_columns = required_columns

        if selected_columns:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("📁 Dosya Sayısı", len(dataframes))
                st.metric("📊 Toplam Kayıt", f"{total_rows:,}")
            
            with col2:
                combinations_count = len(list(combinations(dataframes.keys(), 2)))
                st.metric("🔄 Karşılaştırma Sayısı", combinations_count)
                st.metric("📋 Kullanılacak Sütun", len(selected_columns))
                
            # Kullanılan sütunları göster
            st.subheader("🔍 Karşılaştırmada Kullanılacak Sütunlar")
            st.write("**Şirket Standart Kolonlar:**")
            st.code("Data Source, CompanyName, CompanyWebsite, CompanyMail")
            
            if len(selected_columns) > 4:
                additional = [col for col in selected_columns if col not in ["Data Source", "CompanyName", "CompanyWebsite", "CompanyMail"]]
                st.write("**Ek Kolonlar:**")
                st.code(", ".join(additional))

            # Yeni algoritma açıklaması
            st.subheader("🆕 Kolon Bazlı Eşleştirme Algoritması")
            st.info("""
            **Nasıl Çalışır:**
            - Her satır, diğer dosyadaki tüm satırlarla karşılaştırılır
            - Her kolon ayrı ayrı benzerlik skorları alır  
            - Eşleşme koşulları:
              - En az bir kolon eşik değeri geçerse ✅
              - Veya ortalama benzerlik eşik değeri geçerse ✅
            - Sonuçta hangi kolonların eşleştiği gösterilir
            """)

            if st.button("🚀 Karşılaştırmaya Başla", type="primary"):
                with st.spinner("🧠 Tüm dosyalar karşılaştırılıyor..."):
                    start_time = time.time()
                    
                    # Matcher oluştur
                    matcher = OptimizedDataMatcher(
                        similarity_threshold=similarity,
                        batch_size=batch_size
                    )
                    matcher.use_tfidf_for_large_data = use_tfidf
                    
                    results = []
                    file_pairs = list(combinations(dataframes.items(), 2))
                    
                    # Her dosya çiftini karşılaştır
                    for i, ((file1, df1), (file2, df2)) in enumerate(file_pairs):
                        st.subheader(f"🔄 {i+1}/{len(file_pairs)}: {file1} ↔ {file2}")
                        
                        result = matcher._match_two_files(
                            df1, df2, selected_columns, 
                            file1.replace(".xlsx", "").replace(".xls", ""), 
                            file2.replace(".xlsx", "").replace(".xls", "")
                        )
                        
                        if not result.empty:
                            results.append(result)
                        
                        # Bellek temizliği
                        gc.collect()

                    # Sonuçları birleştir
                    if results:
                        final_df = pd.concat(results, ignore_index=True)
                        
                        # Sütun sıralaması - Şirket standardı
                        ordered_columns = []
                        
                        # Şirket standart kolonları (her dosya için)
                        company_columns = ["Data Source", "CompanyName", "CompanyWebsite", "CompanyMail"]
                        
                        for col in company_columns:
                            for file_name in [f.replace(".xlsx", "").replace(".xls", "") for f in dataframes.keys()]:
                                col_name = f"{col}_{file_name}"
                                if col_name in final_df.columns:
                                    ordered_columns.append(col_name)
                        
                        # Eşleşme bilgileri
                        ordered_columns.extend(["Match_Type", "Average_Similarity", "Max_Similarity", "Matching_Columns", "Source"])
                        
                        # Final dataframe
                        final_df = final_df[ordered_columns]
                        
                        # Sonuçları göster
                        total_time = time.time() - start_time
                        
                        st.success(f"🎉 Eşleşmeler tamamlandı!")
                        
                        # Metrikler
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📈 Toplam Eşleşme", f"{len(final_df):,}")
                        with col2:
                            exact_matches = len(final_df[final_df['Match_Type'] == 'Exact Match'])
                            st.metric("🎯 Exact Match", f"{exact_matches:,}")
                        with col3:
                            fuzzy_matches = len(final_df) - exact_matches
                            st.metric("🔍 Fuzzy Match", f"{fuzzy_matches:,}")
                        with col4:
                            st.metric("⏱️ Süre", f"{total_time:.1f}s")
                        
                        # Match türlerine göre dağılım
                        st.subheader("📊 Eşleşme Türleri Dağılımı")
                        match_counts = final_df['Match_Type'].value_counts()
                        for match_type, count in match_counts.items():
                            st.write(f"**{match_type}:** {count:,} adet")
                        
                        # Veri önizlemesi
                        st.subheader("📋 Sonuç Önizlemesi")
                        st.dataframe(final_df.head(100))
                        
                        # En çok eşleşen kolonlar
                        if 'Matching_Columns' in final_df.columns:
                            st.subheader("🏆 En Çok Eşleşen Kolonlar")
                            all_matches = []
                            for matches_str in final_df['Matching_Columns'].dropna():
                                if matches_str:
                                    matches = matches_str.split(';')
                                    for match in matches:
                                        if ':' in match:
                                            col_name = match.split(':')[0].strip()
                                            all_matches.append(col_name)
                            
                            if all_matches:
                                from collections import Counter
                                match_counter = Counter(all_matches)
                                for col, count in match_counter.most_common(5):
                                    st.write(f"**{col}:** {count:,} eşleşme")
                        
                        # İndirme butonu
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                            final_df.to_excel(tmp.name, index=False)
                            with open(tmp.name, "rb") as f:
                                st.download_button(
                                    "📥 Tüm Eşleşmeleri İndir (.xlsx)",
                                    f,
                                    file_name=f"kolon_bazli_eslesmeler_{int(time.time())}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    else:
                        st.warning("😔 Hiç eşleşme bulunamadı.")
        else:
            st.warning("⚠️ Lütfen karşılaştırmak için en az bir sütun seçin.")


if __name__ == "__main__":
    main()