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

    def _combine_columns(self, df: pd.DataFrame, columns: list, new_col: str) -> pd.DataFrame:
        """Sütunları birleştir ve temizle"""
        try:
            df[new_col] = df[columns].astype(str).agg(" ".join, axis=1)
            df[new_col] = df[new_col].apply(self._clean_value)
            return df
        except Exception as e:
            logger.error(f"Sütun birleştirme hatası: {e}")
            raise

    def _exact_match(self, df1: pd.DataFrame, df2: pd.DataFrame, combo_col: str, 
                    cols: List[str], file1_name: str, file2_name: str, 
                    original_df1: pd.DataFrame, original_df2: pd.DataFrame) -> List[Dict]:
        """Exact match işlemi"""
        results = []
        
        # Inner join ile exact match
        merged = pd.merge(df1, df2, on=combo_col, suffixes=(f'_{file1_name}', f'_{file2_name}'), how="inner")
        
        for _, row in merged.iterrows():
            result_row = {}
            
            # Data Source kolonunu kontrol et ve ekle
            if "Data Source" in original_df1.columns:
                df1_idx = df1[df1[combo_col] == row[combo_col]].index[0]
                result_row[f"Data Source_{file1_name}"] = original_df1.loc[df1_idx, "Data Source"]
            if "Data Source" in original_df2.columns:
                df2_idx = df2[df2[combo_col] == row[combo_col]].index[0]
                result_row[f"Data Source_{file2_name}"] = original_df2.loc[df2_idx, "Data Source"]
            
            # Seçilen sütunları ekle
            for col in cols:
                result_row[f"{col}_{file1_name}"] = row.get(f"{col}_{file1_name}", "")
                result_row[f"{col}_{file2_name}"] = row.get(f"{col}_{file2_name}", "")
            
            # Eşleşme bilgileri
            result_row["Match_Type"] = "Exact Match"
            result_row["Similarity"] = 1.0
            result_row["Source"] = f"{file1_name} vs {file2_name}"
            
            results.append(result_row)
        
        return results

    def _fuzzy_match_tfidf(self, left_values: List[str], right_values: List[str], 
                          left_indices: List[int], right_indices: List[int]) -> List[Tuple[int, int, float]]:
        """TF-IDF tabanlı fuzzy matching (büyük veri setleri için)"""
        try:
            if not left_values or not right_values:
                return []
            
            # TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 3),
                min_df=1,
                max_features=10000
            )
            
            # Tüm değerleri birleştir ve vektörize et
            all_values = left_values + right_values
            tfidf_matrix = vectorizer.fit_transform(all_values)
            
            # Sol ve sağ matrisleri ayır
            left_matrix = tfidf_matrix[:len(left_values)]
            right_matrix = tfidf_matrix[len(left_values):]
            
            # Cosine similarity hesapla
            similarity_matrix = cosine_similarity(left_matrix, right_matrix)
            
            # Eşik değeri üzerindeki eşleşmeleri bul
            matches = []
            rows, cols = np.where(similarity_matrix >= self.similarity_threshold)
            
            for row, col in zip(rows, cols):
                similarity = similarity_matrix[row, col]
                matches.append((left_indices[row], right_indices[col], float(similarity)))
            
            return matches
            
        except Exception as e:
            logger.error(f"TF-IDF fuzzy matching hatası: {e}")
            return []

    def _fuzzy_match_levenshtein_batch(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                                     combo_col: str) -> List[Tuple[int, int, float]]:
        """Batch işleme ile Levenshtein fuzzy matching"""
        matches = []
        
        # Progress bar için
        total_batches = (len(left_df) + self.batch_size - 1) // self.batch_size
        progress_bar = st.progress(0)
        
        for i, batch_start in enumerate(range(0, len(left_df), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(left_df))
            batch_left = left_df.iloc[batch_start:batch_end]
            
            for _, row1 in batch_left.iterrows():
                val1 = row1[combo_col]
                if not val1:
                    continue
                    
                for _, row2 in right_df.iterrows():
                    val2 = row2[combo_col]
                    if not val2:
                        continue
                        
                    sim = ratio(val1, val2)
                    if sim >= self.similarity_threshold:
                        matches.append((row1.name, row2.name, sim))
            
            # Progress güncelle
            progress_bar.progress((i + 1) / total_batches)
            
            # Bellek temizliği
            if i % 10 == 0:
                gc.collect()
        
        progress_bar.empty()
        return matches

    def _fuzzy_match(self, df1: pd.DataFrame, df2: pd.DataFrame, combo_col: str,
                    cols: List[str], file1_name: str, file2_name: str,
                    original_df1: pd.DataFrame, original_df2: pd.DataFrame) -> List[Dict]:
        """Fuzzy match işlemi"""
        results = []
        
        # Exact match'te bulunanları çıkar
        left_only = df1[~df1[combo_col].isin(df2[combo_col])].copy()
        right_only = df2[~df2[combo_col].isin(df1[combo_col])].copy()
        
        if left_only.empty or right_only.empty:
            return results
        
        st.info(f"Fuzzy matching başlıyor: {len(left_only)} x {len(right_only)} karşılaştırma")
        
        # Büyük veri seti kontrolü
        total_comparisons = len(left_only) * len(right_only)
        use_tfidf = (self.use_tfidf_for_large_data and 
                    total_comparisons > self.large_data_threshold)
        
        if use_tfidf:
            st.info("🚀 Büyük veri seti tespit edildi. TF-IDF algoritması kullanılıyor...")
            
            left_values = left_only[combo_col].tolist()
            right_values = right_only[combo_col].tolist()
            left_indices = left_only.index.tolist()
            right_indices = right_only.index.tolist()
            
            matches = self._fuzzy_match_tfidf(left_values, right_values, left_indices, right_indices)
        else:
            st.info("📝 Levenshtein algoritması kullanılıyor...")
            matches = self._fuzzy_match_levenshtein_batch(left_only, right_only, combo_col)
        
        # Sonuçları formatla
        for match in matches:
            left_idx, right_idx, similarity = match
            
            result_row = {}
            
            # Data Source bilgileri
            if "Data Source" in original_df1.columns:
                result_row[f"Data Source_{file1_name}"] = original_df1.loc[left_idx, "Data Source"]
            if "Data Source" in original_df2.columns:
                result_row[f"Data Source_{file2_name}"] = original_df2.loc[right_idx, "Data Source"]
            
            # Sütun değerleri
            for col in cols:
                result_row[f"{col}_{file1_name}"] = df1.loc[left_idx, col] if left_idx in df1.index else ""
                result_row[f"{col}_{file2_name}"] = df2.loc[right_idx, col] if right_idx in df2.index else ""
            
            # Eşleşme bilgileri
            result_row["Match_Type"] = "Fuzzy Match (TF-IDF)" if use_tfidf else "Fuzzy Match (Levenshtein)"
            result_row["Similarity"] = round(float(similarity), 3)
            result_row["Source"] = f"{file1_name} vs {file2_name}"
            
            results.append(result_row)
        
        return results

    def _match_two_files(self, df1: pd.DataFrame, df2: pd.DataFrame, cols: List[str], 
                        file1_name: str, file2_name: str) -> pd.DataFrame:
        """İki dosyayı karşılaştır"""
        try:
            start_time = time.time()
            
            # Kopya oluştur
            df1 = df1.copy()
            df2 = df2.copy()
            original_df1 = df1.copy()
            original_df2 = df2.copy()
            
            # Sütunları birleştir
            combo_col = "_combo_col"
            df1 = self._combine_columns(df1, cols, combo_col)
            df2 = self._combine_columns(df2, cols, combo_col)
            
            # Boş değerleri filtrele
            df1 = df1[df1[combo_col] != ""]
            df2 = df2[df2[combo_col] != ""]
            
            if df1.empty or df2.empty:
                st.warning(f"Karşılaştırma için yeterli veri yok: {file1_name} vs {file2_name}")
                return pd.DataFrame()
            
            st.info(f"📊 Karşılaştırılıyor: {file1_name} ({len(df1)} kayıt) vs {file2_name} ({len(df2)} kayıt)")
            
            results = []
            
            # 1. Exact Match
            with st.spinner("🎯 Exact matching..."):
                exact_results = self._exact_match(
                    df1, df2, combo_col, cols, file1_name, file2_name, original_df1, original_df2
                )
                results.extend(exact_results)
                st.success(f"✅ Exact match tamamlandı: {len(exact_results)} eşleşme")
            
            # 2. Fuzzy Match
            with st.spinner("🔍 Fuzzy matching..."):
                fuzzy_results = self._fuzzy_match(
                    df1, df2, combo_col, cols, file1_name, file2_name, original_df1, original_df2
                )
                results.extend(fuzzy_results)
                st.success(f"✅ Fuzzy match tamamlandı: {len(fuzzy_results)} eşleşme")
            
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
    st.header("📁 Optimize Edilmiş Çoklu Excel Dosyası Karşılaştırıcı")
    
    # Yan panel - Ayarlar
    with st.sidebar:
        st.subheader("⚙️ Ayarlar")
        similarity = st.slider("Benzerlik Eşiği (%):", 0, 100, 70)
        batch_size = st.number_input("Batch Boyutu:", min_value=100, max_value=5000, value=1000)
        use_tfidf = st.checkbox("Büyük veri için TF-IDF kullan", value=True)
        
        st.subheader("📝 Bilgi")
        st.info("""
        **Optimizasyonlar:**
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
        if not common_cols:
            st.error("❌ Tüm dosyalarda ortak bir sütun bulunamadı!")
            return
            
        # Sütun seçimi
        selected_columns = st.multiselect(
            "Birden fazla sütun seçebilirsiniz:", 
            sorted(common_cols),
            help="Seçilen sütunlar birleştirilerek karşılaştırma yapılacak"
        )

        if selected_columns:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("📁 Dosya Sayısı", len(dataframes))
                st.metric("📊 Toplam Kayıt", f"{total_rows:,}")
            
            with col2:
                combinations_count = len(list(combinations(dataframes.keys(), 2)))
                st.metric("🔄 Karşılaştırma Sayısı", combinations_count)
                st.metric("📋 Seçilen Sütun", len(selected_columns))

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
                        
                        # Sütun sıralaması
                        ordered_columns = []
                        
                        # Data Source kolonları
                        for file_name in [f.replace(".xlsx", "").replace(".xls", "") for f in dataframes.keys()]:
                            data_source_col = f"Data Source_{file_name}"
                            if data_source_col in final_df.columns:
                                ordered_columns.append(data_source_col)
                        
                        # Seçilen sütunlar
                        for col in selected_columns:
                            for file_name in [f.replace(".xlsx", "").replace(".xls", "") for f in dataframes.keys()]:
                                col_name = f"{col}_{file_name}"
                                if col_name in final_df.columns:
                                    ordered_columns.append(col_name)
                        
                        # Eşleşme bilgileri
                        ordered_columns.extend(["Match_Type", "Similarity", "Source"])
                        
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
                        
                        # Veri önizlemesi
                        st.subheader("📋 Sonuç Önizlemesi")
                        st.dataframe(final_df.head(100))
                        
                        # İndirme butonu
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                            final_df.to_excel(tmp.name, index=False)
                            with open(tmp.name, "rb") as f:
                                st.download_button(
                                    "📥 Tüm Eşleşmeleri İndir (.xlsx)",
                                    f,
                                    file_name=f"eslesmeler_{int(time.time())}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    else:
                        st.warning("😔 Hiç eşleşme bulunamadı.")
        else:
            st.warning("⚠️ Lütfen karşılaştırmak için en az bir sütun seçin.")


if __name__ == "__main__":
    main()