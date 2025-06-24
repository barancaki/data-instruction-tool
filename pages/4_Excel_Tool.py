import streamlit as st
import pandas as pd
from Levenshtein import ratio
import tempfile
from itertools import combinations


class DataMatcher:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold / 100

    def _clean_value(self, value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _combine_columns(self, df: pd.DataFrame, columns: list, new_col: str) -> pd.DataFrame:
        df[new_col] = df[columns].astype(str).agg(" ".join, axis=1)
        df[new_col] = df[new_col].apply(self._clean_value)
        return df

    def _match_two_files(self, df1, df2, cols, file1_name, file2_name):
        df1 = df1.copy()
        df2 = df2.copy()
        
        # Orijinal dataframe'leri sakla (Exhibition Name için)
        original_df1 = df1.copy()
        original_df2 = df2.copy()

        combo_col = "_combo_col"
        df1 = self._combine_columns(df1, cols, combo_col)
        df2 = self._combine_columns(df2, cols, combo_col)

        df1 = df1[df1[combo_col] != ""]
        df2 = df2[df2[combo_col] != ""]

        results = []

        # Exact Match
        merged = pd.merge(df1, df2, on=combo_col, suffixes=(f'_{file1_name}', f'_{file2_name}'), how="inner")
        
        for _, row in merged.iterrows():
            result_row = {}
            
            # Exhibition Name kolonunu kontrol et ve ekle (varsa)
            if "Exhibition Name" in original_df1.columns:
                # Orijinal index'i kullanarak Exhibition Name'i al
                df1_idx = df1[df1[combo_col] == row[combo_col]].index[0]
                result_row[f"Exhibition Name_{file1_name}"] = original_df1.loc[df1_idx, "Exhibition Name"]
            if "Exhibition Name" in original_df2.columns:
                # Orijinal index'i kullanarak Exhibition Name'i al  
                df2_idx = df2[df2[combo_col] == row[combo_col]].index[0]
                result_row[f"Exhibition Name_{file2_name}"] = original_df2.loc[df2_idx, "Exhibition Name"]
            
            # Seçilen sütunları ekle (her iki dosya için)
            for col in cols:
                result_row[f"{col}_{file1_name}"] = row.get(f"{col}_{file1_name}", "")
                result_row[f"{col}_{file2_name}"] = row.get(f"{col}_{file2_name}", "")
            
            # Eşleşme bilgilerini ekle
            result_row["Match_Type"] = "Exact Match"
            result_row["Similarity"] = 1.0
            result_row["Source"] = f"{file1_name} vs {file2_name}"
            
            results.append(result_row)

        # Fuzzy Match
        left_only = df1[~df1[combo_col].isin(df2[combo_col])]
        right_only = df2[~df2[combo_col].isin(df1[combo_col])]

        for _, row1 in left_only.iterrows():
            val1 = row1[combo_col]
            for _, row2 in right_only.iterrows():
                val2 = row2[combo_col]
                sim = ratio(val1, val2)
                if val1 and val2 and sim >= self.similarity_threshold:
                    result_row = {}
                    
                    # Exhibition Name kolonunu kontrol et ve ekle (varsa)
                    if "Exhibition Name" in original_df1.columns:
                        result_row[f"Exhibition Name_{file1_name}"] = original_df1.loc[row1.name, "Exhibition Name"]
                    if "Exhibition Name" in original_df2.columns:
                        result_row[f"Exhibition Name_{file2_name}"] = original_df2.loc[row2.name, "Exhibition Name"]
                    
                    # Seçilen sütunları ekle (her iki dosya için)
                    for col in cols:
                        result_row[f"{col}_{file1_name}"] = row1.get(col, "")
                        result_row[f"{col}_{file2_name}"] = row2.get(col, "")
                    
                    # Eşleşme bilgilerini ekle
                    result_row["Match_Type"] = "Fuzzy Match"
                    result_row["Similarity"] = round(sim, 3)
                    result_row["Source"] = f"{file1_name} vs {file2_name}"
                    
                    results.append(result_row)

        return pd.DataFrame(results) if results else pd.DataFrame()


# Streamlit Arayüzü
st.header("📁 Çoklu Excel Dosyası Karşılaştırıcı")

uploaded_files = st.file_uploader("Birden fazla Excel dosyasını buraya yükleyin", type=["xlsx"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    st.subheader("📄 Dosya Önizlemeleri")

    dataframes = {}
    for file in uploaded_files:
        df = pd.read_excel(file)
        dataframes[file.name] = df
        st.write(f"**{file.name}:**")
        st.dataframe(df.head())

    st.subheader("🔗 Karşılaştırılacak Ortak Sütunları Seçin")
    common_cols = list(set.intersection(*(set(df.columns) for df in dataframes.values())))
    if not common_cols:
        st.error("Tüm dosyalarda ortak bir sütun bulunamadı!")
    else:
        selected_columns = st.multiselect("Birden fazla sütun seçebilirsiniz:", sorted(common_cols))

        if selected_columns:
            similarity = st.slider("Benzerlik Eşiği (%):", 0, 100, 70)

            if st.button("🧠 Karşılaştırmaya Başla"):
                with st.spinner("Tüm dosyalar karşılaştırılıyor..."):
                    matcher = DataMatcher(similarity_threshold=similarity)
                    results = []

                    for (file1, df1), (file2, df2) in combinations(dataframes.items(), 2):
                        result = matcher._match_two_files(df1, df2, selected_columns, file1.replace(".xlsx", ""), file2.replace(".xlsx", ""))
                        if not result.empty:
                            results.append(result)

                    if results:
                        final_df = pd.concat(results, ignore_index=True)
                        
                        # Sütun sıralaması: Exhibition Name + Seçilen sütunlar + eşleşme bilgileri
                        ordered_columns = []
                        
                        # Önce Exhibition Name kolonunu ekle (varsa)
                        for file_name in [f.replace(".xlsx", "") for f in dataframes.keys()]:
                            exhibition_col = f"Exhibition Name_{file_name}"
                            if exhibition_col in final_df.columns:
                                ordered_columns.append(exhibition_col)
                        
                        # Sonra seçilen sütunları ekle
                        for col in selected_columns:
                            # Her dosya için seçilen sütunu ekle
                            for file_name in [f.replace(".xlsx", "") for f in dataframes.keys()]:
                                col_name = f"{col}_{file_name}"
                                if col_name in final_df.columns:
                                    ordered_columns.append(col_name)
                        
                        # Eşleşme bilgilerini ekle
                        ordered_columns.extend(["Match_Type", "Similarity", "Source"])
                        
                        # Sadece istenen sütunları seç ve sırala
                        final_df = final_df[ordered_columns]
                        
                        st.success(f"Eşleşmeler tamamlandı! Toplam: {len(final_df)} kayıt")
                        st.dataframe(final_df.head(50))

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                            final_df.to_excel(tmp.name, index=False)
                            with open(tmp.name, "rb") as f:
                                st.download_button(
                                    "📥 Eşleşen Verileri İndir",
                                    f,
                                    file_name="tum_eslesmeler.xlsx"
                                )
                    else:
                        st.warning("Hiç eşleşme bulunamadı.")
        else:
            st.warning("Lütfen karşılaştırmak için en az bir sütun seçin.")