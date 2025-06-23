import streamlit as st
import pandas as pd
import numpy as np
from Levenshtein import ratio
import tempfile
import os
import gc

st.set_page_config(page_title="Excel Veri Eşleştirici", layout="wide")


class DataMatcher:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold / 100

    def _read_excel_file(self, file) -> pd.DataFrame:
        try:
            df = pd.read_excel(file)
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception as e:
            st.error(f"{file.name} dosyası okunamadı: {str(e)}")
            raise

    def _clean_value(self, value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _is_column_match(self, target: str, column: str) -> bool:
        target = target.strip().lower()
        column = column.strip().lower()
        if target == column:
            return True
        return ratio(target, column) >= self.similarity_threshold

    def _find_matching_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, target_columns: list) -> list:
        matching_columns = []
        for target in target_columns:
            matches_df1 = [col for col in df1.columns if self._is_column_match(target, col)]
            matches_df2 = [col for col in df2.columns if self._is_column_match(target, col)]
            for col1 in matches_df1:
                for col2 in matches_df2:
                    matching_columns.append((col1, col2))
        return matching_columns

    def _find_matching_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, matching_columns: list) -> pd.DataFrame:
        if not matching_columns:
            return pd.DataFrame()

        matches = []

        for col1, col2 in matching_columns:
            df1_clean = df1.copy()
            df2_clean = df2.copy()
            df1_clean[col1] = df1_clean[col1].apply(self._clean_value)
            df2_clean[col2] = df2_clean[col2].apply(self._clean_value)

            df1_clean = df1_clean[df1_clean[col1].str.strip() != '']
            df2_clean = df2_clean[df2_clean[col2].str.strip() != '']

            if df1_clean.empty or df2_clean.empty:
                continue

            merged = pd.merge(df1_clean, df2_clean, left_on=col1, right_on=col2, suffixes=('_file1', '_file2'), how='inner')
            if not merged.empty:
                merged['Match_Type'] = 'Exact Match'
                merged['Matched_Column_File1'] = col1
                merged['Matched_Column_File2'] = col2
                matches.append(merged)

            if self.similarity_threshold < 1.0:
                left_only = df1_clean[~df1_clean[col1].isin(df2_clean[col2])]
                right_only = df2_clean[~df2_clean[col2].isin(df1_clean[col1])]

                fuzzy_matches = []
                for _, row1 in left_only.iterrows():
                    val1 = self._clean_value(row1[col1])
                    for _, row2 in right_only.iterrows():
                        val2 = self._clean_value(row2[col2])
                        if val1 and val2 and ratio(val1, val2) >= self.similarity_threshold:
                            match_row = pd.concat([row1, row2])
                            match_row['Match_Type'] = 'Fuzzy Match'
                            match_row['Matched_Column_File1'] = col1
                            match_row['Matched_Column_File2'] = col2
                            match_row['Similarity'] = ratio(val1, val2)
                            fuzzy_matches.append(match_row)

                if fuzzy_matches:
                    fuzzy_df = pd.DataFrame(fuzzy_matches)
                    matches.append(fuzzy_df)

        if matches:
            result = pd.concat(matches, ignore_index=True).drop_duplicates()
            return result
        return pd.DataFrame()

    def find_matches(self, files: list, target_columns: list) -> str:
        if len(files) < 2:
            raise ValueError("En az iki dosya yüklenmeli")
        if not target_columns:
            raise ValueError("Hedef sütun adı girilmeli")

        result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name

        with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
            summary_data = []

            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    file1, file2 = files[i], files[j]
                    df1 = self._read_excel_file(file1)
                    df2 = self._read_excel_file(file2)

                    matching_columns = self._find_matching_columns(df1, df2, target_columns)
                    if matching_columns:
                        matches = self._find_matching_rows(df1, df2, matching_columns)
                        if not matches.empty:
                            sheet_name = f"Match_{i+1}_{j+1}"
                            matches.to_excel(writer, sheet_name=sheet_name, index=False)
                            summary_data.append({
                                "Dosya 1": file1.name,
                                "Dosya 2": file2.name,
                                "Eşleşen Sütunlar": len(matching_columns),
                                "Eşleşen Satırlar": len(matches),
                                "Sayfa": sheet_name
                            })
                    gc.collect()

            summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame({'Durum': ['Hiçbir eşleşme bulunamadı.']})
            summary_df.to_excel(writer, sheet_name='Özet', index=False)

        return result_file


# ----- STREAMLIT UI -----
st.title("📊 Excel Dosya Karşılaştırıcı ve Veri Eşleştirici")

uploaded_files = st.file_uploader("Birden fazla Excel dosyası yükleyin (.xlsx):", type=['xlsx'], accept_multiple_files=True)

columns_input = st.text_input("Hedef sütun adlarını girin (virgülle ayırın):", placeholder="Firma Adı, Telefon, Email")

similarity = st.slider("Benzerlik eşiği (%):", 0, 100, 70)

if st.button("Eşleştirmeyi Başlat"):
    if uploaded_files and columns_input:
        with st.spinner("Eşleştiriliyor..."):
            matcher = DataMatcher(similarity_threshold=similarity)
            try:
                columns = [col.strip() for col in columns_input.split(",") if col.strip()]
                result_path = matcher.find_matches(uploaded_files, columns)
                with open(result_path, "rb") as f:
                    st.success("✅ Eşleştirme tamamlandı!")
                    st.download_button("📥 Sonucu İndir (.xlsx)", f, file_name="eslesen_veriler.xlsx")
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
    else:
        st.warning("Lütfen hem dosya yükleyin hem de hedef sütun girin.")