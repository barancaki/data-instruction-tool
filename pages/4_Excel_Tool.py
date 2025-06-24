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

        combo_col = "_combo_col"
        df1 = self._combine_columns(df1, cols, combo_col)
        df2 = self._combine_columns(df2, cols, combo_col)

        df1 = df1[df1[combo_col] != ""]
        df2 = df2[df2[combo_col] != ""]

        # Exact Match
        merged = pd.merge(df1, df2, on=combo_col, suffixes=(f'_{file1_name}', f'_{file2_name}'), how="inner")
        merged["Match_Type"] = "Exact Match"
        merged["Source"] = f"{file1_name} vs {file2_name}"

        # Fuzzy Match
        fuzzy_matches = []
        left_only = df1[~df1[combo_col].isin(df2[combo_col])]
        right_only = df2[~df2[combo_col].isin(df1[combo_col])]

        for _, row1 in left_only.iterrows():
            val1 = row1[combo_col]
            for _, row2 in right_only.iterrows():
                val2 = row2[combo_col]
                sim = ratio(val1, val2)
                if val1 and val2 and sim >= self.similarity_threshold:
                    match_data = {
                        f"Match_Val_{file1_name}": val1,
                        f"Match_Val_{file2_name}": val2,
                        "Match_Type": "Fuzzy Match",
                        "Similarity": sim,
                        "Source": f"{file1_name} vs {file2_name}"
                    }
                    match_data.update({f"{k}_{file1_name}": v for k, v in row1.items() if k != combo_col})
                    match_data.update({f"{k}_{file2_name}": v for k, v in row2.items() if k != combo_col})
                    fuzzy_matches.append(match_data)

        if fuzzy_matches:
            fuzzy_df = pd.DataFrame(fuzzy_matches)
            merged = pd.concat([merged, fuzzy_df], ignore_index=True)

        return merged


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