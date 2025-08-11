import sqlite3
from io import BytesIO

def save_to_sqlite(df, db_name="fuar_data.db", table_name="katilimcilar"):
    import sqlite3

    # List veya dict tiplerini stringe çevir
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()