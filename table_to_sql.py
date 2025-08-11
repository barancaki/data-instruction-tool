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

def create_mysql_dump_from_sqlite(sqlite_file="fuar_data.db", dump_file="fuar_data.sql"):
    """SQLite veritabanını MySQL uyumlu SQL dump dosyasına çevirir"""
    conn = sqlite3.connect(sqlite_file)
    with open(dump_file, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            # MySQL uyumu için bazı ifadeleri dönüştür
            line = line.replace("AUTOINCREMENT", "AUTO_INCREMENT")
            line = line.replace("INTEGER PRIMARY KEY", "INT PRIMARY KEY AUTO_INCREMENT")
            f.write(f"{line}\n")
    conn.close()