import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os


def load_local_data():
    path = os.path.join('data', 'amazon_sales_dataset.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def clear_data(path2df=None, df=None):
    if df is None:
        if path2df is None:
            raise ValueError('Either path2df or df must be provided')
        df = pd.read_csv(path2df)

    # Небольшая предобработка для amazon_sales_dataset
    df = df.copy()
    # Преобразуем даты
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month

    # Удалим строки с пропусками в ключевых столбцах
    df = df.dropna(subset=['price', 'quantity_sold', 'total_revenue'])

    # Категориальные столбцы для кодирования
    cat_columns = ['product_category', 'customer_region', 'payment_method']

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns].astype(str))

    # Выбираем признаки и цель
    features = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count',
                'product_category', 'customer_region', 'payment_method', 'order_year', 'order_month']

    df_out = df[features + ['total_revenue']].reset_index(drop=True)
    df_out.to_csv('df_clear.csv', index=False)
    return df_out


if __name__ == "__main__":
    df = load_local_data()
    df_clean = clear_data(df=df)
    print(f"df_clear.csv saved with shape {df_clean.shape}")