import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("titanic.csv")

print("Первые 10 строк:")
print(df.head(10))
print("\nИнформация о данных:")
print(df.info())

print("\nПропущенные значения:")
print(df.isnull().sum())

df_filled = df.copy()

for col in df_filled.columns:
    if df_filled[col].isnull().any():
        if df_filled[col].dtype in ['int64', 'float64']:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        else:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

print("\nПосле заполнения пропусков:")
print(df_filled.isnull().sum())

if 'Score' in df_filled.columns:
    scaler_minmax = MinMaxScaler()
    df_filled['Score_normalized'] = scaler_minmax.fit_transform(df_filled[['Score']])
    print("Score нормализован")

if 'SubmissionCount' in df_filled.columns:
    scaler_standard = StandardScaler()
    df_filled['SubmissionCount_normalized'] = scaler_standard.fit_transform(df_filled[['SubmissionCount']])
    print("SubmissionCount нормализован")

if 'TeamName' in df_filled.columns:
    df_final = pd.get_dummies(df_filled, columns=['TeamName'], drop_first=True)
    print("One-Hot Encoding применен для TeamName")
else:
    df_final = df_filled
print(df_final.head(10))
print(f"Столбцов ДО преобразования: {len(df_filled.columns)}")
print(f"Итоговое количество столбцов: {len(df_final.columns)}")

df_final.to_csv('titanic_processed.csv', index=False)

print("\nОбработка завершена! Файлы сохранены.")
