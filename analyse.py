import pandas as pd
import sys

# Повећавамо ширину приказа да се лепо виде дугачки текстови
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Учитавамо главни фајл са резултатима
try:
    df_results = pd.read_csv("response/class1.csv", delimiter=';')
except FileNotFoundError:
    print("Greška: Fajl 'response/class1.csv' nije pronađen.")
    print("Molimo Vas da prvo pokrenete 'python rag.py' da biste generisali rezultate.")
    sys.exit() # Прекини извршавање

# Штампамо резултате на читљив начин
print("--- АНАЛИЗА РЕЗУЛТАТА (class1.csv) ---")

for index, row in df_results.iterrows():
    print(f"\n==========================================")
    print(f"             УПИТ {index + 1} / {len(df_results)}")
    print(f"==========================================\n")
    print(f"ТВРДЊА:\n{row['Sentence']}\n")
    print("------------------------------------------")
    print(f"ПРОНАЂЕН КОНТЕКСТ:\n{row['Relevant_Context']}\n")

    print("***************** МОДЕЛ А (Cohere) *****************")
    print(f"Одговор: {row['Response_Cohere']}")
    print(f"Брзина: {row['Latency_Cohere']:.2f} секунди\n")

    print("***************** МОДЕЛ Б (Haiku) *****************")
    print(f"Одговор: {row['Response_Haiku']}")
    print(f"Брзина: {row['Latency_Haiku']:.2f} секунди\n")

print("\n--- КРАЈ АНАЛИЗЕ ---")

# (Опционо) Можемо додати и приказ class0.csv
try:
    df_class0 = pd.read_csv("response/class0.csv", delimiter=';')
    print("\n\n--- ПРОВЕРА class0.csv ---")
    print("Тврдње класификоване као '0' (не-чињеничне):\n")
    for sentence in df_class0['Sentence']:
        print(f"- {sentence}")
except FileNotFoundError:
    print("\nInfo: Fajl 'response/class0.csv' nije pronađen.")