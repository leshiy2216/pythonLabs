import argparse
import pandas as pd

def filter_dataframe(df, target_class):
    filtered_df = df[df['class'] == target_class].copy()
    return filtered_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter DataFrame by class label')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file with DataFrame')
    parser.add_argument('target_class', type=str, help='Class label to filter by')

    args = parser.parse_args()

    # Чтение DataFrame из CSV-файла
    df = pd.read_csv(args.csv_path)

    # Получение отфильтрованного DataFrame
    filtered_df = filter_dataframe(df, args.target_class)

    # Вывод результата
    print(filtered_df)
