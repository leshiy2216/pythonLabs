import argparse
import pandas as pd
from PIL import Image

def analyze_dataset(cat_annotation_file, dog_annotation_file):
    # Чтение данных из аннотационных файлов
    cat_df = pd.read_csv(cat_annotation_file)
    dog_df = pd.read_csv(dog_annotation_file)

    # Добавление столбцов 'class' и числовых меток
    cat_df['class'] = 'cat'
    dog_df['class'] = 'dog'

    cat_df.rename(columns={'the text name of the class': 'class_cat'}, inplace=True)
    dog_df.rename(columns={'the text name of the class': 'class_dog'}, inplace=True)

    df = pd.concat([cat_df, dog_df], ignore_index=True)
    df['label'] = df['class'].astype('category').cat.codes

    # Добавление столбцов с высотой, шириной и глубиной изображения
    df['height'] = df['The absolute path'].apply(lambda x: Image.open(x).height)
    df['width'] = df['The absolute path'].apply(lambda x: Image.open(x).width)
    df['channels'] = df['The absolute path'].apply(lambda x: len(Image.open(x).split()))

    df.rename(columns={'The absolute path': 'absolute_path'}, inplace=True)
    df = df[['class', 'absolute_path', 'label', 'height', 'width', 'channels']]

    print(df)

    print("\nStatistical information for image sizes:")
    print(df[['width', 'height', 'channels']].describe())

    print("\nStatistical information for class labels:")
    print(df['label'].value_counts())

    class_balance = df['label'].value_counts().tolist()
    is_balanced = all(balance == class_balance[0] for balance in class_balance[1:])

    if is_balanced:
        print("\nThe dataset is balanced.")
    else:
        print("\nThe dataset is not balanced.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset with cat and dog annotations')
    parser.add_argument('cat_annotation_file', type=str, help='Path to the cat annotation file')
    parser.add_argument('dog_annotation_file', type=str, help='Path to the dog annotation file')

    args = parser.parse_args()

    analyze_dataset(args.cat_annotation_file, args.dog_annotation_file)