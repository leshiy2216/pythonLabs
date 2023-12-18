import argparse
import pandas as pd
from PIL import Image


def analyze_dataset(cat_annotation_file, dog_annotation_file):
    cat_df = pd.read_csv(cat_annotation_file)
    dog_df = pd.read_csv(dog_annotation_file)

    cat_df['class'] = 'cat'
    dog_df['class'] = 'dog'

    cat_df.rename(columns={'the text name of the class': 'class_cat'}, inplace=True)
    dog_df.rename(columns={'the text name of the class': 'class_dog'}, inplace=True)

    df = pd.concat([cat_df, dog_df], ignore_index=True)
    df['label'] = df['class'].astype('category').cat.codes

    df['height'] = df['The absolute path'].apply(lambda x: Image.open(x).height)
    df['width'] = df['The absolute path'].apply(lambda x: Image.open(x).width)
    df['channels'] = df['The absolute path'].apply(lambda x: len(Image.open(x).split()))

    df.rename(columns={'The absolute path': 'absolute_path'}, inplace=True)
    df = df[['class', 'absolute_path', 'label', 'height', 'width', 'channels']]

    df['pixel_count'] = df.apply(lambda row: Image.open(row['absolute_path']).size[0] * Image.open(row['absolute_path']).size[1], axis=1)

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

    return df

def filter_by_class(df, class_label):
    filtered_df = df[df['class'] == class_label].reset_index(drop=True)
    return filtered_df

def filter_by_size_and_class(df, class_label, max_width, max_height):
    filtered_df = df[(df['class'] == class_label) & (df['width'] <= max_width) & (df['height'] <= max_height)].reset_index(drop=True)
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Dataset Analysis and Filtering')
    parser.add_argument('cat_annotation_file', type=str, help='Path to the cat annotation file')
    parser.add_argument('dog_annotation_file', type=str, help='Path to the dog annotation file')
    parser.add_argument('--filter-class', type=str, help='Filter by class label')
    parser.add_argument('--filter-width', type=int, help='Max width for filtering')
    parser.add_argument('--filter-height', type=int, help='Max height for filtering')
    args = parser.parse_args()

    df = analyze_dataset(args.cat_annotation_file, args.dog_annotation_file)

    if args.filter_class:
        filtered_df = filter_by_class(df, class_label=args.filter_class)
        print("\nDataFrame filtered by class:")
        print(filtered_df)

    if args.filter_width and args.filter_height:
        filtered_size_df = filter_by_size_and_class(df, class_label=args.filter_class,
                                                     max_width=args.filter_width, max_height=args.filter_height)
        print("\nDataFrame filtered by size and class:")
        print(filtered_size_df)

    grouped_df = df.groupby('class')['pixel_count'].agg(['min', 'max', 'mean']).reset_index()
    print("\nStatistical information for pixel count:")
    print(grouped_df)