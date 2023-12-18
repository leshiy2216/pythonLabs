import pandas as pd
from PIL import Image

cat_annotation_file = "C:/Users/User/Desktop/testing/dataset/cat_annotation.csv"
dog_annotation_file = "C:/Users/User/Desktop/testing/dataset/dog_annotation.csv"

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

print(df)