import pandas as pd
import numpy as np

#Load dataset from data folder
df = pd.read_csv('data/synthetic_student_data.csv')



#1.Selection and Indexing

#Label based indexing
loc_prac = df.loc[df['Name'] == 'Alice']
print("LOC Example:\n", loc_prac)

#Position based indexing
iloc_prac = df.iloc[:5]
print("\nILOC Example:\n", iloc_prac)

#Boolean Indexing
bool_indexing = df[df['Score'] > 85]
print("\nBoolean Indexing (Score > 85):\n", bool_indexing)

#Access single values
at_prac = df.at[0, 'Name']
iat_prac = df.iat[0, 3]
print(f"\n.at example: {at_prac}, .iat example: {iat_prac}")



#2.Missing Data

print("\nMissing values:\n", df['Age'].isnull())

print("\nMissing values:\n", df.isnull().sum())
print("\nNon-missing values:\n", df.notnull().sum())

#Fill missing values
df_filled = df.fillna({'Age': df['Age'].mean(), 'Score': df['Score'].mean()})
print("\nFilled missing values:\n", df_filled)

df_ffilled = df.fillna(method= 'ffill', inplace=True)
print("\nFilled missing values:\n", df_ffilled)

#Drop missing values
df_dropped = df.dropna()
print("\nDropped rows with missing data:\n", df_dropped)

#Interpolate missing data
df_interpolated = df.interpolate()
print("\nInterpolated Data:\n", df_interpolated)

#3.Removing Duplicates

duplicates = df[df.duplicated()]
print("\nDuplicate rows:\n", duplicates)

df_no_duplicates = df.drop_duplicates()
print("\nAfter removing duplicates:\n", df_no_duplicates)


#4.Data Transformation

#Renaming columns
df_renamed = df.rename(columns={'Score': 'ExamScore'})
print("\nRenamed Column:\n", df_renamed.head())

#Mapping values
df['Grade'] = df['Score'].map(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C' if pd.notnull(x) else np.nan)
print("\nGrade Mapping:\n", df[['Name', 'Score', 'Grade']].head())

#Applying functions
df['AgePlusTen'] = df['Age'].apply(lambda x: x + 10 if pd.notnull(x) else x)
print("\nApplied Function (Age + 10):\n", df[['Name', 'Age', 'AgePlusTen']].head())

#String operations
df['Name_Upper'] = df['Name'].str.upper()
print("\nString Operation (Uppercase Names):\n", df[['Name', 'Name_Upper']].head())

#5.Merging and Concatenation

#Creating a dummy dataframe to merge
df2 = df[['Name', 'City']].copy()
df2['City'] = df2['City'].str.upper()

merged = pd.merge(df, df2, on='Name', suffixes=('', '_Upper'), how='inner')
print("\nMerged Data:\n", merged.head())

concatenated = pd.concat([df.head(2), df.tail(2)], axis=0)
print("\nConcatenated Data:\n", concatenated)

#6.Grouping and Aggregating

grouped = df.groupby('City')['Score'].agg(['mean', 'count'])
print("\nGrouped Data (Score by City):\n", grouped)

#7.Pivot Tables

pivot = pd.pivot_table(df, values='Score', index='City', columns='Name', aggfunc='mean')
print("\nPivot Table:\n", pivot)

#8.Reshaping Data

stacked = df[['Name', 'City']].stack()
print("\nStacked Data:\n", stacked.head())

unstacked = stacked.unstack()
print("\nUnstacked Data:\n", unstacked.head())

melted = pd.melt(df, id_vars=['Name'], value_vars=['Age', 'Score'])
print("\nMelted Data:\n", melted.head())

#9.Sorting and Ranking

sorted_df = df.sort_values(by='Score', ascending=False)
print("\nSorted by Score:\n", sorted_df[['Name', 'Score']])

df['ScoreRank'] = df['Score'].rank(ascending=False)
print("\nScore Ranking:\n", df[['Name', 'Score', 'ScoreRank']])

#10.Time Series Data

#Add synthetic Date column
df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
df['Date'] = pd.to_datetime(df['Date'])

#Resample weekly average
resampled = df.set_index('Date').resample('W').mean(numeric_only=True)
print("\nWeekly Resampled Data:\n", resampled.head())



#Vectorization


df['Score_with_bonus'] = df['Score'] + 5
df['Is_Senior'] = df['Age'] > 30

#Safe check before normalization
if df['Score'].notnull().all():
    df['Score_normalized'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min())
else:
    print("Warning: Missing or non-numeric values in 'Score' column. Normalization skipped.")

#Display vectorized results

print("\nData after vectorized operations:")
cols_to_show = ['Name', 'Score', 'Score_with_bonus', 'Is_Senior']
if 'Score_normalized' in df.columns:
    cols_to_show.append('Score_normalized')
print(df[cols_to_show].head())



#Categorization

#Before: City as object (string)
print("Before dtype:", df['City'].dtype)

#Convert to categorical
df['City'] = df['City'].astype('category')

#After: City as category
print("After dtype:", df['City'].dtype)
