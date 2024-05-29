import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('QueryResults.csv',names=['Date','Tag','Posts'],header=0)

print(df.head())
print(df.shape)
print(df.columns)
print(df.tail())
print(df.groupby('Tag').sum().sort_values('Posts',ascending=False))
df.groupby('Tag').count()
pd.to_datetime(df.Date[1])
result=df.pivot_table(index='Date',columns='Tag',values='Posts')
result.fillna(0,inplace=True)
df.set_index('day')
df.reset_index(inplace=True)

#########################################
plt.figure(figsize=(16,10)) # make chart larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.ylim(0, 35000)
 
plt.plot(result.index, result.java)
plt.plot(result.index, result.python) 

####################################
plt.figure(figsize=(16,10)) # make chart larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.ylim(0, 35000)

for column in result.columns:
    plt.plot(result.index, result[column], 
             linewidth=3, label=result[column].name)
    
plt.legend(fontsize=16) 
##############################################
roll_df = result.rolling(window=6).mean()
plt.figure(figsize=(16,10)) # make chart larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.ylim(0, 35000)

for column in roll_df.columns:
    plt.plot(roll_df.index, roll_df[column], 
             linewidth=3, label=roll_df[column].name)
    
plt.legend(fontsize=16) 
##############################################

print(result.isna().values.any())

test_df = pd.DataFrame({'Age': ['Young', 'Young', 'Young', 'Young', 'Old', 'Old', 'Old', 'Old'],
                        'Actor': ['Jack', 'Arnold', 'Keanu', 'Sylvester', 'Jack', 'Arnold', 'Keanu', 'Sylvester'],
                        'Power': [100, 80, 25, 50, 99, 75, 5, 30]})

pivot_df=test_df.pivot(index="Age",columns="Actor",values="Power")

##################################################
# Define the input CSV file path
csv_file = "weather_data.csv"  # Replace with your actual file path

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Create boolean masks for Monday and sunny condition
is_monday = df['day'] == "Monday"
is_sunny = df['condition'] == "Sunny"

# Update temp column based on combined mask
df.loc[is_monday & is_sunny, 'temp'] = 49  # Update temp where both conditions are True

# Save the DataFrame back to a CSV file (without index)
df.to_csv("final_weather.csv", index=False)
##############################################
df.nunique()
df.is_trans.value_counts()
df.groupby('is_trans').count() 
df.sort_values('Tag').head()
df.sort_values('Tag',ascending=False).head()

sets_by_year=df.groupby("year").count()
sets_by_year["set_num"].head()
plt.plot(sets_by_year.index[:-1],sets_by_year.set_num[:-1])
#################################################

ax1=plt.gca()
ax2=ax1.twinx()
ax1.plot(themes_by_year.index[:-2],themes_by_year.nr_themes[:-2])
ax2.plot(sets_by_year.index[:-1],sets_by_year.set_num[:-1])
##############################################################
parts_per_set=sets_df.groupby('year').agg({'num_parts':pd.Series.mean})
########################################################################


def convert_people_cell(cell):
    if cell=="n.a.":
        return 'Sam Walton'
    return cell

def convert_price_cell(cell):
    if cell=="n.a.":
        return 50
    return cell
    
df = pd.read_excel("stock_data.xlsx","Sheet1", converters= {
        'people': convert_people_cell,
        'price': convert_price_cell
    })
###################################################
with pd.ExcelWriter('stocks_weather.xlsx') as writer:
    df_stocks.to_excel(writer, sheet_name="stocks")
    df_weather.to_excel(writer, sheet_name="weather")
################################################
###############FILLNA############################
new_df = df.fillna({
        'temperature': 0,
        'windspeed': 0,
        'event': 'No Event'
    })
new_df = df.fillna(method="ffill")
new_df = df.fillna(method="bfill")
new_df = df.fillna(method="bfill", axis="columns") # axis is either "index" or "columns"
new_df = df.fillna(method="ffill",limit=1)
new_df = df.interpolate()
new_df = df.interpolate(method="time") 
new_df = df.dropna()
new_df = df.dropna(how='all')
new_df = df.dropna(thresh=1)
dt = pd.date_range("01-01-2017","01-11-2017")
idx = pd.DatetimeIndex(dt)
df.reindex(idx)
###################################################
##############REPLACE##############################
new_df = df.replace(-99999, value=np.NaN)
new_df = df.replace(to_replace=[-99999,-88888], value=0)
new_df = df.replace({
        'temperature': -99999,
        'windspeed': -99999,
        'event': '0'
    }, np.nan)
new_df = df.replace({
        -99999: np.nan,
        'no event': 'Sunny',
    })
new_df = df.replace({'temperature': '[A-Za-z]', 'windspeed': '[a-z]'},'', regex=True)
df = pd.DataFrame({
    'score': ['exceptional','average', 'good', 'poor', 'average', 'exceptional'],
    'student': ['rob', 'maya', 'parthiv', 'tom', 'julian', 'erica']
})
df.replace(['poor', 'average', 'good', 'exceptional'], [1,2,3,4])
#####################GROUP BY###################################
g = df.groupby("city")
for city, data in g:
    print("city:",city)
    print("\n")
    print("data:",data)
g.get_group('mumbai')
g.max()
g.mean()
########################CONCAT#####################################
df = pd.concat([india_weather, us_weather])
df = pd.concat([india_weather, us_weather], ignore_index=True)
df = pd.concat([india_weather, us_weather], keys=["india", "us"])
df.loc["us"]

temperature_df = pd.DataFrame({
    "city": ["mumbai","delhi","banglore"],
    "temperature": [32,45,30],
}, index=[0,1,2])
windspeed_df = pd.DataFrame({
    "city": ["delhi","mumbai"],
    "windspeed": [7,12],
}, index=[1,0])
df = pd.concat([temperature_df,windspeed_df],axis=1)
s = pd.Series(["Humid","Dry","Rain"], name="event")
df = pd.concat([temperature_df,s],axis=1)
##########################JOIN###################
df_final=pd.merge(df1,df2,how="left")
df_final=pd.merge(df1,df2,how="right")
df_final=pd.merge(df1,df2,how="inner")
df_final=pd.merge(df1,df2,how="outer")
df3=pd.merge(df1,df2,on="city",how="outer",indicator=True)
df3= pd.merge(df1,df2,on="city",how="outer", suffixes=('_first','_second'))
df1.join(df2,lsuffix='_l', rsuffix='_r')


