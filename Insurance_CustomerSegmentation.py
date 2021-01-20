#--------------------------------------------------------------------------------------------------------------------------------------
# 1 - LOAD LYBRARIES
#--------------------------------------------------------------------------------------------------------------------------------------

from scipy.cluster.vq import kmeans, vq, kmeans2
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns; sns.set()
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler #StandardScaler gives us the z-score
from sompy.sompy import SOMFactory 
from sompy.visualization.plot_tools import plot_hex_map
from sompy.visualization.mapview import View2DPacked
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView
import logging
from sklearn.cluster import KMeans
import kmodes
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from pylab import rcParams
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split 
from dtreeplt import dtreeplt
import graphviz 
from matplotlib.pylab import rcParams
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score



#--------------------------------------------------------------------------------------------------------------------------------------
# 2 - LOAD THE DATA
#--------------------------------------------------------------------------------------------------------------------------------------

#Load data set
conn = sqlite3.connect('insurance.db')
cursor = conn.cursor()

query = """
select * from lob;
"""

Consumption_df = pd.read_sql_query(query,conn).reset_index(drop=True)
Consumption_df = Consumption_df.drop(['index'], axis=1)



query = '''
select * from engage;
'''
Engage_df = pd.read_sql_query(query,conn)
Engage_df = Engage_df.drop(['index'], axis=1)

#merge dataframes
df = Engage_df.merge(Consumption_df, how = 'inner', left_on='Customer Identity', 
                      right_on='Customer Identity')

#Renaming columns
df = df.rename({'Customer Identity':'ID','First Policy´s Year':'First_Policy','Brithday Year':'Birthday',
                 'Educational Degree':'Education','Gross Monthly Salary':'Salary','Geographic Living Area':'Area',
                 'Has Children (Y=1)':'Children','Customer Monetary Value':'CMV','Claims Rate':'Claims',
                 'Premiums in LOB: Motor':'Motor','Premiums in LOB: Household':'Household',
                 'Premiums in LOB: Health':'Health','Premiums in LOB:  Life':'Life',
                 'Premiums in LOB: Work Compensations':'Work_Compensation'}, axis = 'columns')

#--------------------------------------------------------------------------------------------------------------------------------------
# 3 - DEFINE FUNCTIONS TO USE LATTER
#--------------------------------------------------------------------------------------------------------------------------------------

#3.1 - Outliers detection:
#Outlier detection with IQR rule
#Let's build a function that returns a bloxpot and the number of outliers:
def outliers(column_name, dataframe):
    data = np.array(dataframe[column_name])
    q1, q3= np.percentile(data,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr)
    n_outliers = dataframe[column_name].loc[dataframe[column_name] > upper_bound ].count() + dataframe[column_name].loc[dataframe[column_name] < lower_bound ].count()
    plt.figure(figsize=(12, 8))
    plt.xlabel(column_name)
    sns.boxplot(data)
    plt.title('Outliers of ' + column_name)
    print('Number of Outliers:')
    print(n_outliers)
#--------------------------------------------------------------------------------------------------------------------------------------

#3.2 - Scatter plot
# Scatter plot of the different variables
def scatter_plot(x,y, dataframe):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x, y=y, data=dataframe)

#--------------------------------------------------------------------------------------------------------------------------------------

#3.3 - Silhouette Plot
def silhouette_plot(dataframe):
    X = dataframe
    range_n_clusters = [3, 4, 5, 6, 7, 8]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(12, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init = 5, max_iter = 200)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for {} clusters.".format(str(n_clusters)))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------
# 4 - EXPLORATORY PHASE: 
#--------------------------------------------------------------------------------------------------------------------------------------
#Asses customers behaviour 
#Count how many costumers per year
df = df[df.First_Policy < 2000]
marketing = df.groupby(['First_Policy'])['ID'].count().reset_index()
 
x = marketing['First_Policy']
y = marketing['ID']  
# plotting the points  
plt.plot(x, y)  
# naming the x axis 
plt.xlabel('Years') 
# naming the y axis 
plt.ylabel('Number of costumers') 
# giving a title to my graph 
plt.title('Number of costumers per year') 
# function to show the plot
plt.show()
#From the graph we can understand that from the last few years the Insurance Company has been losing 
#customers.

#------------------------------------------------------------------------------------------------------------------------------------

#SOME DATA DESCRIPTION
df.shape
#(10265, 14)

df.describe()

df['ID'].describe()
#count    10265.000000
#mean      5147.073356
#std       2972.355940
#min          1.000000
#25%       2572.000000
#50%       5146.000000
#75%       7721.000000
#max      10296.000000

df['First_Policy'].describe()
#count    10265.000000
#mean      1986.017048
#std          6.612110
#min       1974.000000
#25%       1980.000000
#50%       1986.000000
#75%       1992.000000
#max       1998.000000

df['Birthday'].describe()
#count    10251.000000
#mean      1968.020388
#std         19.716262
#min       1028.000000
#25%       1953.000000
#50%       1968.000000
#75%       1983.000000
#max       2001.000000

df['Education'].describe()
#count           10263
#unique              4
#top       3 - BSc/MSc
#freq             4792
#This analysis don't make sense since "Education" is a categorical variable. 
#Is not possible to take any interpretation from this.

df.Education.value_counts()
    #3 - BSc/MSc        4792
    #2 - High School    3506
    #1 - Basic          1270
    #4 - PhD             695
#This is the kind of interpretation we are able to do.

df['Salary'].describe()
#count    10231.000000
#mean      2506.073991
#std       1158.098393
#min        333.000000
#25%       1705.000000
#50%       2500.000000
#75%       3290.500000
#max      55215.000000

df['Area'].describe()
#mean         2.711934
#std          1.266173
#min          1.000000
#25%          1.000000
#50%          3.000000
#75%          4.000000
#max          4.000000
#This analysis don't make sense since "Area" is a categorical variable. 
#Is not possible to take any interpretation from this.

df.Area.value_counts()
#4.0    4140
#1.0    3035
#3.0    2063
#2.0    1027
#This is the kind of interpretation we are able to do.

df['Children'].describe()
#count    10252.000000
#mean         0.706984
#std          0.455168
#min          0.000000
#25%          0.000000
#50%          1.000000
#75%          1.000000
#max          1.000000
#This analysis also don't make sense since "Children" is a categorical variable. 
#Is not possible to take any interpretation from this.

df.Children.value_counts()
    #1.0    7248
    #0.0    3004
#This is the kind of interpretation we are able to do.
    
df['CMV'].describe()
#count     10265.000000
#mean        177.829121
#std        1948.709262
#min     -165680.420000
#25%          -9.440000
#50%         187.030000
#75%         399.400000
#max       11875.890000

df['Claims'].describe()
#count    10265.000000
#mean         0.742982
#std          2.921306
#min          0.000000
#25%          0.390000
#50%          0.720000
#75%          0.980000
#max        256.200000

df['Motor'].describe()
#count    10231.000000
#mean       300.361471
#std        212.097772
#min         -4.110000
#25%        190.535000
#50%        298.500000
#75%        408.190000
#max      11604.420000

df['Household'].describe()
#count    10265.000000
#mean       210.607491
#std        352.964293
#min        -75.000000
#25%         49.450000
#50%        132.800000
#75%        290.600000
#max      25048.800000

df['Health'].describe()
#count    10222.000000
#mean       171.615830
#std        296.825949
#min         -2.110000
#25%        111.800000
#50%        162.810000
#75%        219.820000
#max      28272.000000

df['Life'].describe()
#count    10161.000000
#mean        41.878931
#std         47.509040
#min         -7.000000
#25%          9.890000
#50%         25.560000
#75%         57.790000
#max        398.300000

df['Work_Compensation'].describe()
#count    10180.000000
#mean        41.308985
#std         51.556450
#min        -12.000000
#25%         10.670000
#50%         25.670000
#75%         56.790000
#max       1988.700000



#------------------------------------------------------------------------------------------------------------------------------------
df['full_count'] = df.apply(lambda x: x.count(), axis=1)
#We have twelve clientes that doesn't have insurances ('Full_count' column equals 10). Also, from here
#we can conclude that the acquisition cost is 25€ since they present a CMV value of -25€.
#Put this, we decided to drop these rows:
df = df[df.full_count > 10 ]

#------------------------------------------------------------------------------------------------------------------------------------
#INCONSISTENT DATA
#How many clients have a First Policy Year before they were even born? There are 1997 customers
df[df.First_Policy < df.Birthday].count()

#How many customer were born after 2016? Any.
df[df.Birthday > 2016 ].count()

#How many customer were born before 1932? Just 1.
# Note that the life expectancy at birth is over 84 years (1932), considering 2016 - https://observador.pt/2019/05/31/esperanca-media-de-vida-sobe-para-os-8080-anos-segundo-o-ine/
df[df.Birthday < 1932 ].count() 

#Birthday Column - Due to a relevant amount of observations with errors, we decided to drop this column:
df = df.drop(columns=['Birthday'])

#How many first policy years after the year of 2016? Just 1 (year of 53784).
df[df.First_Policy > 2016 ].count()

#Drop the row which First Policy Year occurs after the year of 2016:
clean_df = df[df.First_Policy < 53784 ]

#There are any other living areas? No
clean_df[df.Area > 4 ].count()
clean_df[df.Area < 0 ].count()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# DETECTING AND REPLACING MISSING VALUES
#Count the number of missing values in each column
clean_df.isnull().sum()
    #Salary - 34
    #Education - 2
    #Children-13
    #Motor - 22
    #Health - 31
    #Life - 92
    #Work_Compensation - 73
    
#Salary Column
df_NAN_Salary = clean_df.loc[clean_df.Salary.isnull()] 

imputed_Salary_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
imputed_Salary_mean = imputed_Salary_mean.fit(clean_df[['Salary']]) 
clean_df['Salary']=imputed_Salary_mean.transform(clean_df[['Salary']]).ravel()


#Education Column
df_NAN_Education = clean_df.loc[clean_df.Education.isnull()] 
    #Costumer_1 (index = 1707) with a 3827€ salary. Costumer_2 (index = 6560) with a 2503.36€ salary. 
    #They both live in area 4.
    #We can relate the salary with the education level and replace the missing value.

pd.pivot_table(clean_df,
               values='Salary',
               index= 'Area',
               columns= 'Education',
               aggfunc= ['mean'],
               margins=True)
    #According to the living area number 4, average salary of a PhD person 
    #is 2646.048507€, so we should assign the first costumer to category 
    #"4-PhD"
clean_df.loc[1707,['Education']] = clean_df.loc[1707,['Education']].fillna('4 - PhD')

    #According to the living area number 4, average salary of a High School
    #person is 2571.37€, so we should assign the first costumer to 
    #category "2-High School"
clean_df.loc[6560,['Education']] = clean_df.loc[6560,['Education']].fillna('2 - High School')


#Children Column - Given that the only rows with Nan values in this column 
#represent clients that live in area number 2, we are going to replace the 
#Nan Values in this columns by the mode of Childrens in area 2:
df_NAN_Children = clean_df.loc[clean_df.Children.isnull()] 
clean_df['Children'] = clean_df.Children.fillna(clean_df[clean_df['Area']==2].groupby('Area').Children.apply(lambda x: x.mode()).iloc[0])


# Premiums Columns - On the premiums columns we will assume that the NaN values
#represent the customers that don't have that type of insurance, and due to 
#that we will fill null values with zero:
clean_df['Motor'] = clean_df['Motor'].fillna(0)
clean_df['Health'] = clean_df['Health'].fillna(0)
clean_df['Life'] = clean_df['Life'].fillna(0)
clean_df['Work_Compensation'] = clean_df['Work_Compensation'].fillna(0)

print(clean_df.isnull().sum())
#ID                   0
#First_Policy         0
#Education            0
#Salary               0
#Area                 0
#Children             0
#CMV                  0
#Claims               0
#Motor                0
#Household            0
#Health               0
#Life                 0
#Work_Compensation    0
#full_count           0
#As we can see, there are no missing values

#Drop the "Full_count" column, because we don't need it anymore:
clean_df = clean_df.drop(columns=['full_count'])

#-------------------------------------------------------------------------------------------------------------------------------------------------
#CREATE NEW VARIABLES
#Calculate the seniority of a customer and his annual profit:

#To calculate the annual profit of each customer we assume that the acquisition cost equals 25 (as seen 
#above):
clean_df['Client_Years'] = 2016 - clean_df['First_Policy']
clean_df['Client_Profit'] = ( clean_df['CMV'] + 25) / clean_df['Client_Years']

#Join all "Premium" columns in order to facilitate interpretation of data
clean_df['Total_Premium'] = clean_df[['Motor', 'Household','Health','Life','Work_Compensation']].sum(axis=1)

#-------------------------------------------------------------------------------------------------------------------------------------------------
#DETECTING OUTLIERS    
outliers('Salary', clean_df)#2
outliers_Salary = clean_df[clean_df.Salary >= 30000 ] #We transfering all outliers to other dataframe. 
clean_df = clean_df[clean_df.Salary < 30000 ] #0

outliers('Motor', clean_df)#6
outliers_Motor = clean_df[clean_df.Motor >= 1000 ] #We transfering all outliers to other dataframe.
clean_df = clean_df[clean_df.Motor < 1000 ] #0

outliers('Household', clean_df)#632
outliers_Household = clean_df[clean_df.Household >= 2000 ] #We transfering some outliers to other dataframe.
clean_df = clean_df[clean_df.Household < 2000 ] #Now we have 628 outliers in Household

outliers('Health', clean_df)#24
outliers_Health = clean_df[clean_df.Health >= 500 ] #We transfering one outlier to other dataframe.
clean_df = clean_df[clean_df.Health < 500 ] #Now we have 22 outliers in Health variable. 

outliers('Life', clean_df)#649 
outliers_Life = clean_df[clean_df.Life >= 370 ] #We transfering some outliers to other dataframe.
clean_df = clean_df[clean_df.Life < 370] #Now we have 648 outliers in Life variable.

outliers('Work_Compensation', clean_df)#627
outliers_Work_Compensation = clean_df[clean_df.Work_Compensation >= 400 ] #We transfering some outliers to other dataframe.
clean_df = clean_df[clean_df.Work_Compensation < 400 ] #Now we have 623 outliers in Work_Compensation variable.

outliers('CMV', clean_df)#105 
outliers_CMV = clean_df[clean_df.CMV <= -500] 
outliers_CMV_1 = clean_df[clean_df.CMV >= 1500]
final_outliers_CMV = outliers_CMV.append(outliers_CMV_1) #We transfering some outliers to other dataframe.
clean_df = clean_df[ (clean_df.CMV > -500) & (clean_df.CMV < 1500) ] #Now we have 84 outliers in CMV variable.

outliers('Claims', clean_df)#1
outliers_Claims = clean_df[clean_df.Claims >= 2] #We transfering one outlier to other dataframe.
clean_df = clean_df[clean_df.Claims < 2 ] #We have removed the only outlier in Claims variable.

outliers('Total_Premium', clean_df)#609
outliers_Total_Premium = clean_df[clean_df.Total_Premium >= 1900] 
outliers_Total_Premium_1 = clean_df[clean_df.Total_Premium <= 60] 
final_outliers_Total_Premium = outliers_Total_Premium.append(outliers_Total_Premium_1)
clean_df = clean_df[(clean_df.Total_Premium < 1900) & (clean_df.Total_Premium > 60) ] #Now we have 608 outliers in Total_Premium variable.

outliers('Client_Years', clean_df)#0

outliers('Client_Profit', clean_df)#129
outliers_Client_Profit = clean_df[clean_df.Client_Profit >= 70] 
clean_df = clean_df[clean_df.Client_Profit < 70 ] #Now we have 128 outliers in Client_Profit variable.

clean_df.shape
#(10207, 16)
#-------------------------------------------------------------------------------------------------------------------------------------------------
#DIVIDE OUR DATASET IN THREE PARTS 
continuous_engage_variables = clean_df[['First_Policy','Salary','CMV', 'Claims','Client_Years', 'Client_Profit']]
categorical_engage_variables = clean_df[['Education','Area', 'Children']]
consumption_variables= clean_df[['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Total_Premium']]

#--------------------------------------------------------------------------------------------------------------------------------------


#CHECK FOR CORRELATION IN ORDER TO CHOSE WHICH VARIABLES TO USE
    #Before computing correlation matrix we must standardize the data (numerical variables)
#z-score: standard Scaler (good to bring the outliers near the population). The mean will be 0 and the 
#standard deviation will be 1.

scaler = StandardScaler()

consumption_variables_Norm = scaler.fit_transform(consumption_variables)
consumption_variables_Norm = pd.DataFrame(consumption_variables_Norm, columns = consumption_variables.columns)

continuous_engage_variables_Norm = scaler.fit_transform(continuous_engage_variables)
continuous_engage_variables_Norm = pd.DataFrame(continuous_engage_variables_Norm, columns = continuous_engage_variables.columns)

final_df_Norm = pd.DataFrame(pd.concat([pd.DataFrame(consumption_variables_Norm), continuous_engage_variables_Norm], axis = 1),
                        columns = ['First_Policy','Salary','CMV', 'Claims','Client_Years', 'Client_Profit', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Total_Premium'])

#Cheks the distribution of the variables:
fig, axes = plt.subplots(len(final_df_Norm.columns), figsize=(12, 48))
for col, axis in zip(final_df_Norm.columns, axes):
    final_df_Norm.hist(column = col, bins = 100, ax=axis)

#Compute the correlation:
correlation = final_df_Norm.corr().round(2) #<-- Here we are only correlating the standardized variables
# Generate a mask for the upper triangle
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220,10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,square=True, linewidths=.5, annot=True)
    # CMV has a strong and negative correlation with Claims (-93%), as expected. 
    # First Policy has a perfect negative correlation with and Client_Years (-1).
    # Client_Profit and CMV has a strong and positive correlation (95%).
    # Client_Profit and Claims has a strong and negative correlation (89%).
    # Motor variable has a strong and negative correlation with:
        # - Household variable (-65%);
        # - Health variable (-62%);
        # - Life variable (-65%);
        # - Work Compensation variable (-64%);
        # - Total Premium variable (-62%);
    # Total Premium has a strong and positive correlation with Household (99%);

#Below, the scatter plots were produced as a tool to visualize the correlation   
scatter_plot('CMV', 'Claims', final_df_Norm) #There is a negative relationship

scatter_plot('First_Policy', 'Client_Years', final_df_Norm) #There is a perfect negative relationship

scatter_plot('Client_Profit', 'CMV', final_df_Norm) #There is a strong and negative correlation

scatter_plot('Client_Profit', 'Claims', final_df_Norm) #There is a strong and negative correlation

scatter_plot('Motor', 'Total_Premium', final_df_Norm) #There is a weak negative relationship    

scatter_plot('Motor', 'Work_Compensation', final_df_Norm) #There is a weak negative relationship

scatter_plot('Motor', 'Life', final_df_Norm) #There is a weak negative relationship

scatter_plot('Motor', 'Health', final_df_Norm) #There is a weak negative relationship

scatter_plot('Motor', 'Household', final_df_Norm) #There is a weak negative relationship   

scatter_plot('Total_Premium', 'Household', final_df_Norm) #There is a strong negative relationship        
     
#According to this correlation matrix we choose to drop Claims, Total_Premium, First_Policy and Client_Profit variables.
clean_df = clean_df.drop(columns=['Total_Premium', 'First_Policy','Claims','Client_Profit'])
continuous_engage_variables = continuous_engage_variables.drop(columns = ['First_Policy','Claims', 'Client_Profit'])
continuous_engage_variables_Norm = continuous_engage_variables_Norm.drop(columns = ['First_Policy','Claims', 'Client_Profit'])
consumption_variables= consumption_variables.drop(columns =['Total_Premium'])
consumption_variables_Norm = consumption_variables_Norm.drop(columns = ['Total_Premium'])
final_df_Norm = final_df_Norm.drop(columns=['Total_Premium', 'Claims', 'First_Policy','Client_Profit'])

#--------------------------------------------------------------------------------------------------------------------------------------
# 5 - CLUSTER ANALYSIS
#--------------------------------------------------------------------------------------------------------------------------------------
#5.1. HIERARCHICAL CLUSTERING - consumption variables

#1. Select data that we want:
variables = clean_df.loc[:,['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']]

#2. Normalize the variables selected:
CA_Norm = scaler.fit_transform(variables)
CA_Norm = pd.DataFrame(CA_Norm, columns = variables.columns)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

Z = linkage(CA_Norm, method ='ward')
#here we are using ward method because tipically is the one who works better

hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k']) #this give us the color
 
dendrogram(Z,
           truncate_mode='lastp',
           p=10,
           orientation = 'top',
           leaf_rotation=45.,
           leaf_font_size=10.,
           show_contracted=True,
           show_leaf_counts=True, color_threshold=75, above_threshold_color='k')
plt.title('Consumption Variables: Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.axhline(y=85)
plt.show()
#As we can see we should use 3 clusters.
# Blue color - 1 cluster
# Purple color - 1 cluster
# Yellow color - 1 cluster

k = 3

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

#Replace the test with proper data
my_HC = Hclustering.fit(CA_Norm)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']

Affinity = pd.DataFrame(pd.concat([pd.DataFrame(CA_Norm), my_labels], axis = 1),
                        columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Labels'])


#Calculate the centroid using a groupby. The labels are what define the centroids. The centroid are 
#the average of all the data points according to x-axis and y-axis.
to_revert = Affinity.groupby(['Labels'])['Motor', 'Household', 'Health', 'Life', 'Work_Compensation'].mean()

X=to_revert
Consumo_Hirarquical = pd.DataFrame(scaler.inverse_transform(X),
                            columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation'])

#--------------------------------------------------------------------------------------------------------------------------------------------------
#5.1. HIERARCHICAL CLUSTERING - engage variables

#1. Select data that we want:
engage_var = clean_df.loc[:,['Salary', 'CMV', 'Client_Years']]

#2. Normalize the variables selected:
EN_Norm = scaler.fit_transform(engage_var)
EN_Norm = pd.DataFrame(EN_Norm, columns = engage_var.columns)

plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

Z = linkage(EN_Norm, method ='ward')
#here we are using ward method because tipically is the one who works better

hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k']) #this give us the color
 
dendrogram(Z,
           truncate_mode='lastp',
           p=10,
           orientation = 'top',
           leaf_rotation=45.,
           leaf_font_size=10.,
           show_contracted=True,
           show_leaf_counts=True, color_threshold=70, above_threshold_color='k')
plt.title('Engage Variables: Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.axhline(y=70)
plt.show()
#As we can see we should use 4 clusters.
# Blue color - 1 cluster
# Purple color - 1 cluster
# Yellow color - 1 cluster
# Green color - 1 cluster

k = 4

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

#Replace the test with proper data
my_HC = Hclustering.fit(EN_Norm)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']

Affinity2 = pd.DataFrame(pd.concat([pd.DataFrame(EN_Norm), my_labels], axis = 1),
                        columns = ['Salary', 'CMV', 'Client_Years', 'Labels'])

#Calculate the centroid using a groupby. 
to_revert2 = Affinity2.groupby(['Labels'])['Salary', 'CMV', 'Client_Years'].mean()

X=to_revert2
Eng_Hierarquical = pd.DataFrame(scaler.inverse_transform(X),
                            columns = ['Salary', 'CMV', 'Client Years'])
#--------------------------------------------------------------------------------------------------------------------------------------------------
#5.2. CLUSTERING BY K-MODES AND K-MEANS

#5.2.1.  - K-Modes of the categorial variables

#1. Select data that we want:
categ_var = clean_df.loc[:,['Education', 'Area', 'Children']]

#elbow graph - Huang
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Huang", n_init = 1, verbose=1)
    kmode.fit_predict(categ_var)
    cost.append(kmode.cost_)
y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)
plt.xlabel('N.º of clusters')
plt.ylabel('Cost')
plt.title('Elbow Graph - For Haung Init')

#elbow graph - Cao
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(categ_var)
    cost.append(kmode.cost_)
y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)
plt.xlabel('N.º of clusters')
plt.ylabel('Cost')
plt.title('Elbow Graph - For Cao Init')

#2. Since the Huang is the one with a lower cost we are going to compute the Kmodes with init Huang:
km = KModes(n_clusters=3, init='huang', n_init=50, verbose=1)

#Attribute the cluster to each cliente:
clusters = km.fit_predict(categ_var) # Compute cluster centers and predict cluster index for each sample.

# Print the cluster centroids:
print(km.cluster_centroids_)
cat_centroids = pd.DataFrame(km.cluster_centroids_,
                             columns = ['Education', 'Area', 'Children'])

#Count the number of customers per cluster
unique, counts = np.unique(km.labels_, return_counts=True)

cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Label','Number'])


#Huang has a cost of 8633 
cat_centroids_huang = pd.concat([cat_centroids, cat_counts], axis = 1)


clean_df['Cat_cluster'] = clusters
#----------------------------------------------------------------------------------------------------------------------

#5.2.2. - K-Means of continuos variables that characterize the clients:

#1. Select the variables:
continuos = clean_df.loc[:,['Salary','CMV', 'Client_Years']]

#2. Normalize the variables selected:
scaler = StandardScaler()
cont_norm2 = scaler.fit_transform(continuos)
cont_norm2 = pd.DataFrame(cont_norm2, columns = continuos.columns)

#3. Check the  best number of clusters - Elbow graph:
L = []

for i in range(1,10):
    kmeans2 = KMeans(n_clusters = i,
                    random_state = 0,
                    n_init = 4,
                    max_iter = 200).fit(cont_norm2)
    L.append(kmeans2.inertia_)

plt.figure(figsize=(10, 8))
plt.title('Elbow Graph for Engage Variables')
plt.xlabel('N.º of Clusters')
plt.ylabel('Inertia')
plt.plot(range(1,10), L)

#4. Plot the silhouette graphs
silhouette_plot(cont_norm2)

# Given the results above we decided that the best number of clusters for engage variables is 4
n_clusters = 4

#5. Compute K-means:

kmeans2 = KMeans(n_clusters= n_clusters, 
                random_state=0,
                n_init = 4,
                max_iter = 200).fit(cont_norm2)

#6. Calculate the custers centroids:
my_clusters2 = kmeans2.cluster_centers_

getting_labels_km_2 = pd.DataFrame(pd.concat([cont_norm2, pd.DataFrame(kmeans2.labels_)],axis=1))
getting_labels_km_2.columns = ['Salary', 'CMV', 'Client_Years','Labels']

my_clusters2 = pd.DataFrame(scaler.inverse_transform(X = my_clusters2), 
                           columns = cont_norm2.columns)

#7. Count the number of clients per cluster
unique2, counts2 = np.unique(kmeans2.labels_, return_counts=True)

cont_counts = pd.DataFrame(np.asarray((unique2, counts2)).T, columns = ['Label','Number'])


#8. Assign the corresponding cluster to each client
clean_df['Cont_cluster']=kmeans2.fit_predict(cont_norm2)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                      
#5.2.3 - K-Means of the consume variables:

#1. Select the variables:
consumo = clean_df.loc[:,['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']]

#2. Normalize the variables selected:
scaler = StandardScaler()
cons_norm = scaler.fit_transform(consumo)
cons_norm = pd.DataFrame(cons_norm, columns = consumo.columns)


#3 - Check the  best number of clusters - Elbow graph:
L = []

for i in range(1,10):
    kmeans = KMeans(n_clusters = i,
                    random_state = 0,
                    n_init = 4,
                    max_iter = 200).fit(cons_norm)
    L.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.title('Elbow Graph for Consumption Variables')
plt.xlabel('N.º of Clusters')
plt.ylabel('Inertia')
plt.plot(range(1,10), L)

#4. Plot the silhouette graphs
silhouette_plot(cons_norm)

# Given the results above we decided that the best number of clusters for engage variables is 3
n_clusters =3

#5. Compute K-means:
kmeans = KMeans(n_clusters= n_clusters, 
                random_state=0,
                n_init = 4,
                max_iter = 200).fit(cons_norm)

#6. Calculate the custers centroids:
my_clusters = kmeans.cluster_centers_

getting_labels_km = pd.DataFrame(pd.concat([cons_norm, pd.DataFrame(kmeans.labels_)],axis=1))
getting_labels_km.columns = ['Motor', 'Household', 'Health','Life', 'Work_Compensation', 'Labels']


my_clusters = pd.DataFrame(scaler.inverse_transform(X = my_clusters), 
                           columns = cons_norm.columns)

#7. Count the number of clients per cluster
unique3, counts3 = np.unique(kmeans.labels_, return_counts=True)

cons_counts = pd.DataFrame(np.asarray((unique3, counts3)).T, columns = ['Label','Number'])



#8. Assign the corresponding cluster to each client
clean_df['Cons_cluster'] = kmeans.fit_predict(cons_norm)

#-----------------------------------------------------------------------------------------------------------------
#Groupby clusters
count_clusters1 = clean_df.groupby(['Cont_cluster','Cat_cluster'])['ID'].count().reset_index()

count_clusters2 = clean_df.groupby(['Cons_cluster','Cont_cluster'])['ID'].count().reset_index()

count_clusters3 = clean_df.groupby(['Cons_cluster','Cont_cluster', 'Cat_cluster'])['ID'].count().reset_index()


#------------------------------------------------------------------------------------------------------------------------------------

# 5.3. SOM
    #Self-organizing maps (SOM) have been recognized as a powerful tool in data exploratoration, 
    #especially for the tasks of clustering on high dimensional data.

    #The U*-Matrix, as defined here, combines the information of local distances and local 
    #density. The calculation of the U*-Matrix is such, that inner cluster distances are 
    #depicted with lower heights than on the corresponding U-Matrix. In the extreme case, 
    #where density is maximal, U*-heights vanish. 

    # 5.3.1. For numerical engage variables 
X = cont_norm2.values

names = ['Salary', 'CMV', 'Client_Years']
sm_eng = SOMFactory().build(data = X,
               mapsize=(10,10), 
               normalization = 'var',
               initialization = 'random', 
               component_names = names,
               lattice= 'hexa', 
               training = 'seq') 

#we have used 4 clusters and this is confirmed with SOM.

sm_eng.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=100)

final_clusters_sm_eng = pd.DataFrame(sm_eng._data, columns = ['Salary', 'CMV', 'Client_Years'])
                              
#BMU = Best matching unit is like the nearest neighbor or centroid
my_labels_sm_eng = pd.DataFrame(sm_eng._bmu[0])
    
final_clusters_sm_eng = pd.concat([final_clusters_sm_eng,my_labels_sm_eng], axis = 1)

final_clusters_sm_eng.columns = ['Salary', 'CMV', 'Client_Years', 'Labels']

"""
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)
        :param normalization: normalizer object calculator. Options are:
            - var
        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)
        :param lattice: type of lattice. Options are:
            - rect
            - hexa
        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
"""
#SOM as three key ouputs:
    # - U-Matrices
    # - Component planes
    # - Hit Plots.

# - Component planes
    # Each component plane shows the values of one variable in each map unit using color-coding. 
    # This gives the possibility to visually examine every cell (each cell corresponding to each 
    # data point.

view2D_eng  = View2DPacked(10,10,"", text_size=7)
view2D_eng.show(sm_eng, col_sz=5, what = 'codebook', which_dim = 'all', cmap=plt.cm.gray_r) 
plt.show()
   

# - Distance matrix in 2D
    # The 2D surface plot of distance matrix use color to indicate the average distance to 
    # neighboring map units. They use a landscape metaphor to represent the density, shape, and size or
    # volume of clusters. The user has the flexibility to manipulate the coordinates 
    # and the view in 2D space. 

view2D_eng  = View2D(10,10,"", text_size=7)
view2D_eng.show(sm_eng, col_sz=5, what = 'codebook', cmap=plt.cm.gray_r)
plt.show()


#- U-Matrix (Unified Distance Matrix)
    # The unified distance matrix or U-matrix is a representation of the Self-Organizing Map that 
    # visualizes the distances between the network neurons or units. It contains the distances from 
    # each unit center to all of its neighbors. The neurons of the SOM network are represented here 
    # by hexagonal cells. The distance between the adjacent neurons is calculated and presented with 
    # different colorings.
vhts_eng  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts_eng.show(sm_eng, anotate=True, onlyzeros=False, labelsize=10, cmap=plt.cm.gray_r, logaritmic=False)
#As we can see we cannot properly segmentate customers forming clusters.

# K-Means Clustering
sm_eng.cluster(4)
hits_eng  = HitMapView(10,10,"Clustering",text_size=7)
a_eng=hits_eng.show(sm_eng, labelsize=12)
#If we apply k-means over the neurons the result is not so great. 


    # 5.3.2. For numerical consumption variables 
X_1 = cons_norm.values


names = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']
sm_cons = SOMFactory().build(data = X_1,
               mapsize=(10,10), 
               normalization = 'var',
               initialization = 'random', 
               component_names = names,
               lattice= 'hexa', 
               training = 'seq') 

#we have used 3 clusters and this is confirmed with SOM.

sm_cons.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=100)


final_clusters_sm_cons = pd.DataFrame(sm_cons._data, columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation'])
                              
#BMU = Best matching unit is like the nearest neighbor or centroid
my_labels_sm_cons = pd.DataFrame(sm_cons._bmu[0])
    
final_clusters_sm_cons = pd.concat([final_clusters_sm_cons,my_labels_sm_cons], axis = 1)

final_clusters_sm_cons.columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Labels']

"""
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)
        :param normalization: normalizer object calculator. Options are:
            - var
        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)
        :param lattice: type of lattice. Options are:
            - rect
            - hexa
        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
"""
# Component planes
view2D_cons  = View2DPacked(10,10,"", text_size=7)
view2D_cons.show(sm_cons, col_sz=5, what = 'codebook', which_dim = 'all', cmap=plt.cm.gray_r) 
plt.show()

# Distance matrix in 2D
view2D_cons  = View2D(10,10,"", text_size=7)
view2D_cons.show(sm_cons, col_sz=5, what = 'codebook', cmap=plt.cm.gray_r)
plt.show()

# U-Matrix (Unified Distance Matrix)
vhts_cons  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts_cons.show(sm_cons, anotate=True, onlyzeros=False, labelsize=10,  cmap=plt.cm.gray_r, logaritmic=False)
#As we can see we cannot properly segmentate customers forming clusters.

# K-Means Clustering
sm_eng.cluster(3)
hits_cons  = HitMapView(10,10,"Clustering",text_size=7)
a_cons=hits_cons.show(sm_eng, labelsize=12)
#If we apply k-means over the neurons the result is not so great.

#------------------------------------------------------------------------------------------------------------------------------------

# 5.4. DB SCAN
    #DBSCAN groups together points that are close to each other based on a distance measurement 
    #(usually Euclidean distance) and a minimum number of points. It also marks as outliers the 
    #points that are in low-density regions.

# 5.4.1. For numerical engage variables 
    #We have used PCA
db = DBSCAN(eps= 0.5, min_samples= 10).fit(cont_norm2)

labels = db.labels_

# This gives the number of clusters in labels, ignoring noise if present.
n_clusters_db_eng = len(set(labels)) - (1 if -1 in labels else 0) #noise is -1

unique_clusters_db_eng, counts_clusters_db_eng = np.unique(db.labels_, return_counts = True)
print(np.asarray((unique_clusters_db_eng, counts_clusters_db_eng)))
#[[   -1     0]
#[   91 10116]]

# One limitation of DBSCAN is that it is sensitive to the choice of epsilon (eps), in 
#particular if clusters have different densities. If epsilon is too small, sparser clusters 
#will be defined as noise. If epsilon is too large, denser clusters may be merged 
#together. 

#2D
pca_db_eng = PCA(n_components=2).fit(cont_norm2)
pca_db_eng_2d = pca_db_eng.transform(cont_norm2)
for i in range(0, pca_db_eng_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 1:
        c2 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='g',marker='o')
    elif db.labels_[i] == 3:
        c5 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='y',marker='s')
    elif db.labels_[i] == 4:
        c6 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='m',marker='p')
    elif db.labels_[i] == 5:
        c7 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='c',marker='H')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_db_eng_2d[i,0],pca_db_eng_2d[i,1],c='b',marker='*')

plt.legend([c1, c3], ['Cluster 1', 'Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()
# So with this kind of data DBSCAN doesn´t work very well


#3D
pca_db_eng = PCA(n_components=3).fit(cont_norm2)
pca_db_eng_3d = pca_db_eng.transform(cont_norm2)
#Add my visuals
my_color_db=[]
my_marker_db=[]
#Load my visuals
for i in range(pca_db_eng_3d.shape[0]):
    if db.labels_[i] == 0:
        my_color_db.append('r')
        my_marker_db.append('+')
    elif db.labels_[i] == 1:
        my_color_db.append('b')
        my_marker_db.append('o')
    elif db.labels_[i] == 2:
        my_color_db.append('g')
        my_marker_db.append('*')
    elif db.labels_[i] == -1:
        my_color_db.append('k')
        my_marker_db.append('<')
        
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(100): #the range should be the shape
    ax.scatter(pca_db_eng_3d[i,0], pca_db_eng_3d[i,1], pca_db_eng_3d[i,2], c=my_color_db[i], marker=my_marker_db[i])

#here we used the 3 most relevant PCA's    
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')



# 5.4.2. For numerical consumption variables 
new_db = DBSCAN(eps= 0.5, #eps = radius
            min_samples= 10).fit(cons_norm)

labels = new_db.labels_

# Number of clusters in labels, ignoring noise if present.
new_clusters_db_cons = len(set(labels)) - (1 if -1 in labels else 0) #noise is -1


unique_clusters_db_cons, counts_clusters_db_cons = np.unique(new_db.labels_, return_counts = True)
print(np.asarray((unique_clusters_db_cons, counts_clusters_db_cons)))
#[[  -1    0    1    2    3]
#[1904 8275    8    9   11]]

pca_db_cons = PCA(n_components=2).fit(cons_norm)
pca_db_cons_2d = pca_db_cons.transform(cons_norm)
for i in range(0, pca_db_eng_2d.shape[0]):
    if db.labels_[i] == 0:
        c1 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='r',marker='+')
    elif db.labels_[i] == 2:
        c4 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='k',marker='v')
    elif db.labels_[i] == 3:
        c5 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='y',marker='s')
    elif db.labels_[i] == 4:
        c6 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='m',marker='p')
    elif db.labels_[i] == 5:
        c7 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='c',marker='H')
    elif db.labels_[i] == -1:
        c3 = plt.scatter(pca_db_cons_2d[i,0],pca_db_cons_2d[i,1],c='b',marker='*')

plt.legend([c1, c3], ['Cluster 1','Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()

#3D
pca_db_cons = PCA(n_components=3).fit(cons_norm)
pca_db_cons_3d = pca_db_cons.transform(cons_norm)
#Add my visuals
my_color_db=[]
my_marker_db=[]
#Load my visuals
for i in range(pca_db_cons_3d.shape[0]):
    if db.labels_[i] == 0:
        my_color_db.append('r')
        my_marker_db.append('+')
    elif db.labels_[i] == 1:
        my_color_db.append('b')
        my_marker_db.append('o')
    elif db.labels_[i] == 2:
        my_color_db.append('g')
        my_marker_db.append('*')
    elif db.labels_[i] == -1:
        my_color_db.append('k')
        my_marker_db.append('<')
        
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(100):
    ax.scatter(pca_db_cons_3d[i,0], pca_db_cons_3d[i,1], pca_db_cons_3d[i,2], c=my_color_db[i], marker=my_marker_db[i])

#here we have used the 3 most relevant PCA's    
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')



# Our dataset is sparse and with high-dimensionality. High-dimensional density estimation is a 
# properly hard problem. The resulting clustering returned by DBSCAN is rather sparse itself 
# and assumes (probably wrongly) that the data at hand are riddled with outliers.
#So, DBSCAN doens´t work with our data.

#------------------------------------------------------------------------------------------------------------------------------------
# 5.5. MEAN SHIFT

    # 5.5.1. For numerical engage variables 
scaler = StandardScaler()
cont_norm2 = scaler.fit_transform(continuous_engage_variables)

cont_norm2 = pd.DataFrame(cont_norm2, columns = continuous_engage_variables.columns)

to_MS_eng = cont_norm2

my_bandwidth_eng = estimate_bandwidth(to_MS_eng,
                               quantile=0.2, #the quantile default is 0.3
                               n_samples=1000)

ms_eng = MeanShift(bandwidth = my_bandwidth_eng,
               bin_seeding=True)

ms_eng.fit(to_MS_eng)
labels = ms_eng.labels_
cluster_centers_eng = ms_eng.cluster_centers_

labels_unique_eng = np.unique(labels)
n_clusters_eng_ = len(labels_unique_eng) # 2 clusters 

#Values
X=cluster_centers_eng
scaler.inverse_transform(X)
# array([[2493.72130822,  143.91287628,   31.06335034]])

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)
#[[    0 10207]]


#2D
pca_eng = PCA(n_components=2).fit(to_MS_eng)
pca_eng_2d = pca_eng.transform(to_MS_eng)
for i in range(0, pca_eng_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_eng_2d[i,0],pca_eng_2d[i,1],c='r',marker='+')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_eng_2d[i,0],pca_eng_2d[i,1],c='b',marker='*')

plt.legend([c1,  c3], ['Cluster 1', 'Cluster 3 '])
plt.title('Mean Shift found 2 clusters')
plt.show()


#3D
pca_eng = PCA(n_components=3).fit(to_MS_eng)
pca_eng_3d = pca_eng.transform(to_MS_eng)
#Add my visuals
my_color=[]
my_marker=[]
#Load my visuals
for i in range(pca_eng_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('b')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('g')
        my_marker.append('*')
    elif labels[i] == 3:
        my_color.append('k')
        my_marker.append('<')
        
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(250):
    ax.scatter(pca_eng_3d[i,0],
               pca_eng_3d[i,1], 
               pca_eng_3d[i,2], c=my_color[i], marker=my_marker[i])


ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

    # 5.5.2. For numerical consumption variables 
scaler = StandardScaler()
cons_norm = scaler.fit_transform(consumo)
cons_norm = pd.DataFrame(cons_norm, columns = consumo.columns)

to_MS_cons = cons_norm

my_bandwidth_cons = estimate_bandwidth(to_MS_cons,
                               quantile=0.2,
                               n_samples=1000)

ms_cons = MeanShift(bandwidth = my_bandwidth_cons,
               bin_seeding=True)

ms_cons.fit(to_MS_cons)
labels = ms_cons.labels_
cluster_centers_cons = ms_cons.cluster_centers_

labels_unique_cons = np.unique(labels)
n_clusters_cons_ = len(labels_unique_cons)  

#Values
X_1=cluster_centers_cons
scaler.inverse_transform(X_1)

#Count
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)
#[[   0 7568]
# [   1 2639]]

#2D
pca_cons = PCA(n_components=2).fit(to_MS_cons)
pca_cons_2d = pca_cons.transform(to_MS_cons)
for i in range(0, pca_cons_2d.shape[0]):
    if labels[i] == 0:
        c1 = plt.scatter(pca_cons_2d[i,0],pca_cons_2d[i,1],c='r',marker='+')
    elif labels[i] == 1:
        c2 = plt.scatter(pca_cons_2d[i,0],pca_cons_2d[i,1],c='g',marker='o')
    elif labels[i] == 2:
        c3 = plt.scatter(pca_cons_2d[i,0],pca_cons_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Cluster 3 '])
plt.title('Mean Shift found 3 clusters')
plt.show()


#3D
pca_cons = PCA(n_components=3).fit(to_MS_cons)
pca_cons_3d = pca_cons.transform(to_MS_cons)
#Add my visuals
my_color=[]
my_marker=[]
#Load my visuals
for i in range(pca_cons_3d.shape[0]):
    if labels[i] == 0:
        my_color.append('r')
        my_marker.append('+')
    elif labels[i] == 1:
        my_color.append('b')
        my_marker.append('o')
    elif labels[i] == 2:
        my_color.append('g')
        my_marker.append('*')
    elif labels[i] == 3:
        my_color.append('k')
        my_marker.append('<')
         
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(250):
    ax.scatter(pca_cons_3d[i,0],
               pca_cons_3d[i,1], 
               pca_cons_3d[i,2], c=my_color[i], marker=my_marker[i])
    
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

#------------------------------------------------------------------------------------------------------------------------------------

# 5.6 Gaussian Mixture Model
#We must define how much cluster we want. We usually start with the mean being the seeds, and 
#the  standard deviation being equal to one. In k-means a point belong just to one cluster. 
#Here (Gaussian Misture Models) one point belong to all clusters (with a high probability to 
#ones, with a lower probability to others).

    # 5.6.1. Continuous engage variables
scaler = StandardScaler()
cont_norm2 = scaler.fit_transform(continuous_engage_variables)

cont_norm2 = pd.DataFrame(cont_norm2, columns = continuous_engage_variables.columns)


gmm_eng = mixture.GaussianMixture(n_components= 4, #number of clusters that we have found in k-means
                              init_params='kmeans', 
                              max_iter=1000,
                              n_init=10,
                              verbose = 1)    

#creates 4 centroids and we will use gaussian distribution
gmm_eng.fit(cont_norm2) #every points belongs to all clusters

EM_eng_labels_ = gmm_eng.predict(cont_norm2)

#Individual
EM_score_samp_eng = gmm_eng.score_samples(cont_norm2)
#Individual
EM_pred_prob_eng = gmm_eng.predict_proba(cont_norm2)

scaler.inverse_transform(gmm_eng.means_)  
#           'Salary'        'CMV'        'Client_Years'
#array([[1568.04263109,  315.9212725 ,   27.37144588],
#       [3279.63232267,  308.30555788,   25.3728734 ],
#       [2584.37048394,  -23.0607013 ,   30.00663847],
#       [2663.10154373,  277.04544558,   37.10716328]])


    # 5.6.2. Continuous consumption variables
#1. Select the variables:
consumo = clean_df.loc[:,['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']]

#2. Normalize the variables selected:
scaler = StandardScaler()
cons_norm = scaler.fit_transform(consumo)
cons_norm = pd.DataFrame(cons_norm, columns = consumo.columns)

gmm_cons = mixture.GaussianMixture(n_components= 3, #number of clusters that we have found in k-means
                              init_params='kmeans', 
                              max_iter=1000,
                              n_init=10,
                              verbose = 1)
#creates 3 centroids and we will use gaussian distribution
gmm_cons.fit(cons_norm) #every points belongs to all clusters

EM_cons_labels_ = gmm_cons.predict(cons_norm)

#Individual
EM_score_samp_cons = gmm_cons.score_samples(cons_norm)
#Individual
EM_pred_prob_cons = gmm_cons.predict_proba(cons_norm)


scaler.inverse_transform(gmm_cons.means_) #with this we get the centroids
#          'Motor'       'Household'    'Health'       'Life'   'Work_Compensation'
#array([[402.81448212,  69.80679341, 143.75205103,  13.77493389, 13.94724544],
#      [263.90855756, 209.78705835, 194.29400283,  42.53825498, 41.07221435],
#       [143.18027532, 439.49268151, 182.46876362,  89.05199611, 86.78122656]])
   

#------------------------------------------------------------------------------------------------------------------------------------   
    

#5.7 CLASSIFICATION TREES: REASSIGNING CUSTOMERS ('OUTLIERS')
    # 5.7.1. For continuos variables that characterize the clients
values, counts = np.unique(kmeans2.labels_, return_counts=True)
pd.DataFrame(np.asarray((values, counts))).T, 
columns = ['Label', 'Number']   

print(values, counts)
# [0 1 2 3] [2389 2551 2663 2604]

numberOfElements = list(zip(values, counts))
for cnt in numberOfElements:
    print(cnt[0], 'there are', cnt[1])
#0 there are 2389
#1 there are 2551
#2 there are 2663
#3 there are 2604


le = preprocessing.LabelEncoder()

c_parameter_name = 'max_depth'
c_parameter_values = [1,2,3,4,5,6,7,8,9,10,11]
c_best_parameter = 0
c_best_accuracy = 0
c_worst_parameter = 0
c_worst_accuracy = 100   

dt_eng= pd.DataFrame(columns=[c_parameter_name, 'accuracy'])
for input_parameter in c_parameter_values:
    clf_2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=input_parameter, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=21, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    X_2 = continuous_engage_variables[['Salary', 'CMV']] #Since Client_Years variable don't has outliers was not included here.
    y_2 = getting_labels_km_2[['Labels']]

# Split dataset into training set and test set
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, 
                                                            test_size=0.3, 
                                                            random_state=1) 
# 70% training and 30% test. If we continue to increase the splits we will decrease the error 
# till reach zero but there is a problem, this will create overfitting (we are losing the 
#habillity to generalize).This is why we must have a test set.


# Train Decision Tree Classifer
clf_2 = clf_2.fit(X_2_train, y_2_train)


clf_2.feature_importances_
#This result is a normalized value with GINI
# array([0.45781434, 0.54218566])
    
y_2_pred = clf_2.predict(X_2_test)
                       
acc_score = accuracy_score(y_2_test,y_2_pred)*100
dt_eng = dt_eng.append({c_parameter_name : input_parameter , 'accuracy' : acc_score}, ignore_index=True)
for input_parameter in c_parameter_values:
    clf_2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=input_parameter, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=21, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    X_2 = continuous_engage_variables[['Salary', 'CMV']] #Since Client_Years variable don't has outliers was not included here.
    y_2 = getting_labels_km_2[['Labels']]

# Split dataset into training set and test set
    X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, 
                                                    test_size=0.3, 
                                                    random_state=1) # 70% training and 30% test
#If we continue to increase the splits we will decrease the error till reach zero but there is a problem, this will 
#create overfitting (we are losing the habillity to generalize).This is why we must have a test set.


# Train Decision Tree Classifer
    clf_2 = clf_2.fit(X_2_train, y_2_train)

    clf_2.feature_importances_
    
    y_2_pred = clf_2.predict(X_2_test)

    acc_score = accuracy_score(y_2_test,y_2_pred)*100
    dt_eng = dt_eng.append({c_parameter_name : input_parameter , 'accuracy' : acc_score}, ignore_index=True)
    if acc_score > c_best_accuracy:
        c_best_accuracy = acc_score
        c_best_parameter = input_parameter
        c_best_clf_2 = clf_2
        
    if acc_score < c_worst_accuracy:
        c_worst_accuracy = acc_score
        c_worst_parameter = input_parameter
        c_worst_clf_2 = clf_2
    
print(dt_eng)
print("")
plt.figure(figsize=(12,6))
sns.pointplot(x=c_parameter_name, y="accuracy", data=dt_eng)
title = 'Enagage variables Accuracy(%) vs ' + c_parameter_name + ' parameter'
plt.title(title)
plt.xticks(rotation= 90)
plt.grid()

clf_2 = DecisionTreeClassifier(random_state=0,
                             max_depth=3) 

clf_2 = clf_2.fit(X_2_train, y_2_train)
clf_2.feature_importances_
# array([0.46689757, 0.53310243])
#This result is a normalized value with GINI

rcParams['figure.figsize'] = 30,30
plot_tree(clf_2, filled=True)

dot_data_2 = tree.export_graphviz(clf_2, out_file=None) 
graph_2 = graphviz.Source(dot_data_2) 
print(graph_2)

dot_data_2 = tree.export_graphviz(clf_2, out_file=None,
                                feature_names= list(X_2.columns),
                                class_names = ['class_' + str(x) for x in np.unique(y_2)],
                                filled=True,
                                rounded=True,
                                special_characters=True)  
graph_2 = graphviz.Source(dot_data_2)

#Join all outliers
new_outliers_df = pd.concat([final_outliers_CMV, outliers_Salary])
new_outliers_df = pd.concat([new_outliers_df, outliers_Health])
new_outliers_df = pd.concat([new_outliers_df, outliers_Household])
new_outliers_df = pd.concat([new_outliers_df, outliers_Life])
new_outliers_df = pd.concat([new_outliers_df, outliers_Motor])
new_outliers_df = pd.concat([new_outliers_df, outliers_Work_Compensation])
new_outliers_df = new_outliers_df.drop(columns = ['First_Policy', 'Education', 'Area', 'Children', 'Claims', 'Client_Years', 'Client_Profit', 'Total_Premium'])


to_class_2 = {'Salary': [3234.0, 2354.0, 2176.0, 1086.0, 3279.0, 4435.0, 1634.0, 1117.0, 1370.0, 3355.0, 1586.0, 1771.0, 4566.0, 3574.0, 1089.0, 782.0, 1133.0, 870.0, 1482.0, 1094.0, 1421.0, 566.0, 55215.0, 34490.0, 4002.0, 987.0, 2642.0, 2832.0, 427.0, 2618.0, 1696.0, 3763.0, 4135.0, 3564.0, 2947.0, 2460.0, 2354.0, 984.0, 376.0, 1121.0, 3330.0],
              'CMV':[-14714.08, -8719.04, -10198.91, -165680.42, -64891.00, -52382.76, -28945.40, -10107.37, -7851.17, -26130.45, -2642.91, -6115.85, -2082.83, -37327.08, 1716.00, 2054.07, 1801.45, 1571.76, 1634.97, 1891.04, 1997.60, 1691.43, 122.25, 608.89, 1457.99, 804.05, 0.78, -31.00, -61.34, 473.54, 535.22, 466.21, 130.14, 11875.89, 4328.50, 5596.84, -46.89, 255.71, 797.92, -318.73, 2314.21]} 
                

# Creates pandas DataFrame. 
to_class_2 = pd.DataFrame(to_class_2, index =['cust1', 'cust2', 'cust3', 'cust4','cust5', 'cust6', 'cust7', 'cust8', 'cust9', 'cust10', 'cust11', 'cust12', 'cust13', 'cust14', 'cust15', 'cust16', 'cust17','cust18', 'cust19', 'cust20', 'cust21', 'cust22', 'cust23', 'cust24','cust25', 'cust26', 'cust27', 'cust28' , 'cust29', 'cust30', 'cust31', 'cust32', 'cust33', 'cust34', 'cust35', 'cust36', 'cust37', 'cust38', 'cust39', 'cust40', 'cust41']) 

to_class_2['Cont_cluster'] = clf_2.predict(to_class_2)

#Join outliers dataframe with labels
new_outliers_df = new_outliers_df.merge(to_class_2, left_on='Salary', right_on='Salary')
new_outliers_df = new_outliers_df.drop (columns = ['CMV_y'])
new_outliers_df.columns = ['ID', 'Salary', 'CMV', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Cont_cluster']


# Classify these new elements
#We have reassigned 43 customers (the ones that was previously removed by us since were 
# considered outliers). With this decision tree we can reassign them based on labels (Cont_cluster)


    # 5.7.2. For consumption variables
values, counts = np.unique(kmeans.labels_, return_counts=True)
pd.DataFrame(np.asarray((values, counts))).T, 
columns = ['Label', 'Number']   

print(values, counts)
# [0 1 2] [4213 1798 4196]

numberOfElements = list(zip(values, counts))
for cnt in numberOfElements:
    print(cnt[0], 'there are', cnt[1])
#0 there are 4213
#1 there are 1798
#2 there are 4196


le = preprocessing.LabelEncoder()
from sklearn.metrics import accuracy_score
c_parameter_name = 'max_depth'
c_parameter_values = [1,2,3,4,5,6,7,8,9,10,11]
c_best_parameter = 0
c_best_accuracy = 0
c_worst_parameter = 0
c_worst_accuracy = 100   
                       
dt_cons= pd.DataFrame(columns=[c_parameter_name, 'accuracy'])                         
for input_parameter in c_parameter_values:
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=input_parameter, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=21, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    X = consumption_variables[['Motor', 'Household', 'Health','Life', 'Work_Compensation']]
    y = getting_labels_km[['Labels']]

# Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1) # 70% training and 30% test

# Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)


    clf.feature_importances_
# array([0.24156886, 0.2994572 , 0.45897394])
#This result is a normalized value with GINI
    y_pred = clf.predict(X_test)


    acc_score = accuracy_score(y_test,y_pred)*100
    dt_cons = dt_cons.append({c_parameter_name : input_parameter , 'accuracy' : acc_score}, ignore_index=True)
    if acc_score > c_best_accuracy:
        c_best_accuracy = acc_score
        c_best_parameter = input_parameter
        c_best_clf = clf
        
    if acc_score < c_worst_accuracy:
        c_worst_accuracy = acc_score
        c_worst_parameter = input_parameter
        c_worst_clf = clf
    
print(dt_cons)
print("")
plt.figure(figsize=(12,6))
sns.pointplot(x=c_parameter_name, y="accuracy", data=dt_cons)
title = 'Consumption variables Accuracy(%) vs ' + c_parameter_name + ' parameter'
plt.title(title)
plt.xticks(rotation= 90)
plt.grid()


clf = DecisionTreeClassifier(random_state=0,
                             max_depth=3)

clf = clf.fit(X_train, y_train)

clf.feature_importances_

y_pred = clf.predict(X_test)

rcParams['figure.figsize'] = 30,30
plot_tree(clf, filled=True)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names= list(X.columns),
                                class_names = ['class_' + str(x) for x in np.unique(y)],
                                filled=True,
                                rounded=True,
                                special_characters=True)  

graph = graphviz.Source(dot_data)


to_class = {'Motor' : [557.44, 518.32, 297.61, 378.07, 410.30, 197.48, 175.70, 193.37, 127.58, 135.58, 501.65, 83.35, 370.07, 319.06, 67.90, 39.12, 53.01, 99.02, 32.56, 13.67, 30.34, 14.56, 281.83, 57.01, 431.86, 26.34, 424.19, 535.10, 16.67, 508.43, 161.92, 4003.44, 8744.61, 11604.42, 4273.49, 5645.50, 3106.62, 64.90, 28.34, 37.45, 350.51],
            'Household':[20.00, 4.45, 162.80, 78.90, 117.25, 280.60, 319.50, 342.85, 48.35, 290.05, -20.00, 98.35, 14.45, -25.55, 1673.10, 1957.60, 1826.45, 1544.75, 1748.10, 1918.15, 1924.25, 1777.55, 147.25, 358.95, 107.80, 829.05, 4130.70, 8762.80, 2223.75, 25048.80, 593.40, 612.90, 101.70, 48.90, 83.90, -25.55, 30.00, 197.25, 1223.00, 123.35, 201.70],
            'Health' : [29.56, 55.90, 143.36, 166.81, 95.35, 276.94, 294.39, 276.94, 398.41, 221.82, 90.46, 336.84, 189.59, 245.38, 65.90, 47.23, 68.68, 106.13, 51.01, 51.90, 37.23, 49.23, 130.58, 195.26, 7322.48, 28272.00, 118.69, 41.12, 37.45, 36.23, 310.17, 137.36, 1767.00, 1045.52, 105.13, 49.01, 59.01, 29.56, 106.91, 68.79, 125.80],
            'Life' : [5.00, 3.89, 136.47, 6.89, 37.34, 51.12, 44.12, 8.78, 19.56, 86.46, -1.00, 74.68, 14.78, 4.89, 112.02, 15.78, 3.89, 55.90, 132.47, 15.78, 33.34, 46.01, 84.46, 113.80, 12.89, 65.68, 15.67, -6.00, 36.34, 9.89, 398.30, 121.69, 155.14, 103.13, 3.00, 1.89, 12.89, 18.56, 65.68, 54.12, 39.23],
            'Work_Compensation' : [-9.00, 10.89, -3.00, 18.45, 22.56, 38.34, 16.89, 47.23, 12.78, 100.13, -2.00, 93.46, 23.56, 4.78, 27.56, 82.35, 88.13, 3.89, 44.34, 99.02, 98.35, 121.69, 66.68, 161.14, 930.44, 138.25, 41.45, 9.78, 67.79, 11.89, 50.23, 31.34, 130.58, 296.47, 3.89, 1.89, 9.00, 451.53, 494.10, 417.08, 1988.70]}
   

# Creates pandas DataFrame. 
to_class = pd.DataFrame(to_class, index =['cust1', 'cust2', 'cust3', 'cust4','cust5', 'cust6', 'cust7', 'cust8', 'cust9', 'cust10', 'cust11', 'cust12', 'cust13', 'cust14', 'cust15', 'cust16', 'cust17','cust18', 'cust19', 'cust20', 'cust21', 'cust22', 'cust23', 'cust24','cust25', 'cust26', 'cust27', 'cust28' , 'cust29', 'cust30', 'cust31', 'cust32', 'cust33', 'cust34', 'cust35', 'cust36', 'cust37', 'cust38', 'cust39', 'cust40', 'cust41']) 
to_class['Cons_cluster'] = clf.predict(to_class)

#Join outliers dataframe with labels
new_outliers_df = new_outliers_df.merge(to_class, left_on='Motor', right_on='Motor')
new_outliers_df = new_outliers_df.drop (columns = ['Household_y', 'Health_y', 'Life_y', 'Work_Compensation_y'])
new_outliers_df.columns = ['ID', 'Salary', 'CMV', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Cont_cluster', 'Cons_cluster']

# Classify these new elements
# Now we are able to reassign those 43 customers since now each customer has a Cont_cluster
# and Cons_cluster label.

#Concat all
frames= [clean_df, new_outliers_df]
result = pd.concat(frames, sort = False)
result= result.drop(columns = ['Education', 'Area', 'Children', 'Cat_cluster', 'Client_Years'])

count_clusters4 = result.groupby(['Cons_cluster','Cont_cluster'])['ID'].count()
count_clusters4 = count_clusters4.reset_index()
count_clusters4.to_excel('cons_eng_out.xlsx')
