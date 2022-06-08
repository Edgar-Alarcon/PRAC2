import time

import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def manageNanValues(df):
    missing_values=df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    #missing_values.plot.bar()
    missing_values[missing_values>0]/len(df)*100
    df.drop(["Cabin"],axis=1 ,inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return df

def preprocessColumns(df):

    df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    return df

def manageOutliers(df):
    

    
    #sns.boxplot(df['Fare'],data=df)
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    whisker_width = 1.5
    lower_whisker = Q1 - whisker_width*IQR
    upper_whisker = Q3 + whisker_width*IQR
    df = df[(df['Fare']<upper_whisker) & (df['Fare']>lower_whisker)]

    #sns.boxplot(df['Fare'],data=train_df)
    

    return df







def analyzeSex(titanic_df):

    # Implementing this using existing python library functions
    from scipy.stats import chi2_contingency
    import seaborn as sns
    import matplotlib.pyplot as plt


    sns.catplot(x="Sex", hue="Survived", kind="count", data=titanic_df)

    groupedby_genderval = titanic_df.groupby('Survived')['Sex'].value_counts()

    pivot_cont = pd.pivot_table(titanic_df[['Survived', 'Sex']], index=['Survived'], columns='Sex', aggfunc=len)


    pivot_cont_class = pd.pivot_table(titanic_df[['Survived', 'Pclass']], index=['Survived'], columns='Pclass',
                                      aggfunc=len)

    chi2, pval, dof, expected = chi2_contingency(pivot_cont)

    print("Tabla de contingencia ", pivot_cont)
    print('Chi cuadrado' ,chi2)




def analyzeClass(titanic_df):
    """
    group = titanic_df.groupby(['Pclass', 'Survived'])
    pclass_survived = group.size().unstack()

    # Heatmap - Color encoded 2D representation of data.
    sns.heatmap(pclass_survived, annot=True, fmt="d")

    """


    groupby_class = titanic_df.groupby('Survived')['Pclass'].value_counts()

    # General parameters

    num_passengers = titanic_df.shape[0]
    row_ind = ['Survived', 'Not Survived']  # Dependant variable
    col_names = ['class1', 'class2', 'class3']  # Independant variable



    contigency_df = pd.DataFrame({'class1': [groupby_class[0][1], groupby_class[1][1]], 'class2': [groupby_class[0][2], groupby_class[1][2]], 'class3': [groupby_class[0][3], groupby_class[1][3]]},
                                 index=row_ind)


    marginalvalues = contigency_df.sum(axis=1)
    conditionalvalues = contigency_df.sum(axis=0)



    def expectedvalues_calc(marginal, conditional, num_passengers):
        res_df = pd.DataFrame(columns=col_names,  # creating an empty DF to fill with expected values
                              index=row_ind)

        res_df.loc['Survived'] = pd.Series(list((conditional * marginal[0]) / num_passengers), index=col_names)
        res_df.loc['Not Survived'] = pd.Series(list((conditional * marginal[1]) / num_passengers), index=col_names)

        return res_df

    expected_df = expectedvalues_calc(marginalvalues, conditionalvalues, num_passengers)


    chisquarevalue = ((contigency_df - expected_df) ** 2 / expected_df).values.sum()

    print("Tabla de contingencia ", contigency_df)
    print('Chi cuadrado' ,chisquarevalue)




def testNormalityFare(titanic_df):

    variable_to_study = titanic_df["Fare"]
    shapiro_test = stats.shapiro(variable_to_study)
    print("Shapiro test de normalidad = ", shapiro_test)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(x=variable_to_study, density=True, bins=10, color="#3182bd", alpha=0.5)
    ax.plot(variable_to_study, np.full_like(variable_to_study, -0.01), '|k', markeredgewidth=1)
    ax.set_title('Fare')

    ax.set_ylabel('Densidad de probabilidad')
    ax.legend();
    plt.clf()





titanic_df = pd.read_csv("train.csv")


titanic_df = manageNanValues(titanic_df)
titanic_df = preprocessColumns(titanic_df)
titanic_df = manageOutliers(titanic_df)
testNormalityFare(titanic_df)
analyzeSex(titanic_df)
analyzeClass(titanic_df)


