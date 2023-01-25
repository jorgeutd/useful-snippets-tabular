import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency


# ===============
#    GENERAL
# ===============


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}  -->', cat_cols)
    print(f'num_cols: {len(num_cols)}  -->', num_cols)
    print(f'cat_but_car: {len(cat_but_car)}  -->', cat_but_car)
    print(
        f'num_but_cat: {len(num_but_cat)}    <---   (already included in "cat_cols". Just given for reporting purposes)')

    print("{cat_cols + num_cols + cat_but_car = all variables}")

    # cat_cols + num_cols + cat_but_car = number of variables.
    # all variables: cat_cols + num_cols + cat_but_car
    # num_but_cat is included "cat_cols".
    # num_but_cat is just given for reporting purposes.

    return cat_cols, cat_but_car, num_cols, num_but_cat


# ===============================
# CATEGORICAL VARIABLES ANALYSIS
# ===============================

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("===================")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def cat_summary_adv(dataframe, categorical_cols, number_of_classes=10):
    col_count = 0
    cols_more_classes = []
    for col in categorical_cols:
        if dataframe[col].nunique() <= number_of_classes:
            print(pd.DataFrame({col: dataframe[col].value_counts(),
                                "Ratio (%)": round(100 * dataframe[col].value_counts() / len(dataframe), 2)}),
                  end="\n\n\n")
            col_count += 1
        else:
            cols_more_classes.append(dataframe[col].name)

    print(f"{col_count} categorical variables have been described.\n")
    if len(cols_more_classes) > 0:
        print(f"There are {len(cols_more_classes)} variables which have more than {number_of_classes} classes:")
        print(cols_more_classes)



"""
Performs chi-squared test of independence on all pairs of categorical variables in a DataFrame.

df: DataFrame containing the data
alpha: significance level for rejecting the null hypothesis (default 0.05)
min_count: minimum count for each cell of the contingency table (default 5)

Returns a DataFrame containing the results of the chi-squared test for each pair of variables, 
including the contingency table, p-value, and whether to reject the null hypothesis (p < alpha).
"""
def chi_squared_test(df, alpha=0.05, min_count=5):
    results = []
    categorical_vars = df.select_dtypes(include=['category', 'object']).columns
    for var1 in categorical_vars:
        for var2 in categorical_vars:
            if var1 != var2:
                ct = pd.crosstab(df[var1], df[var2])
                if (ct < min_count).any().any():
                    print(f"Skipping {var1} - {var2} because one of the cells has a count less than {min_count}.")
                    continue
                chi2, p, dof, expected = chi2_contingency(ct)
                results.append({'var1': var1, 'var2': var2, 'ct': ct, 'p': p})
    results_df = pd.DataFrame(results)
    results_df['reject_null'] = results_df['p'] < alpha
    return results_df

# ===============================
# NUMERICAL VARIABLES ANALYSIS
# ===============================

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=30, figsize=(12, 12), density=False)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def num_hist_boxplot(dataframe, numeric_col):
    col_counter = 0
    for col in numeric_col:
        dataframe[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        sns.boxplot(x=dataframe[col], data=dataframe)
        plt.show()
        col_counter += 1
    print(f"{col_counter} variables have been plotted")


def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers



def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


# ===============================
# TARGET VARIABLE ANALYSIS
# ===============================

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def two_t_test(data, quantitative_vars, target_variable):
    """
    The function takes in a dataframe, a list of quantitative variables, and a target variable. It then
    performs a Shapiro-Wilk test to check for normality, and if the p-value is less than 0.05, it
    performs a Mann-Whitney U test. If the p-value is greater than 0.05, it performs a Levene test to
    check for homogeneity of variance, and if the p-value is less than 0.05, it performs a Mann-Whitney
    U test.
    """
    columns = []
    p_values = []
    test_significance = []
    for var in quantitative_vars:
        columns.append(var)
        category_1 = data[var][data[target_variable] == False]
        category_2 = data[var][data[target_variable] == True]
        for bol in [category_1]:
            t_stats1, p_val1 = stats.shapiro(bol)
        for bin in [category_2]:
            t_stats2, p_val2 = stats.shapiro(bin)
        if p_val1 > 0.05 or p_val2 > 0.05:
            stats_3, p_val3 = stats.levene(category_1, category_2)

        if p_val1 <= 0.05 or p_val2 <= 0.05 or p_val3 <= 0.05:
            ms, mp = stats.mannwhitneyu(category_1, category_2)
            p_values.append(round(mp, 4))
        if mp < 0.05:
            test_significance.append('significant')
        else:
            test_significance.append('insignificant')

    return pd.DataFrame({'Feature': columns, 'P-Value': p_values, 'Significance': test_significance})


# ====================================================
# Correlations between Target and Independent Variables
# ====================================================

def find_correlation(dataframe, numeric_cols, target, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == target:
            pass
        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

# low_corrs, high_corrs = find_correlation(df, num_cols, "TARGET")


def correlation_heatmap(dataframe):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(dataframe.corr(), annot=True, cmap=colormap)
    plt.show()

# ===============================
# FEATURE ENGINEERING 
# ===============================

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, rare_perc):

    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df