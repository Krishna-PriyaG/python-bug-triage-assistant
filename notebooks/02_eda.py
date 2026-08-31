# %% [markdown]
# # Feature Description
# 
# ## Original Dataset Features
# 
# | Feature | Type | Description |
# |---------|------|-------------|
# | `project` | Categorical | Name of the Python project/repository containing the bug (e.g., pandas, keras, scrapy). |
# | `bug_id` | Categorical | Unique bug identifier within a project. |
# | `python_version` | Categorical | Python version used to reproduce the bug. |
# | `buggy_commit_id` | String | Git commit hash corresponding to the buggy version. |
# | `fixed_commit_id` | String | Git commit hash corresponding to the bug-fixed version. |
# | `test_file` | String | Test file(s) used to reproduce or validate the bug fix. |
# | `pythonpath` | String | Optional Python module search path required for certain projects. |
# | `patch` | Text | Unified Git diff representing the bug fix. |
# 
# ## Engineered Features
# 
# | Feature | Type | Description | Extraction Logic |
# |---------|------|-------------|------------------|
# | `modified_files` | List | List of source files modified by the bug fix. | Extracted from `diff --git` headers. |
# | `num_files_changed` | Integer | Number of files modified in the patch. | `len(modified_files)` |
# | `lines_added` | Integer | Number of added lines of code. | Count lines beginning with `+` (excluding `+++`). |
# | `lines_removed` | Integer | Number of removed lines of code. | Count lines beginning with `-` (excluding `---`). |
# | `num_hunks` | Integer | Number of contiguous change regions in the patch. | Count occurrences of `@@`. |
# | `patch_length` | Integer | Total number of lines in the Git patch, including metadata, context, additions, and deletions. | `len(patch.splitlines())` |
# | `has_comment` | Boolean | Indicates whether the patch contains comment syntax. | Regex search for `#`, `//`, `/*`, or `*/`. |
# | `declared_classes_in_patch` | List | Class declarations appearing in the patch. | Regex extraction using `class`. |
# | `declared_functions_in_patch` | List | Function declarations appearing in the patch. | Regex extraction using `def`. |
# | `number_of_declared_classes_in_patch` | Integer | Number of class declarations present in the patch. | `len(declared_classes_in_patch)` |
# | `number_of_declared_functions_in_patch` | Integer | Number of function declarations present in the patch. | `len(declared_functions_in_patch)` |
# 
# > **Note:** `declared_classes_in_patch` and `declared_functions_in_patch` count declarations appearing anywhere within the Git patch. Since Git patches include surrounding context lines, these features represent the structural scope of the patch rather than the exact number of modified class or function definitions.

# %%
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %% [markdown]
# Two Ways to Connect to PostgreSQL

# %%
engine = create_engine('postgresql+psycopg2://krishnapriyagitalaxmi@localhost:5432/bugsinpy')
test = pd.read_sql("SELECT * FROM bug_dataset LIMIT 5", engine)
test

# %%
conn = psycopg2.connect(
    host="localhost",
    database="bugsinpy",
    user="krishnapriyagitalaxmi")
cursor = conn.cursor()
cursor.execute("SELECT * FROM bug_dataset LIMIT 5")
rows = cursor.fetchall()
rows

# %%
cursor.close()
conn.close()

# %% [markdown]
# EDA - Exploratory Data Analysis 

# %%
#import data from database
df_raw=pd.read_sql("SELECT * FROM bug_dataset", engine)
df_raw.head(10)

# %%
df_raw.info()

# %% [markdown]
# Observed nulls in pythonpath column - need to investigate further

# %%
df_raw[df_raw["pythonpath"].notnull()]["project"].value_counts()

# %%
df_raw[df_raw["pythonpath"].isnull()]["project"].value_counts()

# %% [markdown]
# Pythonpath is populated for a small subset of bugs. It appears to be optional execution metadata rather than an analytical feature, so missing values are expected and were left unchanged

# %%
df_raw.describe(include='number')

# %% [markdown]
# Observed min =0 in all raws depicting some row/rows have no patch information attached to it

# %%
df_raw[df_raw['num_files_changed']==0]

# %% [markdown]
# One bug instance (keras, bug 12) contained an empty bug_patch.txt. Since this project analyzes source code patches and all engineered patch features depend on the patch content, this record will be excluded from downstream analysis. (Cleaning will be handled together)

# %%
df_raw[df_raw['num_files_changed']==8]

# %%
df_raw[df_raw['lines_added']==764]

# %%
df_raw.describe(include='string')

# %%
df_raw[df_raw['lines_added']==0]

# %%
df_raw[df_raw['lines_removed']==0]

# %%
df_clean=df_raw[~((df_raw['bug_id']=="12") & (df_raw['project']=='keras'))]
df_clean.describe(include='number')

# %% [markdown]
# Data Coverage Investigation

# %%
%matplotlib inline
df_projectwise_bugs=df_clean["project"].value_counts().sort_values(ascending=False)
df_projectwise_bugs.plot(kind='bar', figsize=(10, 5))
plt.title("Number of Bugs per Project")
plt.xlabel("Project")  
plt.ylabel("Number of Bugs")
plt.show()

# %% [markdown]
# The BugsInPy dataset exhibits an uneven distribution of bug instances across projects. Pandas contributes the largest number of bugs (169), while several projects contribute fewer than 10. 

# %%
plt.figure(figsize=(10, 5))
bar = plt.bar(df_projectwise_bugs.index, df_projectwise_bugs.values)
plt.title("Number of Bugs per Project")
plt.xlabel("Project")
plt.ylabel("Number of Bugs")
plt.xticks(rotation=45)
plt.bar_label(bar)
plt.show()


# %% [markdown]
# * High coverage: pandas, keras, scrapy
# * Low coverage: PySnooper, sanic
# 
# Issues this could cause , can be investigated later:
# eg: if pandas patches are generally larger than FastAPI patches, the model might partly learn “this looks like a pandas bug.” 
# 

# %%
df_clean.info()

# %% [markdown]
# ## What do bug patches generally look line BugsinPy?

# %%
plt.hist(df_clean['patch_length'], bins=30, edgecolor='black')
plt.title("Distribution of Patch Lengths")
plt.xlabel("Patch Length")
plt.ylabel("Frequency")
plt.xlim(0,450)
plt.show()

# %%
df_clean[['patch_length', 'num_files_changed']].describe(include='number', percentiles=[.25, .5, .75, .9, .95, .99])

# %% [markdown]
# * The distribution is strongly **right-skewed**, indicating that most bug fixes involve relatively small patches.
# 
# * The **median patch length is 23**, meaning half of all bug fixes have a patch length of 23 or less.
# 
# * The **75th percentile is 40**, suggesting that most bug fixes remain relatively small.
# 
# * A small number of patches are substantially larger, producing a long right tail and indicating the presence of outliers.
# 
# suggesting that bug fixes in the BugsInPy dataset seem generally localized (The fix is confined to a small part of the codebase rather than spread across many files or many sections of code.), with only a few requiring extensive code modifications.

# %% [markdown]
# ## How much code is typically modiefied to fix a bug?
# 
# ## Code Churn Analysis

# %%
total_lines_changed= df_clean["lines_added"] + df_clean["lines_removed"]
total_lines_changed.plot(kind='hist', bins=30, edgecolor='black')
plt.title("Distribution of Total Lines Changed")
plt.xlabel("Total Lines Changed")
plt.ylabel("Frequency")
plt.xlim(0,240)

# %%
total_lines_changed.describe(percentiles=[.25, .5, .75, .9, .95, .99])

# %% [markdown]
# 

# %%
plt.boxplot(total_lines_changed, vert=False)
plt.title("Boxplot of Total Lines Changed")
plt.xlabel("Total Lines Changed")
plt.xlim(0, 240)
plt.show()

# %% [markdown]
# The distribution of total lines changed is highly right-skewed. While the mean bug fix modifies approximately 31 lines of code, the median is only 7 lines, indicating that most bug fixes are relatively small. The boxplot reveals numerous high-value outliers, suggesting that a small number of complex bugs require substantially larger code modifications.

# %% [markdown]
# We are seeing the same pattern twice:
# 
# * patch_length is right-skewed
# * total_lines_changed is right-skewed
# 
# Are these two features strongly correlated?

# %%
df_clean['lines_changed'] = df_clean['lines_added'] + df_clean['lines_removed']
df_clean.head()

# %% [markdown]
# ## Correlation Analysis between Numerical features

# %%
correlation_matrix = df_clean[['patch_length', 'num_files_changed', 'lines_added', 'lines_removed','num_hunks','number_of_declared_functions_in_patch','number_of_declared_classes_in_patch']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)

# %% [markdown]
# The strongest cluster:
# 
# patch_length - lines_added                     0.99
# 
# patch_length - lines_removed                   0.98
# 
# patch_length - declared_functions              0.98
# 
# lines_added - declared_functions               0.97
# 
# lines_removed - declared_functions             0.96
# 
# Larger patches tend to modify more lines of code and involve more function definitions or declarations.

# %% [markdown]
# Some weak correlations:
# 
# num_files_changed - patch_length    0.31
# 
# num_files_changed - lines_added     0.30
# 
# num_files_changed - lines_removed   0.19
# 
# A large bug fix does not necessarily span many files.

# %% [markdown]
# A moderate-to-strong relationship:
# 
# num_hunks - num_files_changed = 0.61
# 
# A hunk is one contiguous block of changes.If you modify more files,you’ll usually have more hunks.
# 
# num_hunks - declared_classes = 0.84
# 
# Bug fixes touching many classes also tend to be spread across multiple change regions.
# 

# %% [markdown]
# Almost no correlation:
# 
# num_hunks - lines_removed = 0.08
# 
# Removing lots of lines doesn’t necessarily create many hunks.

# %% [markdown]
# ## Which Python versions are represented in BugsInPy?

# %%
python_version_counts = df_clean['python_version'].value_counts()
bar=plt.bar(python_version_counts.index, python_version_counts.values)
plt.xlabel('Python Version')
plt.ylabel('Count')
plt.title('Distribution of Python Versions in BugsInPy')
plt.bar_label(bar)
plt.show()

# %%
plt.pie(python_version_counts.values, labels=python_version_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Python Versions in BugsInPy')
plt.show()

# %% [markdown]
# * Python 3.8.3 dominates the dataset, accounting for well over half of all bug instances.
# * The remaining bugs are distributed across several versions of Python 3.6, 3.7, and 3.8.
# * The dataset spans multiple Python releases, indicating that BugsInPy is not tied to a single runtime environment and contains bugs reproduced under different interpreter versions.
# * Some Python versions (e.g., 3.7.7) are represented by only a handful of bugs, suggesting that version-specific analyses for these releases may not be statistically meaningful.

# %% [markdown]
# ## How frequently do bug-fix patches contain comments?

# %%
has_comments_counts = df_clean['has_comment'].value_counts()
plt.pie(has_comments_counts.values, labels=has_comments_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Bugs with Comments in BugsInPy')
plt.show()

# %% [markdown]
# * 57% of bug-fix patches contain at least one comment.
# * 43% do not contain any detected comments.
# * This suggests that comments frequently accompany bug fixes, either to explain implementation decisions, document behavior, or update existing documentation.
# * However, since the distribution is fairly balanced, the presence of comments alone is unlikely to be a strong feature for the machine learning model .

# %% [markdown]
# ## Project-wise Analysis

# %% [markdown]
# ## Which projects tend to have larger bug fix patches?

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x='project', y='patch_length', data=df_clean)
plt.xticks(rotation=45)
plt.title('Patch Length Distribution by Project')
plt.xlabel('Project')
plt.ylabel('Patch Length')
plt.show()


# %% [markdown]
# Pandas has a few 2500-line patches that stretch the y-axis.
# 
# As a result, every other box gets compressed near zero.

# %%
projects=sorted(df_clean['project'].unique())
fig,axes = plt.subplots(nrows=4, ncols=5, figsize=(20,20))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
for i, project in enumerate(projects):
    sns.boxplot(x='project', y='patch_length', data=df_clean[df_clean['project']==project], ax=axes[i])
    axes[i].set_title(f'Patch Length Distribution for {project}')
    axes[i].set_xlabel('Project')
    axes[i].set_ylabel('Patch Length')
plt.tight_layout()

# %% [markdown]
# Pandas contains the most extreme bug fixes
# 
# * Median patch length is still relatively small.
# * However, it contains several extremely large outliers, including patches exceeding 2500 lines.
# 
# Although most pandas bug fixes are comparable in size to other projects, a small number of exceptionally large fixes substantially increase the overall variability.

# %%
order = df_clean.groupby('project')['patch_length'].median().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='project', y='patch_length', data=df_clean, order=order)
plt.xticks(rotation=45)
plt.title('Patch Length Distribution by Project (Ordered by Median Patch Length)')
plt.xlabel('Project')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.ylabel('Patch Length')  


# %%
plt.figure(figsize=(10, 5))
sns.boxplot(x='project', y='patch_length', data=df_clean,order=order, showfliers=False)
plt.xticks(rotation=45)
plt.title('Patch Length Distribution by Project (Without Outliers)')
plt.xlabel('Project')
plt.ylabel('Patch Length')
plt.show()

# %% [markdown]
# Two complementary visualizations are presented: one excluding outliers to facilitate comparison of the central distributions, and another using a logarithmic scale to visualize the complete range of patch lengths, including extreme observations.
# 
# Despite the diversity of projects, the median patch length for most repositories lies between approximately 15 and 35 lines. This suggests that the majority of bug fixes involve relatively localized code modifications rather than extensive refactoring.
# 
# Among the repositories, PySnooper exhibits the highest median patch length and a comparatively wide interquartile range, indicating that, within this dataset, its bug fixes generally involve larger code modifications than those of other projects.
# 
# Projects such as Black, Keras, Pandas, and Luigi exhibit relatively large interquartile ranges, indicating greater variability in patch lengths within these repositories. While many bug fixes remain small, these projects also contain a wider range of bug-fix sizes than the other repositories.

# %% [markdown]
# ## Do projects differ in the amount of code they actually modify to fix a bug?

# %%
order = df_clean.groupby('project')['lines_changed'].median().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='project', y='lines_changed', data=df_clean, order=order,showfliers=False)
plt.xticks(rotation=45)
plt.title('Lines Changed Distribution by Project (Without Outliers)')
plt.xlabel('Project')
plt.ylabel('Lines Changed')
plt.show()

# %% [markdown]
# Across most repositories, the median number of lines changed remains below 15 lines, indicating that the majority of bug fixes require relatively localized code changes.
# 
# Black, PySnooper, and Keras exhibit the largest typical code changes.These repositories have the highest median values and comparatively wide interquartile ranges, suggesting that their bug fixes generally involve modifying more lines of code than those of other repositories within this dataset.
# 
# Repositories such as Black, Keras, Pandas, HTTPie, and PySnooper display relatively large interquartile ranges, indicating that the amount of code modified varies substantially between bug fixes.
# 
# Projects including spaCy, Luigi,and Scrapy have comparatively compact distributions, suggesting that their bug fixes tend to involve smaller and more consistent code modifications.
# 

# %% [markdown]
# * In the patch-length plot, PySnooper was clearly the largest.
# * Here, Black actually has the highest median code churn.
# 
# A larger Git patch does not necessarily correspond to more lines of code being modified.
# 

# %% [markdown]
# ## Semantic Feature Analysis

# %% [markdown]
# ## What are the top exception types?

# %%
df_clean.info()

# %%
from collections import Counter

exception_counts= Counter(exception for exception_list in df_clean["exception_types"] for exception in exception_list)

exception_counts

# %%
type(df_clean["exception_types"].iloc[0])

# %%
df_clean["exception_types"] = df_clean["exception_types"].apply(
    lambda x: [] if x == "{}" else x.strip("{}").split(",")
)

# %%
exception_counts= Counter(exception for exception_list in df_clean["exception_types"] for exception in exception_list)

exception_counts

# %%
top_exceptions = (pd.Series(exception_counts).sort_values(ascending=False).head(15))
plt.figure(figsize=(10,6))
bars=plt.bar(top_exceptions.index,top_exceptions.values)
plt.bar_label(bars)
plt.ylabel("Frequency")
plt.xticks(rotation=45,ha="right")
plt.title("Top 15 Exception Types Mentioned in Bug Fixes")
plt.show()

# %% [markdown]
# ValueError is the most frequently referenced exception type, appearing in 72 bug-fix patches. This is expected, as it is commonly used in Python libraries to signal invalid inputs or parameter values.
# 
# The distribution is highly skewed. ValueError, TypeError, KeyError, and NotImplementedError occur substantially more often than the remaining exception types, indicating that a relatively small set of exception classes accounts for a large proportion of bug fixes.
# 
# project-specific exception classes such as ExtractorError, InvalidIndexError, and AbstractMethodError. This shows that the feature captures repository-specific semantics alongside generic Python exceptions.
# 
# Most exception types appear only a few times, suggesting a long-tailed distribution in which many specialized exceptions are associated with only a small number of bug fixes.

# %% [markdown]
# ## Which are the Top Modified Modules?

# %%
#Convert back to lists
df_clean["modified_modules"] = df_clean["modified_modules"].apply(
    lambda x:
    [] if x == "{}"
    else [m.strip() for m in x.strip("{}").split(",")]
    )

df_clean["modified_modules"]


# %%
module_counts= Counter( 
    module
    for module_list in df_clean["modified_modules"]
    for module in module_list
    )
module_counts

# %%
top_modules = (pd.Series(module_counts).sort_values(ascending=False).head(15))
plt.figure(figsize=(12,6))
bars=plt.bar(top_modules.index,top_modules.values)
plt.title("Top 15 Most Frequently Modified Modules")
plt.ylabel("Frequency")
plt.xticks(rotation = 45, ha="right")
plt.tight_layout()
plt.show()


# %%
df_clean["modified_modules"] = df_clean["modified_modules"].apply(
    lambda modules: [
        "root" if module == "." else module
        for module in modules
    ]
)

# %% [markdown]
# module_counts= Counter( 
#     module
#     for module_list in df_clean["modified_modules"]
#     for module in module_list
#     )
# module_counts

# %%
top_modules = (pd.Series(module_counts).sort_values(ascending=False).head(15))
plt.figure(figsize=(12,6))
bars=plt.bar(top_modules.index,top_modules.values)
plt.title("Top 15 Most Frequently Modified Modules")
plt.ylabel("Frequency")
plt.xticks(rotation = 45, ha="right")
plt.bar_label(bars)
plt.tight_layout()
plt.show()


# %% [markdown]
# Modules such as pandas/core, pandas/core/indexes, and pandas/core/arrays are among the most frequently modified components, suggesting that bug fixes are concentrated in the core implementation of several projects.
# 
# A relatively small number of modules account for a disproportionately large number of bug fixes, indicating that some components undergo maintenance more frequently than others.
# 
# Unlike file-level features, the engineered modified_modules feature captures subsystem-level information (e.g., pandas/core/groupby, keras/engine), providing higher-level semantic context that can support repository-aware retrieval.
# 
# The most frequently modified modules span several independent projects, including Pandas, Keras, Luigi, Tornado, and YouTube-DL, demonstrating that the dataset covers a diverse set of Python ecosystems rather than being dominated by a single repository

# %% [markdown]
# ## Which exception types are most common in each repository?

# %%
exception_project = df_clean.explode("exception_types").reset_index(drop=True)
exception_project

# %%
top10=exception_project["exception_types"].value_counts().head(10).index
top10

# %%
exception_project = exception_project[exception_project["exception_types"].isin(top10)]
cross = pd.crosstab(exception_project["project"],exception_project["exception_types"])
cross

# %%
plt.figure(figsize=(12,7))
sns.heatmap(cross,annot=True,fmt="d",cmap="Blues")
plt.title("Exception types by Project")
plt.xlabel("Exception Type")
plt.ylabel("Project")

plt.tight_layout()
plt.show()

# %% [markdown]
# The most obvious observation is that Pandas contributes the majority of the occurrences for several common exception types.Many patches in the BugsInPy dataset involving these exception types originate from the Pandas project.
# 
# ValueError appears in many repositories, This indicates that ValueError is a broadly used built-in exception rather than being associated with a specific project.
# 
# Certain exception types appear almost exclusively in one repository.
# * KeyError is almost entirely associated with Pandas in this dataset.
# * OverflowError also appears only in Pandas among the top exceptions.
# shows that some semantic features are repository-dependent.
# 
# This shows that exception types are not uniformly distributed across repositories. Combining repository information with extracted exception types may therefore improve repository-aware bug retrieval by narrowing the search space to semantically similar historical fixes.

# %% [markdown]
# 


