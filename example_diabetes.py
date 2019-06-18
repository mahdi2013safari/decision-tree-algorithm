from __future__ import print_function
import os
import subprocess

from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

d = defaultdict(LabelEncoder)
import pandas as pd

d = defaultdict(LabelEncoder)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_dataset():
    if os.path.exists("diabetes2.csv"):
        print("-- diabetes2.csv found in path")
        df = pd.read_csv("diabetes2.csv", index_col=0)
    else:
        print("-- dataset diabates no found in path")
        df = pd.read_csv("diabetes2.csv", index_col=0)

        with open("diabetes2.csv", 'w') as f:
            print("-- writing to local diabetes.csv file")
            df.to_csv(f)
    return df
df = get_dataset()

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

#encode the class lable and get features list
df2 , targets = encode_target(df, "Outcome")
features = list(df2.columns[:8])

y = df2["Target"]
X = df2[features]
dt = DecisionTreeClassifier()
dt.fit(X, y)


print("* targets", targets, sep="\n", end="\n\n")
print("* features:", features, sep="\n")



# 0 = No // 1 = Yes
#[['No','Yes']]
#pre=dt.predict([[13,145,82,19,110,22.2,0.245,57]])
##predict_Probability= dt.predict_proba([[1,145,122,19,110,22.2,0.245,30]])

path = dt.decision_path([[13,145,82,19,110,22.2,0.245,57]])

#print("Predict probal = ",pre)
#print("probability tree = ", predict_Probability)
print("decision path = ",path)
def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,node_ids=True,
                        feature_names=feature_names,class_names=["Yes","NO"])
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
visualize_tree(dt, features)

