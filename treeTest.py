import time
import psycopg2
import numpy as np
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
import sklearn
import graphviz
print("running...")
conn = psycopg2.connect("dbname=group_8 password='' user='' host=www.eecs.uottawa.ca port=15432")
cur = conn.cursor()
cur.execute("SELECT F.reported_date_key, MO.residential, WE.daily_low_temperature, PA.age_group FROM fact F, mobility_dimension MO, weather_dimension WE, patient_dimension PA WHERE F.mobility_key = MO.mobility_key AND F.weather_key = WE.weather_key AND F.patient_key = PA.patient_key GROUP BY f.reported_date_key, MO.residential, WE.weather_key, PA.age_group ORDER BY f.reported_date_key, MO.residential;")
l = cur.fetchall()
print("data retrieved")
rd = []
mr = []
wl = []
pa = []
for (W,X,Y,Z) in l:
	rd.append(W)
	mr.append(X)
	wl.append(Y)
	if(Z == '<20'):
		pa.append(1)
	elif(Z == '20s'):
		pa.append(2)
	elif(Z == '30s'):
		pa.append(3)
	elif(Z == '40s'):
		pa.append(4)
	elif(Z == '50s'):
		pa.append(5)
	elif(Z == '60s'):
		pa.append(6)
	elif(Z == '70s'):
		pa.append(7)
	elif(Z == '80s'):
		pa.append(8)
	elif(Z == '90s'):
		pa.append(9)
	else:
		pa.append(0)
data = []
for i in range(len(pa)):
	data.append([rd[i],mr[i],wl[i]])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, pa, test_size=0.33, random_state=15)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
clf = tree.DecisionTreeClassifier()
start = time.time()
clf = clf.fit(X_train,y_train)
stop = time.time()
print("Time to complete: ", stop-start)
prediction = clf.predict(X_test)
print(classification_report(y_test,prediction,zero_division=1))
print(prediction)
print(y_test)
#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = graphviz.Source(dot_data)
#graph.render("treeModel")

cur.close()
conn.close()
