import matplotlib.pyplot as plt
import matplotlib
import psycopg2
import numpy as np
from sklearn import tree
import graphviz
print("running...")
conn = psycopg2.connect("dbname=group_8 host=www.eecs.uottawa.ca port=15432")
cur = conn.cursor()
cur.execute("SELECT F.reported_date_key, MO.residential, PA.age_group FROM fact F, mobility_dimension MO, weather_dimension WE, patient_dimension PA WHERE F.mobility_key = MO.mobility_key AND F.weather_key = WE.weather_key AND F.patient_key = PA.patient_key GROUP BY f.reported_date_key, MO.residential, WE.weather_key, PA.age_group ORDER BY f.reported_date_key, MO.residential;")
l = cur.fetchall()
print("data retrieved")
x = []
y = []
z = []
for (X,Y,Z) in l:
	x.append(X)
	y.append(Y)
	if(Z == '<20'):
		z.append(1)
	elif(Z == '20s'):
		z.append(2)
	elif(Z == '30s'):
		z.append(3)
	elif(Z == '40s'):
		z.append(4)
	elif(Z == '50s'):
		z.append(5)
	elif(Z == '60s'):
		z.append(6)
	elif(Z == '70s'):
		z.append(7)
	elif(Z == '80s'):
		z.append(8)
	elif(Z == '90s'):
		z.append(9)
	else:
		z.append(0)
w = []
for i in range(len(x)):
	w.append([x[i],y[i]])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(w,z)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("test2")
cur.close()
conn.close()