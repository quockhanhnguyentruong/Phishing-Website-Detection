from sklearn import tree # Thuật toán cây quyết định
from sklearn import svm # Thuật toán phân loại
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier # Phân loại
from sklearn.metrics import accuracy_score # Độ chính xác
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import ipaddress
import re
import requests
import tkinter as tk
from utils import generate_data_set

import numpy as np  # Xử lý ngôn ngữ tự nhiên
import sys # Thư viện hệ điều hành trong python


window = tk.Tk()
window.title("Phishing Websites")
window.geometry("1125x400")

def load_data():
    '''
    Load data from CSV file
    '''
    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data
    inputs = training_data[:,:-1]

    # Extract the outputs from the training data
    outputs = training_data[:, -1]

    # This model follow 80-20 rule on dataset
    # Split 80% for traning and 20% testing
    boundary = int(0.8*len(inputs))

    training_inputs, training_outputs, testing_inputs, testing_outputs = train_test_split(inputs, outputs, test_size=0.33)

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs

def run(classifier):
    '''
    Run the classifier to calculate the accuracy score
    '''
    # Load the training data
    train_inputs, test_inputs,train_outputs, test_outputs = load_data()

    # Train the decision tree classifier
    classifier.fit(train_inputs, train_outputs)

    # Use the trained classifier to make predictions on the test data
    predictions = classifier.predict(test_inputs)

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
	#return accuracy
    #print ("Accuracy score using {} is: {}\n".format(name, accuracy))


def abc(name):
    #name = str(entry1.get()) # Lay Cai Nhap Vo
    
    data_set = generate_data_set(name)
		#data_set = generate_data_set(name)

        # Reshape the array
    data_set = np.array(data_set).reshape(1, -1)

        # Load the date
    train_inputs, test_inputs,train_outputs, test_outputs = load_data()

        # Create and train the classifier
    classifier = RandomForestClassifier(n_estimators=500, max_depth=15, max_leaf_nodes=10000)
    classifier.fit(train_inputs, train_outputs)

    return classifier.predict(data_set)
	
	
def abc1():
	greeting=abc(str(entry1.get()))
	
	if(greeting == 1):
		#########################################################################
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Train the decision tree classifier
		classifier = RandomForestClassifier(n_estimators=500, max_depth=15, max_leaf_nodes=10000)
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using Random forest is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		#########################################################################
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Train the decision tree classifier
		classifier = svm.OneClassSVM(gamma='auto')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using One Class SVM is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 1
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Decision tree
		classifier = tree.DecisionTreeClassifier()
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using Decision tree is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 2
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Linear SVC classifier
		classifier = svm.SVC(kernel='linear')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with linear kernel is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 3
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# RBF SVC classifier
		classifier = svm.SVC(kernel='rbf')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with rbf kernel is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 4
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Custom SVC classifier 1
		classifier = svm.SVC(decision_function_shape='ovo', kernel='linear')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with ovo shape 1 is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 5
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Custom SVC classifier 2
		classifier = svm.SVC(decision_function_shape='ovo', kernel='rbf')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with ovo shape 2 is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 6
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# NuSVC classifier
		classifier = svm.NuSVC()
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using NuSVC is: " + str(accuracy) + "\n"
		greeting_display.insert(tk.END, P1)
		
		######################################################################### 
		
		P = str(entry1.get()) + " Phishing \n"
		greeting_display.insert(tk.END, P)
		
		#########################################################################
		
		url = str(entry1.get())
		if not re.match(r"^https?", url):
			url = "http://" + url
		# Stores the response of the given URL
		try:
			response = requests.get(url)
		except:
			response = ""
		domain = re.findall(r"://([^/]+)/?", url)[0]
		# URL_Length (2)
		if len(url) < 54:
			P2 = "Độ dài URL Nhỏ hơn 54 ký tự \n"
		else:
			P2 = "Độ dài URL Lớn hơn 54 ký tự \n"
		greeting_display.insert(tk.END, P2)

		# Shortining_Service (3)
		if re.findall("bit.ly", url):
			P3 = "URL có rút gọn \n"
		else:
			P3 = "URL không có rút gọn \n"
		greeting_display.insert(tk.END, P3)
		
		# having_At_Symbol (4)
		symbol=re.findall(r'@',url)
		if(len(symbol)==0):
			P4 = "URL không có ký tự đặc biệt @ \n"
		else:
			P4 = "URL có ký tự đặc biệt @ \n"
		greeting_display.insert(tk.END, P4)
		
		# double_slash_redirecting (5)
		symbol1=re.findall(r'/',url)
		if(len(symbol1)>7):
			P5 = "URL có ký tự / lớn hơn 7 lần \n"
		else:
			P5 = "URL có ký tự / nhỏ hơn 7 lần \n"
		greeting_display.insert(tk.END, P5)
		
		# Prefix_Suffix (6)
		if re.findall(r"//[^\-]+-[^\-]+", url):
			P6 = "URL có ký tự đặc biệt - giữ các ký tự \n"
		else:
			P6 = "URL không có ký tự đặc biệt - giữ các ký tự \n"
		greeting_display.insert(tk.END, P6)
	
		# having_Sub_Domain (7)
		if len(re.findall("\.", url)) == 1:
			P7 = "URL có dấu . chỉ 1 lần giữ các ký tự \n"
		else:
			P7 = "URL có dấu . lớn hơn 1 lần giữ các ký tự \n"
		greeting_display.insert(tk.END, P7)
		
		# port (8)
		try:
			port = domain.split(":")[1]
			if port:
				P8 = "URL không chứa cổng\n"
			else:
				P8 = "URL có chứa cổng\n"
		except:
			P8 = "URL có chứa cổng\n"
		greeting_display.insert(tk.END, P8)
		
		# HTTPS_token (9)
		if re.findall("https\-", domain):
			P9 = "URL không có chuẩn SSL\n"
		else:
			P9 = "URL có chuẩn SSL\n"
		greeting_display.insert(tk.END, P9)
		
		# Submitting_to_email (10)
		if re.findall(r"[mail\(\)|mailto:?]", str(response)):
			P10 = "URL có gán email submit\n"
		else:
			P10 = "URL không có gán email submit\n"
		greeting_display.insert(tk.END, P10)
		
		# on_mouseover (13)
		if re.findall("<script>.+onmouseover.+</script>", response.text):
			P13 = "URL có gán onmouseover submit\n"
		else:
			P13 = "URL không có gán onmouseover submit\n"
		greeting_display.insert(tk.END, P13)
		# Chuyển hướng (12)
		if len(response.history) <= 1:
			P12 = "URL không chuyển hướng\n"
		elif (len(response.history) >= 2 and len(response.history) < 4):
			P12 = "URL chuyển hướng >= 2 và <= 4 (không xác định được)"
		else:
			P12 = "URL có dữ liệu chuyển hướng truy cập"
		greeting_display.insert(tk.END, P12)
		# popUpWidnow (15)
		if re.findall(r"alert\(", response.text):
			P15 = "URL có chứa pop-up window"
		else:
			P15 = "URL không có pop-up window"
	else:
		#########################################################################
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Train the decision tree classifier
		classifier = RandomForestClassifier(n_estimators=500, max_depth=15, max_leaf_nodes=10000)
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using Random forest is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		#########################################################################
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Train the decision tree classifier
		classifier = svm.OneClassSVM(gamma='auto')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using One Class SVM is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		########################################################################
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Decision tree
		classifier = tree.DecisionTreeClassifier()
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using Decision tree is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		######################################################################### 2
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Linear SVC classifier
		classifier = svm.SVC(kernel='linear')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with linear kernel is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		######################################################################### 3
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# RBF SVC classifier
		classifier = svm.SVC(kernel='rbf')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with rbf kernel is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		######################################################################### 4
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Custom SVC classifier 1
		classifier = svm.SVC(decision_function_shape='ovo', kernel='linear')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with ovo shape 1 is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		######################################################################### 5
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# Custom SVC classifier 2
		classifier = svm.SVC(decision_function_shape='ovo', kernel='rbf')
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using SVC with ovo shape 2 is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		######################################################################### 6
		# Load the training data
		train_inputs, test_inputs,train_outputs, test_outputs = load_data()

		# NuSVC classifier
		classifier = svm.NuSVC()
		classifier.fit(train_inputs, train_outputs)

		# Use the trained classifier to make predictions on the test data
		predictions = classifier.predict(test_inputs)

		# Print the accuracy (percentage of phishing websites correctly predicted)
		accuracy = 100.0 * accuracy_score(test_outputs, predictions)

		P1 = "Accuracy score using NuSVC is: " + str(accuracy) + "\n"
		greeting_display1.insert(tk.END, P1)
		
		#########################################################################
		
		P = str(entry1.get()) + " Non Phishing \n"
		greeting_display1.insert(tk.END, P)
		
		P0 = " WHY? \n"
		greeting_display1.insert(tk.END, P0)
		
		#########################################################################
		url = str(entry1.get())
		if not re.match(r"^https?", url):
			url = "http://" + url
		# Stores the response of the given URL
		try:
			response = requests.get(url)
		except:
			response = ""
		domain = re.findall(r"://([^/]+)/?", url)[0]
		# URL_Length (2)
		if len(url) < 54:
			P2 = "Độ dài URL Nhỏ hơn 54 ký tự \n"
		else:
			P2 = "Độ dài URL Lớn hơn 54 ký tự \n"
		greeting_display1.insert(tk.END, P2)

		# Shortining_Service (3)
		if re.findall("bit.ly", url):
			P3 = "URL có rút gọn \n"
		else:
			P3 = "URL không có rút gọn \n"
		greeting_display1.insert(tk.END, P3)
		
		# having_At_Symbol (4)
		symbol=re.findall(r'@',url)
		if(len(symbol)==0):
			P4 = "URL không có ký tự đặc biệt @ \n"
		else:
			P4 = "URL có ký tự đặc biệt @ \n"
		greeting_display1.insert(tk.END, P4)
		
		# double_slash_redirecting (5)
		symbol1=re.findall(r'/',url)
		if(len(symbol1)>7):
			P5 = "URL có ký tự / lớn hơn 7 lần \n"
		else:
			P5 = "URL có ký tự / nhỏ hơn 7 lần \n"
		greeting_display1.insert(tk.END, P5)
		
		# Prefix_Suffix (6)
		if re.findall(r"//[^\-]+-[^\-]+", url):
			P6 = "URL có ký tự đặc biệt - giữ các ký tự \n"
		else:
			P6 = "URL không có ký tự đặc biệt - giữ các ký tự \n"
		greeting_display1.insert(tk.END, P6)
	
		# having_Sub_Domain (7)
		if len(re.findall("\.", url)) == 1:
			P7 = "URL có dấu . chỉ 1 lần giữ các ký tự \n"
		else:
			P7 = "URL có dấu . lớn hơn 1 lần giữ các ký tự \n"
		greeting_display1.insert(tk.END, P7)
		
		# port (8)
		try:
			port = domain.split(":")[1]
			if port:
				P8 = "URL không chứa cổng\n"
			else:
				P8 = "URL có chứa cổng\n"
		except:
			P8 = "URL có chứa cổng\n"
		greeting_display1.insert(tk.END, P8)
		
		# HTTPS_token (9)
		if re.findall("https\-", domain):
			P9 = "URL không có chuẩn SSL\n"
		else:
			P9 = "URL có chuẩn SSL\n"
		greeting_display1.insert(tk.END, P9)
		
		# Submitting_to_email (10)
		if re.findall(r"[mail\(\)|mailto:?]", str(response)):
			P10 = "URL có gán email submit\n"
		else:
			P10 = "URL không có gán email submit\n"
		greeting_display1.insert(tk.END, P10)
		
		# on_mouseover (13)
		if re.findall("<script>.+onmouseover.+</script>", response.text):
			P13 = "URL có gán onmouseover submit\n"
		else:
			P13 = "URL không có gán onmouseover submit\n"
		greeting_display1.insert(tk.END, P13)
		# Chuyển hướng (12)
		if len(response.history) <= 1:
			P12 = "URL không chuyển hướng\n"
		elif (len(response.history) >= 2 and len(response.history) < 4):
			P12 = "URL chuyển hướng >= 2 và <= 4 (không xác định được)"
		else:
			P12 = "URL có dữ liệu chuyển hướng truy cập"
		greeting_display1.insert(tk.END, P12)
		# popUpWidnow (15)
		if re.findall(r"alert\(", response.text):
			P15 = "URL có chứa pop-up window"
		else:
			P15 = "URL không có pop-up window"

label1= tk.Label(text = "Enter Phishing websites:")
label1.grid(column = 0,row = 0)

entry1 = tk.Entry(width=60)
entry1.grid(column = 1,row = 0)

button1 = tk.Button(text = "    Chẩn đoán!    ",command = abc1)
button1.grid(column = 0,row = 1)
#Label Phishing
label2= tk.Label(text = "Phishing websites:")
label2.grid(column = 0,row = 2)
#Label Non Phishing
label4= tk.Label(text = "Non Phishing websites:")
label4.grid(column = 1,row = 2)
# Group Box Phishing
greeting_display = tk.Text(master=window, height=20, width=70)
greeting_display.grid(column = 0,row = 3) 
# Group Box Non Phishing
greeting_display1 = tk.Text(master=window, height=20, width=70)
greeting_display1.grid(column = 1,row = 3)

window.mainloop()