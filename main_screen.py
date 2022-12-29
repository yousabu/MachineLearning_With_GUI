from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import metrics



root = Tk()
root.geometry('570x350')
root.title("Main Menu")
root.configure(bg='#FFA500')
ClassLabel = Label(root,padx=20, text="Classification",font=('Times', 24)).grid(row=0,column=1)
ClassLabel = Label(root, text="").grid(row=1,column=1)
ClassLabel = Label(root, text="").grid(row=2,column=1)
is_clicked = "False"



def preprosess(estimator,n_features):
    """This function takes the estimator and returns the reduced X and y from the dataset"""
    global X, y
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    selector = RFE(estimator = estimator, n_features_to_select= n_features)
    X = selector.fit_transform(X, y)
    messagebox.showinfo("Success", "Prosessing Done")
    print(X,y)

def svm_fun():
    root.destroy()

    def svm_train(m,X,y,kernal,test_size):
        global is_clicked,X_test,y_test,clf
        is_clicked = m
        from sklearn import svm
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = svm.SVC(kernel=kernal)
        clf.fit(X_train, y_train)
        messagebox.showinfo("Success", "Successfully Train Data")


    def result(kernal):
        y_pred = clf.predict(X_test)
        res_page = Tk()
        label = Label(res_page,padx=20, text="Results",font=('Times', 24)).grid(row=0,column=5)
        label = Label(res_page,padx=20, text="     ").grid(row=1,column=5)
        Accuracy = Label(res_page,padx=20, text=f" Accuracy: {metrics.accuracy_score(y_test, y_pred)}",font=('Times', 24)).grid(row=2,column=5)
        precition = Label(res_page,padx=20, text=f"Precition: {metrics.precision_score(y_test, y_pred)}",font=('Times', 24)).grid(row=3,column=5)
        fi_score = Label(res_page,padx=20, text=f" F1_Score: {metrics.f1_score(y_test, y_pred)}",font=('Times', 24)).grid(row=4,column=5)
        recall = Label(res_page,padx=20, text=f"  Recall: {metrics.recall_score(y_test, y_pred)}",font=('Times', 24)).grid(row=5,column=5)
        res_page.mainloop()


    def svm_test(kernal):
        if is_clicked == "True":
            from sklearn import svm
            clf = svm.SVC(kernel=kernal)
            messagebox.showinfo("Success", "Successfully Test Data")
        else:
            messagebox.showerror('Error', 'Error: The Data Not Train Please Click Train Button')

    svm_page = Tk()
    svm_page.configure(bg='#C0C0C0')
    svm_page.title("SVM")
    e1 = Entry(svm_page,text='')

    def data():
        global filename
        filename = askopenfilename(initialdir='\\',title = "Select file")
        e1.delete(0, END)
        e1.insert(0, filename)
        e1.config(text=filename)
        filename
        global csv_data
        csv_data = pd.read_csv(filename)
        print(csv_data)

    l1=Label(svm_page, text='Select Data File')
    l1.grid(row=0, column=0)
    e1 = Entry(svm_page,text='')
    e1.grid(row=0, column=1)

    Button(svm_page,text='open', command=data).grid(row=0, column=2)

    Label1 = Label(svm_page, text="Enter number of feature:")
    Label1.grid(pady=10)
    n_feat = Entry(svm_page, width=10,  borderwidth=5)
    n_feat.grid(row=1, column=0, columnspan=3, padx=10, pady=2)

    preprosessing = Button(svm_page, text="Preprocessing", padx=39, pady=10, command=lambda: preprosess(svm.SVC(),int(n_feat.get())), fg="blue")
    preprosessing.grid(row=2,column=1)


    Label1 = Label(svm_page, text="Enter Test Size:")
    Label1.grid(pady=10)
    e_test = Entry(svm_page, width=10,  borderwidth=5)
    e_test.grid(row=3, column=0, columnspan=3, padx=10, pady=2)





    options_list = ["rbf", "linear"]
    value_inside = StringVar(svm_page)
    value_inside.set("Select an Option")
    Label3 = Label(svm_page, text="Enter Kernal Value:")
    Label3.grid(pady=10)
    e_knn = OptionMenu(svm_page,value_inside, *options_list)
    e_knn.grid(row=4, column=0, columnspan=3, padx=10, pady=2)




    button_1 = Button(svm_page, text="Train Data", padx=40, pady=20, command=lambda m="True": svm_train(m,X,y,value_inside.get(),float(e_test.get())))
    button_2 = Button(svm_page, text="Test Data", padx=40, pady=20, command=lambda: svm_test(value_inside.get()))
    button_3 = Button(svm_page, text="Print Result", padx=40, pady=20, command=lambda: result(value_inside.get()))
    button_1.grid(row=6,column=0)
    button_2.grid(row=6,column=1)
    button_3.grid(row=6,column=2)


    svm_page.mainloop

def dt_fun():
    root.destroy()

    def dt_train(m,X,y,test_size):
        global is_clicked,X_test,y_test,clf
        is_clicked = m
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        messagebox.showinfo("Success", "Successfully Train Data")



    def dt_result():
        y_pred = clf.predict(X_test)
        res_page = Tk()
        label = Label(res_page, text="Decision Tree Results",font=('Times', 24)).grid(row=0,column=5)
        label = Label(res_page, text="     ").grid(row=1,column=5)
        Accuracy = Label(res_page, text=f" Accuracy: {metrics.accuracy_score(y_test, y_pred)}",font=('Times', 24)).grid(row=2,column=5)
        precition = Label(res_page, text=f"Precition: {metrics.precision_score(y_test, y_pred)}",font=('Times', 24)).grid(row=3,column=5)
        fi_score = Label(res_page, text=f" F1_Score: {metrics.f1_score(y_test, y_pred)}",font=('Times', 24)).grid(row=4,column=5)
        recall = Label(res_page, text=f" Recall: {metrics.recall_score(y_test, y_pred)}",font=('Times', 24)).grid(row=5,column=5)
        res_page.mainloop()


    def dt_test():
        if is_clicked == "True":
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier()
            messagebox.showinfo("Success", "Successfully Train Data")
        else:
            messagebox.showerror('Error', 'Error: The Data Not Train Please Click Train Button')

    dt_page = Tk()
    dt_page.configure(bg='#C0C0C0')
    dt_page.title("Decision Tree")
    e1 = Entry(dt_page,text='')
    def data():
        global filename
        filename = askopenfilename(initialdir='\\',title = "Select file")
        e1.delete(0, END)
        e1.insert(0, filename)
        e1.config(text=filename)
        filename
        global csv_data
        csv_data = pd.read_csv(filename)
        print(csv_data)
    l1=Label(dt_page, text='Select Data File')
    l1.grid(row=0, column=0)
    e1 = Entry(dt_page,text='')
    e1.grid(row=0, column=1)
    Button(dt_page,text='open', command=data).grid(row=0, column=2)

    Label1 = Label(dt_page, text="Enter number of feature:")
    Label1.grid(pady=10)
    n_feat = Entry(dt_page, width=10,  borderwidth=5)
    n_feat.grid(row=1, column=0, columnspan=3, padx=10, pady=2)



    preprosessing = Button(dt_page, text="Preprocessing", padx=39, pady=10, command=lambda: preprosess(DecisionTreeClassifier(),int(n_feat.get())), fg="blue")
    preprosessing.grid(row=2,column=1)

    Label1 = Label(dt_page, text="Enter Test Size:")
    Label1.grid(pady=10)
    e_test = Entry(dt_page, width=10,  borderwidth=5)
    e_test.grid(row=3, column=0, columnspan=3, padx=10, pady=2)

    button_1 = Button(dt_page, text="Train Data", padx=40, pady=20, command=lambda m="True": dt_train(m,X,y,float(e_test.get())))
    button_2 = Button(dt_page, text="Test Data", padx=40, pady=20, command=dt_test)
    button_3 = Button(dt_page, text="Print Result", padx=40, pady=20, command=dt_result)
    button_1.grid(row=6,column=0)
    button_2.grid(row=6,column=1)
    button_3.grid(row=6,column=2)
    dt_page.mainloop

def knn_fun():
    root.destroy()

    def knn_train(m,X,y,n_neighbors,test_size):
        global is_clicked,X_test,y_test,clf
        is_clicked = m
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        clf = KNeighborsClassifier(n_neighbors)
        clf.fit(X_train, y_train)
        messagebox.showinfo("Success", "Successfully Train Data")


    def knn_result():
        y_pred = clf.predict(X_test)
        res_page = Tk()
        label = Label(res_page,padx=20, text="Results",font=('Times', 24)).grid(row=0,column=5)
        label = Label(res_page,padx=20, text="     ").grid(row=1,column=5)
        Accuracy = Label(res_page,padx=20, text=f" Accuracy: {metrics.accuracy_score(y_test, y_pred)}",font=('Times', 24)).grid(row=2,column=5)
        precition = Label(res_page,padx=20, text=f"Precition: {metrics.precision_score(y_test, y_pred)}",font=('Times', 24)).grid(row=3,column=5)
        fi_score = Label(res_page,padx=20, text=f" F1_Score: {metrics.f1_score(y_test, y_pred)}",font=('Times', 24)).grid(row=4,column=5)
        recall = Label(res_page,padx=20, text=f"  Recall: {metrics.recall_score(y_test, y_pred)}",font=('Times', 24)).grid(row=5,column=5)
        res_page.mainloop()


    def knn_test(n_neighbors):
        if is_clicked == "True":
            clf = KNeighborsClassifier(n_neighbors)
            messagebox.showinfo("Success", "Successfully Test Data")
        else:
            messagebox.showerror('Python Error', 'Error: The Data Not Trained Please Click Train Button')

    knn_page = Tk()
    knn_page.configure(bg='#C0C0C0')
    knn_page.title("KNN")
    e1 = Entry(knn_page,text='')
    def data():
        global filename
        filename = askopenfilename(initialdir='\\',title = "Select file")
        e1.delete(0, END)
        e1.insert(0, filename)
        e1.config(text=filename)
        filename
        global csv_data
        csv_data = pd.read_csv(filename)
        print(csv_data)
    l1=Label(knn_page, text='Select Data File')
    l1.grid(row=0, column=0)
    e1 = Entry(knn_page,text='')
    e1.grid(row=0, column=1)
    Button(knn_page,text='open', command=data).grid(row=0, column=2)

    Label1 = Label(knn_page, text="Enter number of feature:")
    Label1.grid(pady=10)
    n_feat = Entry(knn_page, width=10,  borderwidth=5)
    n_feat.grid(row=1, column=0, columnspan=3, padx=10, pady=2)

    preprosessing = Button(knn_page, text="Preprocessing", padx=39, pady=10, command=lambda: preprosess(DecisionTreeClassifier(),int(n_feat.get())), fg="blue")
    preprosessing.grid(row=2,column=1)

    Label1 = Label(knn_page, text="Enter Test Size:")
    Label1.grid(pady=10)
    e_test = Entry(knn_page, width=10,  borderwidth=5)
    e_test.grid(row=3, column=0, columnspan=3, padx=10, pady=2)

    Label2 = Label(knn_page, text="Enter Neighbors Number:")
    Label2.grid(pady=10)
    n_neb = Entry(knn_page, width=10,  borderwidth=5)
    n_neb.grid(row=4, column=0, columnspan=3, padx=10, pady=2)


    button_1 = Button(knn_page, text="Train Data", padx=40, pady=20, command=lambda m="True": knn_train(m,X,y,int(n_neb.get()),float(e_test.get())))
    button_2 = Button(knn_page, text="Test Data", padx=40, pady=20, command=lambda: knn_test(float(n_neb.get())))
    button_3 = Button(knn_page, text="Print Result", padx=40, pady=20, command=knn_result)
    button_1.grid(row=6,column=0)
    button_2.grid(row=6,column=1)
    button_3.grid(row=6,column=2)


    knn_page.mainloop

def lr_fun():
    root.destroy()

    def split():
        """This function takes the estimator and returns the reduced X and y from the dataset"""
        global X, y
        data = pd.read_csv(filename)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, :-1].values
        messagebox.showinfo("Success", "Prosessing Done")
        print(X,y)

    def lr_train(m,X,y,test_size):
        global is_clicked,X_test,y_test,regr
        is_clicked = m
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        messagebox.showinfo("Success", "Successfully Train Data")


    def lr_result():
        y_pred = regr.predict(X_test)
        res_page = Tk()
        label = Label(res_page,padx=20, text="Results",font=('Times', 24)).grid(row=0,column=5)
        label = Label(res_page,padx=20, text="     ").grid(row=1,column=5)
        Accuracy = Label(res_page,padx=20, text=f" MAE: {mean_absolute_error(y_test, y_pred)}",font=('Times', 24)).grid(row=2,column=5)
        precition = Label(res_page,padx=20, text=f"MSE: {mean_squared_error(y_test, y_pred)}",font=('Times', 24)).grid(row=3,column=5)
        fi_score = Label(res_page,padx=20, text=f" R2_Score: {r2_score(y_test, y_pred)}",font=('Times', 24)).grid(row=4,column=5)
        plt.scatter(X_test, y_test, color ='b')
        plt.plot(X_test, y_pred, color ='k')
        plt.show()
        res_page.mainloop()

    def lr_test():
        if is_clicked == "True":
            regr = LinearRegression()
            messagebox.showinfo("Success", "Successfully Test Data")
        else:
            messagebox.showerror('Python Error', 'Error: The Data Not Trained Please Click Train Button')


    lr_page = Tk()
    lr_page.configure(bg='#C0C0C0')
    lr_page.title("LinearRegression")
    e1 = Entry(lr_page,text='')

    def data():
        global filename
        filename = askopenfilename(initialdir='\\',title = "Select file")
        e1.delete(0, END)
        e1.insert(0, filename)
        e1.config(text=filename)
        filename
        global csv_data
        csv_data = pd.read_csv(filename)
        print(csv_data)

    l1=Label(lr_page, text='Select Data File')
    l1.grid(row=0, column=0)
    e1 = Entry(lr_page,text='')
    e1.grid(row=0, column=1)
    Button(lr_page,text='open', command=data).grid(row=0, column=2)
    preprosessing = Button(lr_page, text="Preprocessing", padx=39, pady=10, command=split, fg="blue")
    preprosessing.grid(row=1,column=1)
    Label1 = Label(lr_page, text="Enter Test Size:")
    Label1.grid(pady=10)
    e_test = Entry(lr_page, width=10,  borderwidth=5)
    e_test.grid(row=2, column=0, columnspan=3, padx=10, pady=2)
    button_1 = Button(lr_page, text="Train Data", padx=40, pady=20, command=lambda m="True": lr_train(m,X,y,float(e_test.get())))
    button_2 = Button(lr_page, text="Test Data", padx=40, pady=20, command=lr_test)
    button_3 = Button(lr_page, text="Print Result", padx=40, pady=20, command=lr_result)
    button_1.grid(row=6,column=0)
    button_2.grid(row=6,column=1)
    button_3.grid(row=6,column=2)

    lr_page.mainloop



svmButton = Button(root, text="    SVM    ",padx=30,pady=20, command=svm_fun).grid(row=3,column=0)
randomButton = Button(root, text="    KNN    ",padx=30,pady=20, command=knn_fun).grid(row=3,column=1)
dtButton = Button(root, text="Decision Tree",padx=30,pady=20, command=dt_fun).grid(row=3,column=2)
ClassLabel = Label(root, text="").grid(row=4,column=1)

ClassLabel = Label(root, text="").grid(row=5,column=1)
ClassLabel = Label(root,padx=20, text="Regression",font=('Times', 24)).grid(row=6,column=1)
ClassLabel = Label(root, text="").grid(row=7,column=1)
ClassLabel = Label(root, text="").grid(row=8,column=1)
dtButton = Button(root, text="LinearRegression",padx=60,pady=40, command=lr_fun).grid(row=9,column=1)




root.mainloop()