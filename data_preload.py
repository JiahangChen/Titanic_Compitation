import pandas as pd

def getTrainingData():
  titanic_train_data = pd.read_csv(r'train.csv')
  titanic_test_data = pd.read_csv(r'test.csv')
  titanic_train_data.info()
  titanic_test_data.info()
  y_label_train = titanic_train_data[['Survived']]
  titanic_train_data = titanic_train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
  titanic_test_data = titanic_test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
  y_label_test = pd.read_csv(r'gender_submission.csv')
  y_label_test = y_label_test[['Survived']]
  age_median = titanic_train_data['Age'].mean()
  titanic_train_data['Age'] = titanic_train_data['Age'].fillna(age_median)
  embarked_value = titanic_train_data['Embarked'].value_counts().index[0]
  titanic_train_data['Embarked'] = titanic_train_data['Embarked'].fillna(embarked_value)
  titanic_train_data['Embarked'][titanic_train_data['Embarked'] == 'S'] = int(1)
  titanic_train_data['Embarked'][titanic_train_data['Embarked'] == 'C'] = 2
  titanic_train_data['Embarked'][titanic_train_data['Embarked'] == 'Q'] = 3
  titanic_train_data['Sex'] = [1 if x == 'male' else 0 for x in titanic_train_data.Sex]

  age_median = titanic_test_data['Age'].mean()
  titanic_test_data['Age'] = titanic_test_data['Age'].fillna(age_median)
  embarked_value = titanic_test_data['Embarked'].value_counts().index[0]
  titanic_test_data['Embarked'] = titanic_test_data['Embarked'].fillna(embarked_value)
  titanic_test_data['Embarked'][titanic_test_data['Embarked'] == 'S'] = int(1)
  titanic_test_data['Embarked'][titanic_test_data['Embarked'] == 'C'] = 2
  titanic_test_data['Embarked'][titanic_test_data['Embarked'] == 'Q'] = 3
  titanic_test_data['Sex'] = [1 if x == 'male' else 0 for x in titanic_test_data.Sex]
  return (titanic_train_data, y_label_train,titanic_test_data,y_label_test)
