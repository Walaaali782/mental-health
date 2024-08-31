import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle

warnings.filterwarnings("ignore")

# تحميل البيانات
data = pd.read_csv("survey.csv")

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]

# تنظيف بيانات الجندر
for (row, col) in data.iterrows():
    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    elif str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    elif str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# إزالة القيم غير المرغوب فيها
stk_list = ['A little about you', 'p']
data = data[~data['Gender'].isin(stk_list)]

# ترميز الجندر والقيم النصية الأخرى
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1, 'trans': 2})
data['family_history'] = data['family_history'].map({'No': 0, 'Yes': 1})
data['treatment'] = data['treatment'].map({'No': 0, 'Yes': 1})

# ترميز الأعمدة النصية الأخرى
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# تحويل البيانات إلى مصفوفة
data = np.array(data)

# تحديد X و y
X = data[:, 1:-1]  # تعديل بحيث يشمل كل الأعمدة باستثناء العمود الأخير
y = data[:, -1]

# التأكد من أن y يحتوي فقط على أعداد صحيحة
y = pd.to_numeric(y, errors='coerce')
y = y[~np.isnan(y)]  # إزالة القيم غير الرقمية
y = y.astype('int')

# تحويل X إلى أعداد صحيحة
X = X.astype('int')

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# إنشاء النماذج
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

# إنشاء المكدس
stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
stack.fit(X_train, y_train)

# حفظ النموذج
pickle.dump(stack, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
