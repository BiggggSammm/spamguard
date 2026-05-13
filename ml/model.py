import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class SpamClassifier:
    def __init__(self):
        self.model      = None
        self.vectorizer = None

    def train(self, df):
        X = df['text'] + ' ' + df.get('link_features', '')
        y = df['label']
        self.vectorizer = TfidfVectorizer()
        X_vec = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict(self, text, link_features=''):
        vec   = self.vectorizer.transform([text + ' ' + link_features])
        pred  = self.model.predict(vec)[0]          # 0 or 1
        proba = self.model.predict_proba(vec)[0]    # [prob_ham, prob_spam]
        # Get probability of the predicted class
        class_index = list(self.model.classes_).index(pred)
        prob = proba[class_index]
        return int(pred), float(prob)

    def save(self, path='model.pkl'):
        joblib.dump((self.model, self.vectorizer), path)

    def load(self, path='model.pkl'):
        self.model, self.vectorizer = joblib.load(path)