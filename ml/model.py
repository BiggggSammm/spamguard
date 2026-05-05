import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class SpamClassifier:
    def __init__(self):
        self.model = None
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
        full_text = text + ' ' + link_features
        vec = self.vectorizer.transform([full_text])

        pred = self.model.predict(vec)[0]          # 0 = ham, 1 = spam

        # predict_proba returns [ham_prob, spam_prob]
        # index [0] = probability it is Ham
        # index [1] = probability it is Spam
        # We always want to show the confidence of the WINNING prediction,
        # so we use pred itself as the index — if pred=0 (ham), show ham prob;
        # if pred=1 (spam), show spam prob.
        prob = self.model.predict_proba(vec)[0][pred]

        return int(pred), float(prob)

    def save(self, path='model.pkl'):
        joblib.dump((self.model, self.vectorizer), path)

    def load(self, path='model.pkl'):
        self.model, self.vectorizer = joblib.load(path)