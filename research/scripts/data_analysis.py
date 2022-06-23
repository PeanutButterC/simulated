from paths import SIMULATED_DATA_ROOT
import pandas as pd
from sklearn.dummy import DummyClassifier
from simulated_data import DATA_COLS
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def print_counts():
    stores = {f: pd.HDFStore(f'{SIMULATED_DATA_ROOT}/{f}.h5') for f in ['train', 'dev', 'test']}
    data = stores['dev']['df']
    counts = data['hand_state'].value_counts().sort_index().to_numpy()
    print(', '.join([str(i) for i in counts]))

    data['transition'] = data['hand_state'].diff().abs() > 0
    transitions = data[data['transition']]
    transition_counts = transitions['hand_state'].value_counts().sort_index().to_numpy()
    print(', '.join([str(i) for i in transition_counts]))


def eval(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
    row = f'\
        <tr>\n\
            <th>title</th>\n\
            <td>{acc:.2f}</td>\n\
            <td>{scores[0]:.2f}</td>\n\
            <td>{scores[1]:.2f}</td>\n\
            <td>{scores[2]:.2f}</td>\n\
        </tr>\
    '
    print(row)


def dummy_classifier():
    data = pd.HDFStore(f'{SIMULATED_DATA_ROOT}/dev.h5')['df']
    x = data[DATA_COLS].to_numpy()
    y = data['hand_state'].to_numpy()

    cls = DummyClassifier(strategy='stratified')
    cls.fit(x, y)
    y_pred = cls.predict(x)
    eval(y, y_pred)

    cls = DummyClassifier(strategy='uniform')
    cls.fit(x, y)
    y_pred = cls.predict(x)
    eval(y, y_pred)


if __name__ == '__main__':
    # print_counts()
    dummy_classifier()
