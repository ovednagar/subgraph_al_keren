from DataLoader.refael_data_loader import RefaelDataLoader
import os

from ml_communities import MLCommunities


class RefaelLearner:
    def __init__(self):
        self._params = {
            'database': 'Refael',
            'date_format': "%Y-%m-%d",  # Refael
            'directed': True,
            'max_connected': False,
            'logger_name': "logger",
            'ftr_pairs': 300,
            'identical_bar': 0.95,
            'context_beta': 1,
        }
        self._database = RefaelDataLoader(self._params['database'], os.path.join("..", "data", "refael_001.csv"), self._params)
        self._ml_learner = MLCommunities(method="RF")

    def run_ml(self):
        while self._database._forward_time():
            beta_matrix, best_pairs, nodes_list, edges_list, labels = self._database._calc_curr_time()
            self._ml_learner.forward_time_data(beta_matrix, best_pairs, nodes_list, edges_list, labels)
            self._ml_learner.run()


if __name__ == "__main__":
    r = RefaelLearner()
    r.run_ml()