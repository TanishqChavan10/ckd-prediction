"""
main.py
description: training and evaluating various models
important: explanation on parameters choosing, NN architecture choosing and much more, in the PDF's provided in github:
"https://github.com/eliordadon/ckd-predictor-ml-analysis"
important2: there are sections in the main() functions, you cannot (!) run more than one section at a time,
is it splitted so you can see how to run each one, if you want to run a particular section (for example the Neural
network one), then you need to comment all the other sections.
"""

import logging
import joblib
from helper import load_data, split_data, split_withval, preprocessing, preprocessing_whole
from models import *
from neural_network import run_eval_nn


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='ckd_predictor.log')
    logging.info('Main Started')

    try:
        # load, EDA (applicable for all kinds of models)
        # load
        data_df = load_data('datasetVariations/CKDF_noQmarks_unindexed.csv')
        logging.info('Data loaded successfully')
        # EDA

        # ==== SECTION 1 - SUPERVISED ML (not deep learning) models ====
        # logging.info('Section 1 Started')
        # X_train, X_test, y_train, y_test = split_data(data_df, test_size=0.2)
        # X_train_p, y_train_p, X_test_p, y_test_p = preprocessing(X_train, y_train, X_test, y_test, scaling_method='discretization')
        # rf_info = rand_forest(X_train_p, X_test_p, y_train_p)
        # cart_info = cart_tree(X_train_p, X_test_p, y_train_p)
        # nb_info = naive_bayes(X_train_p, X_test_p, y_train_p)
        # models = {'Random Forest': rf_info, 'CART': cart_info, 'Naive Baysien': nb_info}
        # for model_name, model_info in models.items():
        #     if model_info is not None:
        #         model_info['y_test'] = y_test_p
        #         summary = summarize_results(model_name, model_info)
        #         file_name = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        #         joblib.dump(model_info['model'], file_name)
        # logging.info('Section 1 Ended')
        # ==== End of SECTION 1 ====
        # ==== End of SECTION 1 ====

        # ==== SECTION 2 - UNSUPERVISED clustering models ====
        logging.info('Section 2 Started')
        features = data_df.drop('class', axis=1)
        true_labels = data_df['class']
        features_p = preprocessing_whole(features)
        
        # training K-Means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(features_p)
        kmeans_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # evaluation
        # k_means_evaluation(features_p, kmeans_labels, centers) # Commented to prevent plt blocking
        k_means_supervised_evaluation(kmeans_labels, true_labels)
        joblib.dump(kmeans, 'kmeans_model.pkl')
        logging.info('Successfully exported KMeans to kmeans_model.pkl')
        logging.info('Section 2 Ended')

        # ==== SECTION 3 - Predefined Neural Network model ====
        logging.info('Section 3 Started')
        X_train, X_val, X_test, y_train, y_val, y_test = split_withval(data_df, test_size=0.2, val_size=0.25)
        X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p = \
            preprocessing(X_train, y_train, X_val, y_val, X_test, y_test)
        
        input_dim = X_train_p.shape[1]
        run_eval_nn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p, input_dim, plot_to_file=False)
        logging.info('Section 3 Ended')

    except Exception as e:
        logging.error(f'Encountered an error at main(): {e}')
        raise e

if __name__ == '__main__':
    main()
