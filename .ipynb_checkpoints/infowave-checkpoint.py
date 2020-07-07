import os
import urllib.request
import zipfile
from _commons import warn, error, create_dir_path
import numpy as np

class InfoWaves:
    def __init__(self, variant='ml-100k',
                 pos_rating_threshold=4,
                 data_augmentation_mode='binary_unknown'):    
        np.random.seed(0)
        self.DATA_AUGMENTATION_MODES = ['binary', 'binary_unknown', 'original']
        self.variant = variant
        self.UNKNOWN_CLICK_VAL = 0
        self.POSITIVE_CLICK_VAL = 1
        self.data_augmentation_mode = data_augmentation_mode

        self.num_queries, self.num_pages, self.num_clicks = self._get_num_queries_articles()
        self.num_article_features = 768
        self.orig_R, self.clicks = self._get_clicking_matrix()
        self.R, self.R_mask = self._augment_R(mode=data_augmentation_mode)
        self.page_urls, self.item_genres = self._get_article_info()

        self.current_query_idx = 0  # How many users have I already returned in get_next_user()
        self.user_indexes = np.array(range(self.R.shape[0]))  # order of selection of users to recommend to
        np.random.shuffle(self.user_indexes)  # iterate through users randomly when selecting the next user

        self.arm_feature_dim = self.get_arm_feature_dim()
        print('Statistics about self.R:')
        self.get_statistics()  
        
    def add_random_clicks(self, num_to_each_query=10):
        """
        Adds N random ratings to every user in self.R.
        :param num_to_each_user: Number of random (positive=1 or negative=-1)ratings to be added to each user.
        :return: self.R with added ratings.
        """
        no_articles = self.R.shape[1]
        no_queries = self.R.shape[0]
        
        for u in range(no_queries):
            ids = np.random.randint(no_articles, size=num_to_each_query)
            new_ratings = 1
            self.R[u][ids] = new_ratings
            # print('ids:', ids)
            # print('ratings:', ratings)
            # print('R[u]:', self.R[u])
        return self.R

    def recommend(self, query_id, article_id, fixed_rewards=True, prob_reward_p=0.9):
        """
        Returns reward and updates rating maatrix self.R.
        :param fixed_rewards: Whether to always return 1/0 rewards for already rated items.
        :param prob_reward_p: Probability of returning the correct reward for already rated item.
        :return: Reward = either 0 or 1.
        """
        MIN_PROBABILITY = 0 # Minimal probability to like an item - adds stochasticity

        if self.R[query_id, article_id] == self.POSITIVE_RATING_VAL:
            if fixed_rewards:
                return 1
            else:
                return np.random.binomial(n=1, p=prob_reward_p)  # Bernoulli coin toss
        else:
            item_genres = self.item_genres[article_id]
            query_ratings = self.R[query_id]
            user_pos_rat_idxs = np.argwhere(query_ratings == self.POSITIVE_RATING_VAL).flatten()
            num_known_ratings = len(user_pos_rat_idxs) 
            genre_idxs = np.argwhere(item_genres==1).flatten()

            # Find how much user likes the genre of the recommended movie based on his previous ratings.
            genre_likabilities = []
            for genre_idx in genre_idxs:
                genre_likability = 0
                for query_idx in user_pos_rat_idxs:
                    genre_likability += self.item_genres[item_idx][genre_idx]
                genre_likability /= num_known_ratings
                genre_likabilities.append(genre_likability)

            genre_likabilities = np.array(genre_likabilities)

            # how much user user_id likes the genre of the recommended item item_id
            result_genre_likability = np.average(genre_likabilities)
            binomial_reward_probability = result_genre_likability
            if binomial_reward_probability <= 0:
                #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
                binomial_reward_probability = MIN_PROBABILITY # this could be replaced by small probability

            approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss

            if approx_rating == 1:
                self.R[user_id, item_id] = self.POSITIVE_RATING_VAL
            else:
                self.R[user_id, item_id] = self.NEGATIVE_RATING_VAL

            #return approx_rating
            return approx_rating    
        
    def get_features_of_current_arms(self, t):
        """
        Concatenates article features with query features.
        :param t: Time step = index of query that is being recommended to.
        :return: Matrix of (#arms x #feature_dims) for query t.
        """

        t = t % self.num_users
        query_features = self.R[t]  # vector
        query_features = np.tile(user_features, (self.num_items, 1))  # matrix where each row is R[t]
        article_features = self.item_genres  # matrix
        # arm_feature_dims = item_features.shape[1] + user_features.shape[0]
        arm_features = np.concatenate((query_features, article_features), axis=1)
        return arm_features        
    
    def get_arm_feature_dim(self):
        return self.item_genres.shape[1] + self.R.shape[1]

    def get_uknown_items_of_user(self, user_id):
        user_ratings = self.R[user_id]  # vector
        unknown_item_ids = np.argwhere(user_ratings == self.UNKNOWN_RATING_VAL).flatten()
        return unknown_item_ids

    def get_next_user(self):
        if self.current_user_idx == self.R.shape[0]:
            self.current_user_idx = 0
            np.random.shuffle(self.user_indexes)

        next_user_id = self.user_indexes[self.current_user_idx]
        self.current_user_idx += 1
        return next_user_id
    
    def get_statistics(self):
        """
        Calculates various statistics about given matrix.
        :param R: Rating matrix to get stats about.
        :return: (user_pos_rats, user_neg_rats) = Arrays with numbers of pos/neg ratings per user.
        """
        R = self.R
        total_rats = R.size
        no_rats = len(R[R != self.UNKNOWN_RATING_VAL])
        no_pos_rats = len(R[R == self.POSITIVE_RATING_VAL])
        

        user_pos_rats = np.zeros(shape=(R.shape[0],), dtype=int)
        
        
        for u in range(R.shape[0]):
            user = R[u]
            user_pos_rats[u] = len(user[user == self.POSITIVE_RATING_VAL])


        user_pos_rats_avg = np.average(user_pos_rats)

        user_pos_rats_std = np.std(user_pos_rats)

        print('Number of users:          ', R.shape[0])
        print('Number of items:          ', R.shape[1])
        print('Total number of ratings:  ', total_rats)
        print('Known ratings:            ', no_rats)
        print('Known positive ratings:   ', no_pos_rats)

        print('Ratio of known ratings:   ', no_rats / total_rats)
        print('Ratio of positive ratings:', no_pos_rats / total_rats)
        print('Ratio of negative ratings:', no_neg_rats / total_rats)
        print('Avg number of positive ratings per user: {} +- {}'.format(user_pos_rats_avg, user_pos_rats_std))
        print('Avg number of negative ratings per user: {} +- {}'.format(user_neg_rats_avg, user_neg_rats_std))
        return (user_pos_rats, user_neg_rats)    

    
    def _get_num_users_items(self):
   
        return 25, 41, 41

    def _get_rating_matrix(self):
        r = np.zeros(shape=(self.num_queries, self.num_pages), dtype=float)
        implicit_ratings = np.zeros(shape=(self.num_clicks, 4), dtype=int)

        filename = 'u.data'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        line_idx = 0
        with open(filepath, encoding=ENCODING) as f:
            for line in f:
                chunks = line.split('\t')
                query_id = int(chunks[0]) - 1  # IDs are numbered from 1
                article_id = int(chunks[1]) - 1
                rating = int(chunks[2])
                timestamp = int(chunks[3])

                implicit_ratings[line_idx] = np.array([user_id, item_id, rating, timestamp])
                r[user_id, item_id] = rating
                line_idx += 1

        print("Created a rating matrix of shape={} and dtype={} from {}.".format(r.shape, r.dtype, filename))
        return r, implicit_ratings

    def _get_item_info(self):
        genres = np.zeros(shape=(self.num_pages, self.num_article_features), dtype=float)
        titles = np.empty(shape=(self.num_items,), dtype=object)

        filename = 'u.item'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)

        with open(filepath, encoding=STRING_ENCODING) as f:
            for line in f:
                chunks = line.split('|')
                movie_id = int(chunks[0]) - 1  # IDs are numbered from 1
                title = chunks[1]
                titles[movie_id] = title
                # ignore release dates and url
                for i in range(len(chunks) - 5):
                    genres[movie_id, i] = int(chunks[i + 5])

        print("Created a genre matrix of shape={} and dtype={} from {}.".format(genres.shape, genres.dtype, filename))
        print("Created a titles matrix of shape={} and dtype={} from {}.".format(titles.shape, titles.dtype, filename))
        return titles, genres

    def _get_genre_names(self):
        filename = 'u.genre'
        filepath = os.path.join(DATA_DIR, DATASET_NAME, self.variant, filename)
        with open(filepath, encoding=ENCODING) as f:
            genres = [x.strip().split('|')[0] for x in f.readlines()]
            genres = [x for x in genres if len(x) > 0]
            # print(genres)
        return genres
