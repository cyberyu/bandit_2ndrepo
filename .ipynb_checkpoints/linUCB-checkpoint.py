from _commons import warn, error, create_dir_path
import numpy as np
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import copy 

class LinUCB:
    def __init__(self, alpha, max_items=500, allow_selecting_known_arms=True, fixed_rewards=True,
                 prob_reward_p=0.9):
        
        
        # input files 
        # question_article_ratings.pkl    matrix of all clicks by question x infowave pages
        
        self.alpha = alpha
        self.fixed_rewards = fixed_rewards   #using fix rewards
        self.prob_reward_p = prob_reward_p   
        
        self.UNKNOWN_RATING_VAL = 0
        self.POSITIVE_RATING_VAL = 1
        self.NEGATIVE_RATING_VAL = -1
            
        
        self.type="sampling"
        
        if self.type=="all_intents":
            self.num_articles=321         # number of queries
            self.arm_feature_dim=1764   # the dimensionality of arm features (query features (768) + page featuers (*number of query ratings))
            self.num_queries=996         # number of articles
            
        elif self.type=="sampling":
            self.num_articles=10         # number of queries
            self.arm_feature_dim=818   # the dimensionality of arm features (query features (768) + page featuers (*number of query ratings))
            self.num_queries=50         # number of articles
            
        else:
            self.num_articles=25         # number of queries
            self.arm_feature_dim=809   # the dimensionality of arm features (query features (768) + page featuers (*number of query ratings))
            self.num_queries=41         # number of articles

        #dataarray = pickle.load(open('dataarray.pkl','rb'))
        self.R=np.zeros((self.num_queries,self.num_articles))  # load the correct query-article mappings for intent 1

        
        if self.type=="allintents":
            ratings = pickle.load(open('./data/all_question_article_ratings.pkl','rb'))
        elif self.type=="sampling":    
            ratings = pickle.load(open('./data/sample_by_question_questions_article_ratings.pkl','rb'))
        else:    
            ratings = pickle.load(open('./data/question_article_ratings.pkl','rb'))
            
        for i,j in ratings:
            self.R[i,j]=1
        
        self.oldR = copy.deepcopy(self.R)
        self.users_with_unrated_items = np.array(range(self.num_queries))  # number of unrated items. here is probably wrong
        self.monitored_user = np.random.choice(self.users_with_unrated_items)
        self.allow_selecting_known_arms = allow_selecting_known_arms
        self.d = self.arm_feature_dim
        self.b = np.zeros(shape=(self.num_queries, self.d))
        self.query_titles, self.query_embeddings =self._get_query_info()
#         self.article_titles, self.article_pca =self._get_article_info()

        # More efficient way to create array of identity matrices of length num_items
        print("\nInitializing matrix A of shape {} which will require {}MB of memory."
              .format((self.num_queries, self.d, self.d), 8 * self.num_queries * self.d * self.d / 1e6))
        self.A = np.repeat(np.identity(self.d, dtype=float)[np.newaxis, :, :], self.num_queries, axis=0)
        #(50, 818, 818) queries, fd, fd
        print("\nLinUCB successfully initialized.")

    #  input para (article_id, unknown_query_ids, verbosity)
    def choose_arm(self, t, unknown_query_ids, verbosity):
        """
        Choose an arm to pull = query to matched to infowavepage t that it has not been matched yet.
        :param t: page_id for queries to matched to.
        :param unknown_item_ids: Indexes of query ids that page t has not rated yet.
        :return: Received reward for matched query = 1/0 = page actually matched/unmatched query.
        """
        A = self.A
        b = self.b
        
        arm_features = self.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        p_t -= 9999  # I never want to select the already rated items
        query_ids = unknown_query_ids

        if self.allow_selecting_known_arms:
            query_ids = range(self.num_queries)
            p_t += 9999

        for a in query_ids:  # iterate over all arms
            x_ta = arm_features[a].reshape(arm_features[a].shape[0], 1)  # make a column vector
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + self.alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))

        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("Page {} has max p_t={}, p_t={}".format(t, max_p_t, p_t))

        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        # So I will get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to query t

        # observed reward = 1/0
        r_t = self.recommend(query_id=a_t, page_id=t, fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)

        if verbosity >= 2:
            print("Query {} choosing item {} with p_t={} reward {}".format(t, a_t, p_t[a_t], r_t))

        x_t_at = arm_features[a_t].reshape(arm_features[a_t].shape[0], 1)  # make a column vector
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()  # turn it back into an array because b[a_t] is an array

        return r_t

#     def run_epoch(self, verbosity=2):
#         """
#         Call choose_arm() for each query in the dataset.
#         :return: Average received reward.
#         """
#         rewards = []
#         start_time = time.time()

#         for i in range(self.num_articles):
#             start_time_i = time.time()
#             #user_id = self.get_next_user()
#             query_id = i
#             unknown_page_ids = self.get_uknown_items_of_user(query_id)

#             if self.allow_selecting_known_arms == False:
#                 if query_id not in self.users_with_unrated_items:
#                     continue

#                 if unknown_page_ids.size == 0:
#                     print("Query {} has no more unknown ratings, skipping him.".format(query_id))
#                     self.users_with_unrated_items = self.users_with_unrated_items[
#                         self.users_with_unrated_items != query_id]
#                     continue

#             rewards.append(self.choose_arm(query_id, unknown_page_ids, verbosity))
#             time_i = time.time() - start_time_i
#             if verbosity >= 2:
#                 print("Choosing arm for query {}/{} ended with reward {} in {}s".format(i, self.num_queries,
#                                                                                        rewards[i], time_i))

#         total_time = time.time() - start_time
#         avg_reward = np.average(np.array(rewards))
#         auc_score = self.calculate_auc()
        
#         return avg_reward, auc_score, total_time

    
    def run_epoch(self, verbosity=2):
        """
        Call choose_arm() for each article in the dataset.
        :return: Average received reward.
        """
        rewards = []
        start_time = time.time()

        for i in range(self.num_articles):
            
            
            #print('article id is '+ str(i))
            start_time_i = time.time()
            #user_id = self.get_next_user()
            article_id = i
            unknown_query_ids = self.get_uknown_queries_of_page(article_id)

            if self.allow_selecting_known_arms == False:
                if article_id not in self.articles_with_unmatched_queries:
                    continue

                if unknown_query_ids.size == 0:
                    print("Page {} has no more unknown query matchings, skipping it.".format(article_id))
                    self.articles_with_unmatched_queries = self.articles_with_unmatched_queries[
                        self.articles_with_unmatched_queries != article_id]
                    continue

            rewards.append(self.choose_arm(article_id, unknown_query_ids, verbosity))
            time_i = time.time() - start_time_i
            if verbosity >= 2:
                print("Choosing arm for query {}/{} ended with reward {} in {}s".format(i, self.num_queries,
                                                                                       rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        auc_score = self.calculate_auc()
        
        return avg_reward, auc_score, total_time
    
    def run(self, num_epochs, verbosity=1):
        """
        Runs run_epoch() num_epoch times.
        :param num_epochs: Number of epochs = iterating over all users.
        :return: List of average rewards per epoch.
        """
        self.articles_with_unmatched_queries = np.array(range(self.num_articles))  # this list out all the queries potentially for a page to match
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        auc_scores = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], auc_scores[i], total_time = self.run_epoch(verbosity)
            
            if verbosity >= 1:
                print(
                    "Finished epoch {}/{} with avg reward {}, auc score {} in {}s".format(i, num_epochs, avg_rewards[i], auc_scores[i], total_time))
        return avg_rewards, auc_scores

    def get_uknown_queries_of_page(self, article_id):
        
        query_ratings = self.R[:,article_id]  # vector
        unknown_query_ids = np.argwhere(query_ratings == self.UNKNOWN_RATING_VAL).flatten()
        return unknown_query_ids    
    
    
    def get_features_of_current_arms(self, t):
        """
        Concatenates query features with page features.
        :param t: Time step = index of user that is being recommended to.
        :return: Matrix of (#arms x #feature_dims) for user t.
        """
        # in the original settings, an arm is the article, recommendation is from article to user
        # so in this get arm function, it originally concatenate a single user_feature (rating vector) with multiple article features (genre vectors)
        
        # in our settings, an arm is the query, because I am thinking page is somewhat static, so we are recommendating queries to pages. Page is a user role
        # so for our get arm function, it should concatenate a single page_feature (historical clicks of queries) with queries features (embeddings)
        t = t % self.num_queries
        
#         article_features = self.R[t]   # rating of No.t article x all users
#         article_features = np.tile(article_features, (self.num_queries, 1))
        query_features = self.query_embeddings
        article_features=self.R[:,t].T
        article_features = np.tile(article_features, (self.num_queries, 1))
        arm_features = np.concatenate((query_features, article_features), axis=1)
        print(arm_features.shape)
        print(article_features.shape)
        return arm_features    
    
    def get_featuers_of_new_arms_oos(self, t, new_query_embeddings):
        # this function create an arm feature vector for an out-of-sample query
        # it concatenate the new query embeddings (query features) with the ratings (click vecotr) of a page w.r.t. all in-sample queries
        article_features=self.R[:,t].T
        arm_features = np.concatenate((new_query_embeddings, article_features), axis=0)
        return arm_features
        
    def _get_page_info(self):
        
        page_id = pickle.load(open('',''))
        article_features = pickle.load(open('',''))

        features = np.zeros(shape=(self.num_queries, 768), dtype=float)
        titles = np.empty(shape=(self.num_queries,), dtype=object)
        
    def _get_query_info(self):
        
        if self.type=='1st_intent':  # only data for first intent
            question_id = pickle.load(open('data/question_25_1stintent_dict.pkl','rb'))
            question_features = pickle.load(open('data/question_25_1stintent_title.pkl','rb'))

            features = np.zeros(shape=(self.num_queries, 768), dtype=float)
            titles = np.empty(shape=(self.num_queries,), dtype=object)

            for k,v in question_id.items():
                titles[v]=k

                # average multiple sentences 
                # print(question_features[k].shape[0],question_features[k].shape[1])
                if (question_features[k].shape[0]!=1) & (question_features[k].shape[1]==768):
                    features[v,:]=np.average(question_features[k],axis=0)
                else:
                    features[v,:]=question_features[k]
                    
        elif self.type=="sampling":   
            question_id = pickle.load(open('data/sample_by_question_questions_dict.pkl','rb'))
            question_features = pickle.load(open('data/sample_by_question_questions_features.pkl','rb'))
            
            features = np.zeros(shape=(self.num_queries, 768), dtype=float)
            titles = np.empty(shape=(self.num_queries,), dtype=object)
            
            for k,v in question_id.items():
                #k is the question text,  v is the incremental ID
                titles[v]=k

                # average multiple sentences 
                # print(question_features[k].shape[0],question_features[k].shape[1])
                #print(question_features[k].shape)
                
                if (question_features[k].shape[0]!=768):
                    features[v,:]=np.average(question_features[k],axis=0)
                else:
                    features[v,:]=question_features[k]
            
        else:   # for all intents     
            
            question_id = pickle.load(open('data/all_questions_dict.pkl','rb'))
            question_features = pickle.load(open('data/all_questions_title.pkl','rb'))
            
            features = np.zeros(shape=(self.num_queries, 768), dtype=float)
            titles = np.empty(shape=(self.num_queries,), dtype=object)
            
            for k,v in question_id.items():
                #k is the question text,  v is the incremental ID
                titles[v]=k

                # average multiple sentences 
                # print(question_features[k].shape[0],question_features[k].shape[1])
                #print(question_features[k].shape)
                
                if (question_features[k].shape[0]!=768):
                    features[v,:]=np.average(question_features[k],axis=0)
                else:
                    features[v,:]=question_features[k]
            
        return titles, features    

    def recommend(self, query_id, page_id, fixed_rewards=True, prob_reward_p=0.9):
        """
        Returns reward and updates rating maatrix self.R.
        :param fixed_rewards: Whether to always return 1/0 rewards for already rated items.
        :param prob_reward_p: Probability of returning the correct reward for already rated item.
        :return: Reward = either 0 or 1.
        """
        MIN_PROBABILITY = 0 # Minimal probability to like an item - adds stochasticity
#         print('query_id' + str(query_id))
#         print('page_id' + str(page_id))
        if self.R[query_id, page_id] == self.POSITIVE_RATING_VAL:
            if fixed_rewards:
                return 1
            else:
                return np.random.binomial(n=1, p=prob_reward_p)  # Bernoulli coin toss
        else:
            
            # the goal is to update a missing ""
            current_query_features = self.query_embeddings[query_id]  #get the features
            
            
            # find out for an article, what others questions are rated relevant
            article_ratings = self.R[:,page_id]  #get all ratings by article id, it is a column
            query_pos_rat_idxs = np.argwhere(article_ratings == self.POSITIVE_RATING_VAL).flatten() # get all other positive ratings of the same article
            num_known_ratings = len(query_pos_rat_idxs)  # length of all other positive ratings
            
            genre_likabilities = []
            
            
            for query_idx in query_pos_rat_idxs:
                # calculate the similarty between query
#                 print(current_query_features.shape)
#                 print(self.query_embeddings[query_idx].shape)
                genre_likabilities.append(cosine_similarity(current_query_features.reshape(-1,1), self.query_embeddings[query_idx].reshape(-1,1)))
            
            
            result_genre_likability = np.average(genre_likabilities)
            
            binomial_reward_probability = result_genre_likability
            if binomial_reward_probability <= 0:
                #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
                binomial_reward_probability = MIN_PROBABILITY # this could be replaced by small probability

            approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss

            if approx_rating == 1:
                self.R[query_id, page_id] = self.POSITIVE_RATING_VAL
            else:
                self.R[query_id, page_id] = 0

            #return approx_rating
            return approx_rating
        
    def remove_random_ratings(self, num_to_each_query=5):
        """
        Adds N random ratings to every user in self.R.
        :param num_to_each_user: Number of random (positive=1 or negative=-1)ratings to be added to each user.
        :return: self.R with added ratings.
        """
        no_aricles = self.R.shape[1]
        no_queries = self.R.shape[0]
        
        
        
        ids = np.random.randint(no_queries, size=num_to_each_query)
        
        for i in ids:
            for j in range(no_aricles):
                if (self.R[i,j]==1):
                    self.R[i,j]=0
                    break
        
        return self.R        
    
    
    def calculate_auc(self):
        A = self.A
        b = self.b        
        
        query_ids = range(self.num_queries)
        page_ids = range(self.num_articles)
        
        allscores=np.zeros((self.num_queries,self.num_articles))
        
        for j in page_ids:
            for i in query_ids:
                # get the arm features given page id, and a query embeddings
                # though here I all queries are insample, I still use this out-of-sample arm function so it 
                # can be applied on held-out query data sets for validation in the future
                arm_features = self.get_featuers_of_new_arms_oos(j, new_query_embeddings=self.query_embeddings[i])
                x_ta = arm_features.reshape(-1, 1)  # make a column vector
                
                A_a_inv = np.linalg.inv(A[j])
                theta_a = A_a_inv.dot(b[j])
                score_a = theta_a.T.dot(x_ta)
                allscores[i,j]=score_a
            
        
        truelabels = copy.deepcopy(self.oldR.T)  # transpose the truelabels as predicting the most relevant page to a query
        
        truelabels[truelabels<-0.5]=0   # change -1 to 0, so it only contains 0,1 
#         print(allscores)
#         print(truelabels)
        
#         try:
        s = metrics.roc_auc_score(truelabels, allscores.T, average='micro')
#         except Exception as inst:
#             s = 99
#             pass
        
        return s
            