from _commons import warn, error, create_dir_path
import numpy as np
import time
from movielens import MovieLens
import pickle



class LinUCB:
    def __init__(self, alpha, max_items=500, allow_selecting_known_arms=True, fixed_rewards=True,
                 prob_reward_p=0.9):
#         if dataset is None:
#             self.dataset = MovieLens(variant='ml-100k',
#                                      pos_rating_threshold=4,
#                                      data_augmentation_mode='binary_unknown')
#         else:
#             self.dataset = dataset
#         self.dataset.shrink(max_items)
#         self.dataset.add_random_ratings(num_to_each_user=3)
        self.alpha = alpha
        self.fixed_rewards = fixed_rewards
        self.prob_reward_p = prob_reward_p
        
        self.UNKNOWN_RATING_VAL = 0
        self.POSITIVE_RATING_VAL = 1
        
        dataarray = pickle.load(open('dataarray.pkl','rb'))
        self.R=dataarray
        
        self.num_users=25
        self.arm_feature_dim=768+41
        self.num_items=41
        self.users_with_unrated_items = np.array(range(self.num_users))
        self.monitored_user = np.random.choice(self.users_with_unrated_items)
        self.allow_selecting_known_arms = allow_selecting_known_arms
        self.d = self.arm_feature_dim
        self.b = np.zeros(shape=(self.num_items, self.d))

        # More efficient way to create array of identity matrices of length num_items
        print("\nInitializing matrix A of shape {} which will require {}MB of memory."
              .format((self.num_items, self.d, self.d), 8 * self.num_items * self.d * self.d / 1e6))
        self.A = np.repeat(np.identity(self.d, dtype=float)[np.newaxis, :, :], self.num_items, axis=0)
        print("\nLinUCB successfully initialized.")

    def choose_arm(self, t, unknown_item_ids, verbosity):
        """
        Choose an arm to pull = item to recommend to user t that he did not rate yet.
        :param t: User_id of user to recommend to.
        :param unknown_item_ids: Indexes of items that user t has not rated yet.
        :return: Received reward for selected item = 1/0 = user liked/disliked item.
        """
        A = self.A
        b = self.b
        arm_features = self.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        p_t -= 9999  # I never want to select the already rated items
        item_ids = unknown_item_ids

        if self.allow_selecting_known_arms:
            item_ids = range(self.num_items)
            p_t += 9999

        for a in item_ids:  # iterate over all arms
            x_ta = arm_features[a].reshape(arm_features[a].shape[0], 1)  # make a column vector
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + self.alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))

        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("User {} has max p_t={}, p_t={}".format(t, max_p_t, p_t))

        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        # So I will get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t

        # observed reward = 1/0
        r_t = self.recommend(user_id=t, item_id=a_t,
                                     fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)

        if verbosity >= 2:
            print("User {} choosing item {} with p_t={} reward {}".format(t, a_t, p_t[a_t], r_t))

        x_t_at = arm_features[a_t].reshape(arm_features[a_t].shape[0], 1)  # make a column vector
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()  # turn it back into an array because b[a_t] is an array

        return r_t

    def run_epoch(self, verbosity=2):
        """
        Call choose_arm() for each user in the dataset.
        :return: Average received reward.
        """
        rewards = []
        start_time = time.time()

        for i in range(self.num_users):
            start_time_i = time.time()
            #user_id = self.get_next_user()
            user_id = 1
            unknown_item_ids = self.get_uknown_items_of_user(user_id)

            if self.allow_selecting_known_arms == False:
                if user_id not in self.users_with_unrated_items:
                    continue

                if unknown_item_ids.size == 0:
                    print("User {} has no more unknown ratings, skipping him.".format(user_id))
                    self.users_with_unrated_items = self.users_with_unrated_items[
                        self.users_with_unrated_items != user_id]
                    continue

            rewards.append(self.choose_arm(user_id, unknown_item_ids, verbosity))
            time_i = time.time() - start_time_i
            if verbosity >= 2:
                print("Choosing arm for user {}/{} ended with reward {} in {}s".format(i, self.num_users,
                                                                                       rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        return avg_reward, total_time

    def run(self, num_epochs, verbosity=1):
        """
        Runs run_epoch() num_epoch times.
        :param num_epochs: Number of epochs = iterating over all users.
        :return: List of average rewards per epoch.
        """
        self.users_with_unrated_items = np.array(range(self.num_users))
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time = self.run_epoch(verbosity)

            if verbosity >= 1:
                print(
                    "Finished epoch {}/{} with avg reward {} in {}s".format(i, num_epochs, avg_rewards[i], total_time))
        return avg_rewards

    def get_uknown_items_of_user(self, user_id):
        user_ratings = self.R[user_id]  # vector
        unknown_item_ids = np.argwhere(user_ratings == self.UNKNOWN_RATING_VAL).flatten()
        return unknown_item_ids    
    
    
    def get_features_of_current_arms(self, t):
        """
        Concatenates item features with user features.
        :param t: Time step = index of user that is being recommended to.
        :return: Matrix of (#arms x #feature_dims) for user t.
        """

        t = t % self.num_users
        user_features = self.R[t]  # vector
        user_features = np.tile(user_features, (self.num_items, 1))  # matrix where each row is R[t]
        item_features = self.item_genres  # matrix
        # arm_feature_dims = item_features.shape[1] + user_features.shape[0]
        arm_features = np.concatenate((user_features, item_features), axis=1)
        return arm_features    
    
    def _get_item_info(self):
        genres = np.zeros(shape=(self.num_items, self.num_genres), dtype=float)
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