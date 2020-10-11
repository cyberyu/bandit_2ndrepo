import numpy as np
import pickle
import time
import tensorflow as tf
import math
tf.executing_eagerly()


num_articles=321
num_arm_features=305

alpha=0.1


query_features=pickle.load(open('data/X_train_bow_features.pkl','rb'))
page_features=pickle.load(open('data/X_all_page_pca_features.pkl','rb'))
ratings = pickle.load(open('data/Y_train_labels.pkl','rb'))

ident = tf.eye(num_arm_features)
A = tf.Variable(tf.ones([num_articles, 1, 1], name='A') * ident)
b = tf.Variable(tf.zeros([num_articles, num_arm_features], name='b'))

# product = tf.matmul(ident3d[0,:], ident3d[0,:])

# #initialize the variable
#init_op = tf.initialize_all_variables()

def initialize_arm_features(queryf,pagef):
    X_query = tf.constant(queryf, dtype=tf.float32, name="X_query")
    X_page = tf.constant(pagef, dtype=tf.float32, name="X_page")
    return X_query, X_page

def get_arm_features(i,j):
    return tf.concat([X_query[i],X_page[j]],0)

def recommend(query_id, page_id, ratings, fixed_rewards=True, prob_reward_p=0.9, ):
    """
    Returns reward and updates rating maatrix self.R.
    :param fixed_rewards: Whether to always return 1/0 rewards for already rated items.
    :param prob_reward_p: Probability of returning the correct reward for already rated item.
    :return: Reward = either 0 or 1.
    """
    MIN_PROBABILITY = 0 # Minimal probability to like an item - adds stochasticity
    
    print(query_id)
    print(page_id)
    
    if ratings[query_id, page_id] == 1:
        if fixed_rewards:
            return 1
        else:
            return np.random.binomial(n=1, p=prob_reward_p)  # Bernoulli coin toss
    else:

        # the goal is to update a missing ""
        current_page_features = page_features[page_id,:] #get the article features
        current_query_features = query_features[query_id,:]  #get the article features

        # find out for a page, what query is rated as relevant (which for new query should be none)
        query_ratings = ratings[:,page_id]  #get all ratings by article id, it is a column
        query_pos_rat_idxs = np.argwhere(query_ratings == 1).flatten() # get all other positive ratings of the same article
        num_known_ratings = len(query_pos_rat_idxs)  # length of all other positive ratings

        match_likabilities=[]

        for query_idx in query_pos_rat_idxs:
            match_likabilities.append(cosine_similarity(current_query_features.reshape(-1,1), query_features[query_idx].reshape(-1,1)))

        result_match_likability = np.average(match_likabilities)

        if math.isnan(result_match_likability):
            result_match_likability=0

        binomial_reward_probability = result_match_likability
        #print (binomial_reward_probability)
        if binomial_reward_probability <= 0:
            #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
            binomial_reward_probability = MIN_PROBABILITY # this could be replaced by small probability

        approx_rating = np.random.binomial(n=1, p=binomial_reward_probability)  # Bernoulli coin toss

        if approx_rating == 1:
            ratings[query_id, page_id] = 1
        else:
            ratings[query_id, page_id] = 0

        #return approx_rating
        return approx_rating
    

#run the graph

t0 = time.time()
#with tf.Session() as sess:
X_query, X_page = initialize_arm_features(query_features, page_features)

#p_t = tf.Variable(tf.zeros([321,]), dtype=tf.float32, name="p_t")    

#print the random values that we sample

#tf.global_variables_initializer().run()

for q_id in tf.range(600):
    axt = []
    for a in tf.range(321):
        x_ta = get_arm_features(q_id,a)
        A_new = A[a,:]
        b_new = b[a,:]
        theta_a = tf.squeeze(tf.linalg.solve(tf.eye(num_arm_features) + A_new, tf.expand_dims(b_new, axis=-1)),axis=-1)
        g_t_a = tf.squeeze(tf.linalg.solve(tf.eye(num_arm_features)+A_new, tf.expand_dims(x_ta, axis=-1)), axis=-1)
        #n_p = tf.add(tf.tensordot(theta_a, x_ta, (0,0)),tf.tensordot(alpha, tf.sqrt(tf.tensordot(x_ta, g_t_a, (0,0))),(0,0)))

        temp=tf.math.add(tf.tensordot(theta_a, x_ta,(0,0)),tf.sqrt(tf.tensordot(x_ta, g_t_a, (0,0))))
        axt.append(temp)
#         print(type(n_p))
#         print(sess.run(n_p))
       #(n_p).eval()
    p_t = tf.stack(axt)
    #print(sess.run(p_t))

    print(p_t)
    max_p_t=tf.argmax(p_t, axis=0)
    #print(max_p_t)

    # need to add tile breaking
    a_t = max_p_t

    r_t = recommend(query_id=q_id, page_id=a_t, ratings=ratings)

    x_t_at = get_arm_features(0,a_t) 
    A_a_t = A[a_t,:]
    A_a_t_new = tf.add(A_a_t, tf.tensordot(x_t_at, x_t_at,(0,0)))
    b_a_t_new = tf.add(b[a_t],r_t*x_t_at)
#     print(sess.run(A_a_t_new))
#     print(A_a_t_new)
    tf.compat.v1.scatter_nd_update(A,[a_t],A_a_t_new)
#     tf.compat.v1.scatter_nd_update()
#     A[a_t].assign(A_a_t_new).eval()
    tf.compat.v1.scatter_nd_update(b,[a_t],b_a_t_new)
    #b[a_t] = b[a_t] + r_t * x_t_at.flatten()  # turn it back into an array because b[a_t] is an array
        
t1 = time.time()

# assign approach timing is 649 s
# append approach timing is 24 s


print("total timing is {}", str(t1-t0))