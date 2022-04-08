# TensorLinUCB algorithm
# Developed by Shi Yu  shi_yu@vanguard.com
# Oct 2020

from sklearn import metrics
import tensorflow as tf
import numpy as np
import pickle
import time
import os

repeat = 600
num_articles=321
num_arm_features=305
alpha=0.1
query_feature_num=300
page_feature_num=5

# i = tf.constant(0)
# repeat = 600
# use_gpu = False
# num_articles=321
# num_arm_features=305
# alpha=0.1
# query_feature_num=300
# page_feature_num=5


def body1(A, b, epoch_id, rewards):
    global X_page, X_query, X_ratings, query_features, page_features, test_query_features, test_ratings
    
    def make_prediction(qt_id, A, b, test_score):
        global test_query_features, X_page

        qt = tf.tile(
            tf.transpose(tf.reshape(test_query_features[qt_id,:],[query_feature_num,1])), [num_articles,1], name='qt'
        )
        qt = tf.cast(qt, dtype=tf.float32, name='qt') 

        # combine query features with all page features
        arm_f_t = tf.transpose(tf.transpose(tf.concat([qt,X_page],1)))

        x_te = tf.expand_dims(arm_f_t, -1) # add a new dimension

        # calculate theta, theta needs to be updated by loop
        theta = tf.linalg.solve(A, b)

        predict_score = tf.reshape(tf.matmul(tf.transpose(theta,[0,2,1]), x_te), [num_articles])

        test_score[:,qt_id].assign(predict_score)
        qt_id=tf.add(qt_id, 1)

        return qt_id, A, b, test_score      

    def body2(q_id, A, b, rewards, epoch_id):
        global X_page, X_query, X_ratings, query_features, page_features
        
        fixed_rewards = True
        def save_variable(contents, filename):
        #     tf.io.write_file(
        #         filename, contents, name=None
        #     )
            np.savetxt(filename, contents, delimiter=",", fmt='%1.2f')
            return None
        
        def recommend(query_id, page_id, ratings):

            @tf.function
            def pred(a,b):
                if tf.math.equal(a,b):
                    return True
                else:
                    return False

            @tf.function
            def pred_smaller(a,b):
                if (tf.math.equal(a,b) or tf.math.less(a,b)):
                    return True
                else:
                    return False        

            @tf.function
            def pred_larger(a,b):
                if tf.math.greater(a,b):
                    return True
                else:
                    return False      
                
            def if_true(fixed_rewards=True):
                if fixed_rewards:
                    return 1
                else:
                    #return tfd.Binomial(total_count=1, logits=0.9)
                    return np.random.binomial(n=1, p=0.9)
                #return 0.2

            def if_false(): 

                def compute_cosine_distances(a, b):
                    # x shape is n_a * dim
                    # y shape is n_b * dim
                    # results shape is n_a * n_b

                    normalize_a = tf.nn.l2_normalize(a,1)        
                    normalize_b = tf.nn.l2_normalize(b,0)
                    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    return distance

                def compute_cosine_similarity(a, b):
                    # x shape is n_a * dim
                    # y shape is n_b * dim
                    # results shape is n_a * n_b
                    #print('a ', a)
                    normalize_a = tf.math.l2_normalize(a,0)        
                    #print('normalize_a ', normalize_a)

                    #print('b ', b)
                    normalize_b = tf.math.l2_normalize(b,0)
                    #print('normalize_b ', normalize_b)

                    similarity = tf.tensordot(normalize_a, normalize_b, (0,0))
                    #print('returning ', similarity)
                    return similarity

                def return_minprob():
                    return 0.0
                    #return tf.constant(0.0)

                def return_normalprob():

                    # get a list of query features that has "clicked" ratings to a page
                    # query_pos_rat_idxs can be a list of indices
                    # 
                    #tp1 = tf.reshape(tf.gather_nd(X_ratings,pos_indices),[X_query.shape[0],-1])
                    tp1 = tf.reshape(tf.gather_nd(X_query,query_pos_rat_idxs),[300,-1])
                    tp2 = tf.reshape(current_query_features,[300,1])

                    match_likabilities = compute_cosine_similarity(tp2, tp1)
                    match_likabilities_without_nans = tf.where(tf.math.is_nan(match_likabilities), tf.zeros_like(match_likabilities), match_likabilities)
                    #sum_ignoring_nans = tf.reduce_sum(tensor_without_nans, axis=-1)

                    #print('match_likabilities is ', match_likabilities)
                    result_match_likability = tf.math.reduce_mean(match_likabilities_without_nans, axis=-1)
                    #print('match_likability is ', result_match_likability)
                    tfzeros = tf.constant(0.0)
        #            print(result_match_likability)
                    binomial_reward_probability = tf.cond(pred_smaller(result_match_likability,tfzeros), lambda: tf.constant(0.0), lambda: result_match_likability) 
                    binomial_reward_probability = tf.cond(pred_smaller(binomial_reward_probability,tf.constant(1.0)), lambda: binomial_reward_probability, lambda: tf.constant(1.0)) 
                    return np.random.binomial(n=1, p=binomial_reward_probability)  

                
                # code of if_false ###############
                ##################################
                
                #current_page_feature=X_page[page_id]
                current_page_feature=tf.gather_nd(X_page, [tf.constant(page_id)])
                current_query_features=tf.gather_nd(X_query,[tf.constant(query_id)])
                indices = [[x, page_id[0][0]] for x in range(X_ratings.shape[0])]
                #print('indices ', indices)
                query_ratings =tf.gather_nd(X_ratings, indices)
                query_pos_rat_idxs = tf.where(query_ratings)
                query_pos_rat_idxs_size = tf.reshape(tf.size(query_pos_rat_idxs),[])

        #             if binomial_reward_probability <= 0:
        #                 #print("User={}, item={}, genre likability={}".format(user_id, item_id, result_genre_likability))
        #                 binomial_reward_probability = MIN_PROBABILITY # this could be replaced by small probability
                # the following condition
                # if query_pos_rat_idxs_size<=0, return minimal probability, otherwise, call return_normal_prob function
                result = tf.cond(pred(query_pos_rat_idxs_size, tfzero), return_minprob, return_normalprob)

                return result       


                #return 0.3
            ## code of recommend #######################
            ##########################################
            
            MIN_PROBABILITY = tf.constant(0.0)
            POSITIVE_RATING_VAL = tf.constant(1)
            clicked = tf.constant(1.0)
            tfzero = tf.constant(0)
            #result = np.random.binomial(n=1, p=0.9) 
            #result = 0.2


        #     print(query_id)
        #     print(page_id)
            #tf.stack(tf.reshape(page_id,[]), query_id.numpy()
            current_ratings =tf.gather_nd(X_ratings, [page_id[0][0],tf.cast(query_id, tf.int64)]) 
        #    print(current_ratings)
            result = tf.cond(tf.math.equal(current_ratings,POSITIVE_RATING_VAL), if_true, if_false )
            #result = 0

        #     tp1 = tf.gather_nd(X_query,query_pos_rat_idxs)
        #     tp2 = tf.reshape(tf.nn.l2_normalize(current_query_features, 0),[300,1])
        #     tp3 = tf.reshape(tf.nn.l2_normalize(tp1, 0),[300,1])
        #     match_likabilities = tf.losses.cosine_distance(tp2, tp3, dim=0)
        #     result_match_likability = tf.math.reduce_mean(match_likabilities)
        #     binomial_reward_probability = tf.cond(result_match_likability < tf.to_float(tf.constant(0)), lambda: MIN_PROBABILITY, lambda: result_match_likability)     
            return result        

        ## code of body 2 #######################
        ##########################################        
        
        q = tf.tile(
            tf.transpose(tf.reshape(query_features[q_id,:],[query_feature_num,1])), [num_articles,1], name='q'
        )
        q = tf.cast(q, dtype=tf.float32, name='q') 

        # combine query features with all page features
        arm_f = tf.transpose(tf.transpose(tf.concat([q,X_page],1)))

        x_t = tf.expand_dims(arm_f, -1) # add a new dimension

        # calculate theta, theta needs to be updated by loop
        theta = tf.linalg.solve(A, b)
        #print('theta ',theta[0])

        g_t_a = tf.linalg.solve(A, x_t)
        #print('g_t_a',g_t_a)

        p_t_a = tf.add(tf.matmul(tf.transpose(theta,[0,2,1]), x_t),alpha*tf.sqrt(tf.matmul(tf.transpose(x_t,[0,2,1]),g_t_a)))

        #save_variable(tf.reshape(p_t_a,[321,1]).numpy(), './tmp/p_t_a_gpu_query_'+str(q_id.numpy())+'.txt')
        # move to next query
        add = tf.add(q_id, 1)

        # pick the maxone 
        a_t= tf.argmax(p_t_a, axis=0)

        

        r_t = recommend(query_id=q_id, page_id=a_t, ratings=ratings)

        rewards = tf.add(rewards, r_t)

        current_page_feature = tf.reshape(tf.gather_nd(page_features, a_t),[page_feature_num,1])
        #print(current_page_feature)
        x_t_at = tf.concat([tf.reshape(query_features[q_id,:],[query_feature_num,1]),current_page_feature],0)

        #A_a_t = A[a_t,:]
        A_a_t = tf.gather_nd(A, a_t)
        A_a_t_new = tf.add(tf.cast(A_a_t, float), tf.cast(tf.matmul(tf.reshape(x_t_at,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at,[num_arm_features,-1]))), float))

        #print("b_a_t")
        #b_a_t = b[a_t]
        b_a_t = tf.gather_nd(b, a_t)
        #print(b_a_t)

        b_a_t_new = tf.add(b_a_t,r_t*tf.cast(x_t_at,float))
        
        #print(a_t)
        
        A=tf.tensor_scatter_nd_update(A, a_t, A_a_t_new)
        b=tf.tensor_scatter_nd_update(b, a_t, b_a_t_new)
        
    
#         tf.compat.v1.scatter_update(A,[a_t],A_a_t_new)
#         tf.compat.v1.scatter_update(b,[a_t],b_a_t_new)        

        return add, A, b, rewards, epoch_id
        

        
        
        
        # return [add, query_features, A, b, page_features, r_t, accr]     
    
    
    #START of BODY1###############################
    #############################################
    
    # reinitialize query id each epoch
    query_id = tf.constant(0)
    query_test_id = tf.constant(0)
    r_t = tf.constant(0.0)
    rewards = tf.constant(0.0)
    test_score = tf.Variable(tf.zeros([321,200]))    
    repeat = 600
    
    
    #for epoch in range(2):
    query_loop_condition = lambda query_id, A, b, rewards, epoch_id: tf.less(query_id, repeat)
    query_id, A, b, rewards, epoch_id = tf.while_loop(query_loop_condition, body2, [query_id, A, b, rewards, epoch_id])      
    
    query_loop_condition_test = lambda query_test_id, A, b, test_score: tf.less(query_test_id, tf.constant(200))
    query_test_id, A, b, test_score = tf.while_loop(query_loop_condition_test, make_prediction, [query_test_id, A, b, test_score])
    
    
    auc_score = metrics.roc_auc_score(test_ratings, test_score.numpy().T, average='micro')
    
    print('Epoch {} trained with rewards {} and out-of-sample auc {}'.format(epoch_id, rewards.numpy(), auc_score))
    
    epoch_id = tf.add(epoch_id, 1)

    return A, b, epoch_id, rewards 
    
# main training program

def train(epochs):
    
    global X_page, X_query, X_ratings, query_features, page_features, ratings, test_query_features, test_ratings
    
    # initialize the data
    
#     question_id = pickle.load(open('data/sample_by_question_questions_dict.pkl','rb'))
#     question_intent_features = pickle.load(open('data/sample_by_question_query_intents.pkl','rb'))
#     page_features=pickle.load(open('data/infowave_tfidf_bow.pickle','rb')).todense()
#     intent_features = np.zeros(shape=(repeat,query_feature_num), dtype=float)
#     for k,v in question_id.items():
#         intent_features[v,:]=question_intent_features[v]   
#     query_features=intent_features

    query_features = pickle.load(open('data/X_train_bow_features.pkl','rb'))
    page_features=pickle.load(open('data/X_all_page_pca_features.pkl','rb'))
    
    test_query_features= pickle.load(open('data/X_test_bow_features.pkl','rb'))

    ratings = pickle.load(open('data/Y_train_labels.pkl','rb'))
    test_ratings = pickle.load(open('data/Y_test_labels.pkl','rb'))
    
#     ratings=np.zeros((repeat,num_articles)) 

#     for i,j in sparse_ratings:
#         ratings[i,j] = 1

    ident = tf.eye(num_arm_features)

    X_page = tf.constant(page_features, dtype=tf.float32, name="X_page")
    X_query = tf.constant(query_features, dtype=tf.float32, name="X_query")
    X_ratings = tf.constant(ratings, dtype=tf.int32, name="X_ratings")

    A = tf.Variable(tf.ones([num_articles, 1, 1], name='A') * ident)
    E = tf.Variable(tf.ones([num_articles, 1, 1], name='E') * ident)
    bv = tf.Variable(tf.zeros([num_arm_features, 1], name='bv'))
    b = tf.Variable(tf.ones([num_articles, 1, 1], name='b') * bv)
    

    
    epoch_id = tf.constant(0)
    rewards = tf.constant(0, dtype=tf.float32)

    #result = tf.constant(0.0)
    
    epoch_loop_condition = lambda  A, b, epoch_id, rewards : tf.less(epoch_id, epochs)
    A, b, epoch_id, rewards = tf.while_loop(epoch_loop_condition, body1, [A, b, epoch_id, rewards])
    

    
    return epoch_id, rewards

        
t0 = time.time()    
eid, re = train(400)
t1 = time.time()
print("total timing is {}", str(t1-t0))