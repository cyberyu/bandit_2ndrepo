from sklearn import metrics 
import tensorflow as tf
import numpy as np
import pickle
import time
import os

repeat = 400
use_gpu = False
num_articles=161
num_arm_features=1500
alpha=0.1
query_feature_num=500
page_feature_num=500

# i = tf.constant(0)
# repeat = 600sss
# use_gpu = False
# num_articles=321
# num_arm_features=305
# alpha=0.1
# query_feature_num=300
# page_feature_num=5


def body1(A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, epoch_id, rewards):
    global X_page, X_query, X_ratings, X_nystrom_text, X_nystrom_title, X_nystrom_keywords, X_nystrom_description, query_features, page_features, test_query_features, test_ratings
    
    def get_arm_features(qt_id):
        #reconstruct the id list of all arms indicies associating with qt_id
        qt_id_to_arm_ids = [[armloop*num_articles+qt_id] for armloop in range(num_articles)] 
        return [tf.expand_dims(tf.gather_nd(X_nystrom_text, tf.convert_to_tensor(qt_id_to_arm_ids)), -1),
                tf.expand_dims(tf.gather_nd(X_nystrom_title, tf.convert_to_tensor(qt_id_to_arm_ids)), -1),
                tf.expand_dims(tf.gather_nd(X_nystrom_keywords, tf.convert_to_tensor(qt_id_to_arm_ids)), -1),
                tf.expand_dims(tf.gather_nd(X_nystrom_description, tf.convert_to_tensor(qt_id_to_arm_ids)), -1)]
    
    def make_prediction(qt_id,A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, test_score):
        global test_query_features, X_page, X_nystrom_text, X_nystrom_title, X_nystrom_keywords, X_nystrom_description

        [x_te_nystroem_text, x_te_nystroem_title, x_te_nystroem_keywords, x_te_nystroem_description] = get_arm_features(qt_id)
        # calculate theta, theta needs to be updated by loop
        theta_text = tf.linalg.solve(A_text, b_text)
        theta_title = tf.linalg.solve(A_title, b_title)
        theta_description = tf.linalg.solve(A_description, b_description)
        theta_keywords = tf.linalg.solve(A_keywords, b_keywords)

        predict_score_text = tf.reshape(tf.matmul(tf.transpose(theta_text,[0,2,1]), x_te_nystroem_text), [num_articles])
        predict_score_title = tf.reshape(tf.matmul(tf.transpose(theta_title,[0,2,1]), x_te_nystroem_title), [num_articles])
        predict_score_keywords = tf.reshape(tf.matmul(tf.transpose(theta_keywords,[0,2,1]), x_te_nystroem_keywords), [num_articles])
        predict_score_description = tf.reshape(tf.matmul(tf.transpose(theta_description,[0,2,1]), x_te_nystroem_description), [num_articles])
        
        
        predict_score = tf.math.reduce_max(
                tf.stack([predict_score_text, predict_score_title, predict_score_keywords, predict_score_description], axis=1), axis=1) 
        
        #print(predict_score)
        test_score[:,qt_id].assign(predict_score)
        qt_id=tf.add(qt_id, 1)

        return qt_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, test_score      

    def body2(q_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, rewards, epoch_id):
        global X_page, X_query, X_ratings, X_nystrom_text, X_nystrom_title, X_nystrom_keywords, X_nystrom_description, query_features, page_features
        
        fixed_rewards = True
        def save_variable(contents, filename):
        #     tf.io.write_file(
        #         filename, contents, name=None
        #     )
            np.savetxt(filename, contents, delimiter=",", fmt='%1.2f')
            return None
        
        def get_arm_features_v2(qt_id, a_t):
            #t_ind = tf.math.add(tf.constant(a_t[0][0]*num_articles,dtype=tf.int64),tf.constant(qt_id,dtype=tf.int64))
            return [tf.expand_dims(tf.gather_nd(X_nystrom_text, tf.convert_to_tensor([[a_t[0][0].numpy()*num_articles+qt_id.numpy()]])), -1),
                    tf.expand_dims(tf.gather_nd(X_nystrom_title, tf.convert_to_tensor([[a_t[0][0].numpy()*num_articles+qt_id.numpy()]])), -1),
                    tf.expand_dims(tf.gather_nd(X_nystrom_keywords, tf.convert_to_tensor([[a_t[0][0].numpy()*num_articles+qt_id.numpy()]])), -1),
                    tf.expand_dims(tf.gather_nd(X_nystrom_description, tf.convert_to_tensor([[a_t[0][0].numpy()*num_articles+qt_id.numpy()]])), -1)]
        
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
                    tp1 = tf.reshape(tf.gather_nd(X_query,query_pos_rat_idxs),[query_feature_num,-1])
                    tp2 = tf.reshape(current_query_features,[query_feature_num,1])
                    match_likabilities = compute_cosine_similarity(tp2, tp1)
                    match_likabilities_without_nans = tf.where(tf.math.is_nan(match_likabilities), tf.zeros_like(match_likabilities), match_likabilities)
                    result_match_likability = tf.math.reduce_mean(match_likabilities_without_nans, axis=-1)
                    tfzeros = tf.constant(0.0)
                    binomial_reward_probability = tf.cond(pred_smaller(result_match_likability,tfzeros), lambda: tf.constant(0.0), lambda: result_match_likability) 
                    binomial_reward_probability = tf.cond(pred_smaller(binomial_reward_probability,tf.constant(1.0)), lambda: binomial_reward_probability, lambda: tf.constant(1.0)) 
                    return np.random.binomial(n=1, p=binomial_reward_probability)  

                
                # code of if_false ###############
                ##################################
                
                #current_page_feature=X_page[page_id]
                #current_page_feature=tf.gather_nd(X_page, [tf.constant(page_id)])
                
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
            current_ratings =tf.gather_nd(X_ratings, [tf.cast(query_id, tf.int64), page_id[0][0]]) 
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
        
#         q = tf.tile(
#             tf.transpose(tf.reshape(query_features[q_id,:],[query_feature_num,1])), [num_articles,1], name='q'
#         )
#         q = tf.cast(q, dtype=tf.float32, name='q') 

#         # combine query features with all page features
#         arm_f = tf.transpose(tf.transpose(tf.concat([q,X_page],1)))

#         x_t = tf.expand_dims(arm_f, -1) # add a new dimension
        
        
        
        ##############################################################
        ##############################################################
        ###############################################################
        # replace with Nystroem arm features
        [x_t_text, x_t_title, x_t_description, x_t_keywords]  = get_arm_features(q_id)

        # calculate theta, theta needs to be updated by loop
        theta_text = tf.linalg.solve(A_text, b_text)
        g_t_a_text = tf.linalg.solve(A_text, x_t_text)
        p_t_a_text = tf.add(tf.matmul(tf.transpose(theta_text,[0,2,1]), x_t_text),alpha*tf.sqrt(tf.matmul(tf.transpose(x_t_text,[0,2,1]),g_t_a_text)))

        theta_title = tf.linalg.solve(A_title, b_title)
        g_t_a_title = tf.linalg.solve(A_title, x_t_title)
        p_t_a_title = tf.add(tf.matmul(tf.transpose(theta_title,[0,2,1]), x_t_title),alpha*tf.sqrt(tf.matmul(tf.transpose(x_t_title,[0,2,1]),g_t_a_title)))

        theta_description = tf.linalg.solve(A_description, b_description)
        g_t_a_description = tf.linalg.solve(A_description, x_t_description)
        p_t_a_description = tf.add(tf.matmul(tf.transpose(theta_description,[0,2,1]), x_t_description),alpha*tf.sqrt(tf.matmul(tf.transpose(x_t_description,[0,2,1]),g_t_a_description)))

        theta_keywords = tf.linalg.solve(A_keywords, b_keywords)
        g_t_a_keywords = tf.linalg.solve(A_keywords, x_t_keywords)
        p_t_a_keywords = tf.add(tf.matmul(tf.transpose(theta_keywords,[0,2,1]), x_t_keywords),alpha*tf.sqrt(tf.matmul(tf.transpose(x_t_keywords,[0,2,1]),g_t_a_keywords)))        
        #save_variable(tf.reshape(p_t_a,[321,1]).numpy(), './tmp/p_t_a_gpu_query_'+str(q_id.numpy())+'.txt')
        # move to next query
        add = tf.add(q_id, 1)
        
        #
        p_t_all = tf.concat([p_t_a_text, p_t_a_title, p_t_a_description, p_t_a_keywords], -1)
        # pick the maxone 
        
        v_t = tf.argmax(tf.math.reduce_max(p_t_all, axis=0), axis=1)
        h_t= tf.argmax(p_t_all, axis=0)
        #print(v_t)        
        a_t = tf.expand_dims(tf.expand_dims(tf.gather_nd(h_t,[0, v_t[0].numpy()]),-1),-1)
        
        #print(a_t)

        r_t = recommend(query_id=q_id, page_id=a_t, ratings=ratings)

        rewards = tf.add(rewards, r_t)

        #current_page_feature = tf.reshape(tf.gather_nd(page_features, a_t),[page_feature_num,1])
        #print(current_page_feature)
        
        
        ##############################################################
        ##############################################################
        ###############################################################
        # replace with Nystroem arm features        
        [x_t_at_nystroem_text, x_t_at_nystroem_title,x_t_at_nystroem_description,x_t_at_nystroem_keywords]=get_arm_features_v2(q_id, a_t)
        
        #x_t_at = tf.concat([tf.reshape(query_features[q_id,:],[query_feature_num,1]),current_page_feature],0)

        #A_a_t = A[a_t,:]
        A_a_t_text = tf.gather_nd(A_text, a_t)
        A_a_t_new_text = tf.add(tf.cast(A_a_t_text, float), tf.cast(tf.matmul(tf.reshape(x_t_at_nystroem_text,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at_nystroem_text,[num_arm_features,-1]))), float))
        b_a_t_text = tf.gather_nd(b_text, a_t)
        b_a_t_new_text = tf.add(b_a_t_text,r_t*tf.cast(x_t_at_nystroem_text,float))
        A_text=tf.tensor_scatter_nd_update(A_text, a_t, A_a_t_new_text)
        b_text=tf.tensor_scatter_nd_update(b_text, a_t, b_a_t_new_text)
        
        A_a_t_title = tf.gather_nd(A_title, a_t)
        A_a_t_new_title = tf.add(tf.cast(A_a_t_title, float), tf.cast(tf.matmul(tf.reshape(x_t_at_nystroem_title,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at_nystroem_title,[num_arm_features,-1]))), float))
        b_a_t_title = tf.gather_nd(b_title, a_t)
        b_a_t_new_title = tf.add(b_a_t_title,r_t*tf.cast(x_t_at_nystroem_title,float))
        A_title=tf.tensor_scatter_nd_update(A_title, a_t, A_a_t_new_title)
        b_title=tf.tensor_scatter_nd_update(b_title, a_t, b_a_t_new_title)  
        
        
        A_a_t_description = tf.gather_nd(A_description, a_t)
        A_a_t_new_description = tf.add(tf.cast(A_a_t_description, float), tf.cast(tf.matmul(tf.reshape(x_t_at_nystroem_description,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at_nystroem_description,[num_arm_features,-1]))), float))
        b_a_t_description = tf.gather_nd(b_description, a_t)
        b_a_t_new_description = tf.add(b_a_t_description,r_t*tf.cast(x_t_at_nystroem_description,float))
        A_description=tf.tensor_scatter_nd_update(A_description, a_t, A_a_t_new_description)
        b_description=tf.tensor_scatter_nd_update(b_description, a_t, b_a_t_new_description)          
      
        A_a_t_keywords = tf.gather_nd(A_keywords, a_t)
        A_a_t_new_keywords = tf.add(tf.cast(A_a_t_keywords, float), tf.cast(tf.matmul(tf.reshape(x_t_at_nystroem_keywords,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at_nystroem_keywords,[num_arm_features,-1]))), float))
        b_a_t_keywords = tf.gather_nd(b_keywords, a_t)
        b_a_t_new_keywords = tf.add(b_a_t_keywords,r_t*tf.cast(x_t_at_nystroem_keywords,float))
        A_keywords=tf.tensor_scatter_nd_update(A_keywords, a_t, A_a_t_new_keywords)
        b_keywords=tf.tensor_scatter_nd_update(b_keywords, a_t, b_a_t_new_keywords)          

        return add, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, rewards, epoch_id
        
        
        # return [add, query_features, A, b, page_features, r_t, accr]     
    
    
    #START of BODY1###############################
    #############################################
    
    # reinitialize query id each epoch
    query_id = tf.constant(0)
    query_test_id = tf.constant(0)
    r_t = tf.constant(0.0)
    rewards = tf.constant(0.0)
    test_score = tf.Variable(tf.zeros([161,400]))  
    repeat=400
    
    
    #for epoch in range(2):
    query_loop_condition = lambda query_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, rewards, epoch_id: tf.less(query_id, tf.constant(repeat))
    query_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, rewards, epoch_id = tf.while_loop(query_loop_condition, body2, [query_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, rewards, epoch_id])      
    
    query_loop_condition_test = lambda query_test_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, test_score: tf.less(query_test_id, tf.constant(repeat))
    query_test_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, test_score = tf.while_loop(query_loop_condition_test, make_prediction, [query_test_id, A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, test_score])
    
    
    auc_score = metrics.roc_auc_score(test_ratings, test_score.numpy().T, average='micro')
    
    print('Epoch {} trained with rewards {} and out-of-sample auc {}'.format(epoch_id, rewards.numpy(), auc_score))
    
    epoch_id = tf.add(epoch_id, 1)

    return A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, epoch_id, rewards 
    

    
#First structure: type=list str=[TensorSpec(shape=(), dtype=tf.int32, name=None), TensorSpec(shape=(321, 325, 325), dtype=tf.float32, name=None), TensorSpec(shape=(321, 325, 1), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.float32, name=None)]    


def train(epochs):
    
    global X_page, X_query, X_ratings, X_nystrom_text, X_nystrom_title, X_nystrom_keywords, X_nystrom_description, query_features, page_features, ratings, test_query_features, test_ratings
    
    # initialize the data
    
#     question_id = pickle.load(open('data/sample_by_question_questions_dict.pkl','rb'))
#     question_intent_features = pickle.load(open('data/sample_by_question_query_intents.pkl','rb'))
#     page_features=pickle.load(open('data/infowave_tfidf_bow.pickle','rb')).todense()
#     intent_features = np.zeros(shape=(repeat,query_feature_num), dtype=float)
#     for k,v in question_id.items():
#         intent_features[v,:]=question_intent_features[v]   
#     query_features=intent_features

    query_features = pickle.load(open('data/X_query_tfidf_400.pkl','rb'))
    page_features=pickle.load(open('data/X_page_text_tfidf_400.pkl','rb'))
    
    test_query_features= pickle.load(open('data/X_query_tfidf_400.pkl','rb'))

    ratings = pickle.load(open('data/X_ass_mat_400.pkl','rb'))
    test_ratings = pickle.load(open('data/X_ass_mat_400.pkl','rb'))
    
    K_pc_500_text = pickle.load(open('data/K_pc_1500_text_sub.pkl','rb'))
    K_pc_500_title = pickle.load(open('data/K_pc_1500_title_sub.pkl','rb'))
    K_pc_500_description = pickle.load(open('data/K_pc_1500_description_sub.pkl','rb'))
    K_pc_500_keywords = pickle.load(open('data/K_pc_1500_keywords_sub.pkl','rb'))
        
#     ratings=np.zeros((repeat,num_articles)) 

#     for i,j in sparse_ratings:
#         ratings[i,j] = 1

    ident = tf.eye(num_arm_features)

    X_page = tf.constant(page_features, dtype=tf.float32, name="X_page")
    X_query = tf.constant(query_features, dtype=tf.float32, name="X_query")
    X_ratings = tf.constant(ratings, dtype=tf.int32, name="X_ratings")
    
    X_nystrom_text = tf.constant(K_pc_500_text, dtype=tf.float32, name="X_nystrom_text")
    X_nystrom_title = tf.constant(K_pc_500_title, dtype=tf.float32, name="X_nystrom_title")
    X_nystrom_description = tf.constant(K_pc_500_description, dtype=tf.float32, name="X_nystrom_description")
    X_nystrom_keywords = tf.constant(K_pc_500_keywords, dtype=tf.float32, name="X_nystrom_keywords")

    A_text = tf.Variable(tf.ones([num_articles, 1, 1], name='A_text') * ident)
    A_title = tf.Variable(tf.ones([num_articles, 1, 1], name='A_title') * ident)    
    A_description = tf.Variable(tf.ones([num_articles, 1, 1], name='A_description') * ident)
    A_keywords = tf.Variable(tf.ones([num_articles, 1, 1], name='A_keywords') * ident)    

    #E = tf.Variable(tf.ones([num_articles, 1, 1], name='E') * ident)
    bv = tf.Variable(tf.zeros([num_arm_features, 1], name='bv'))
    b_text = tf.Variable(tf.ones([num_articles, 1, 1], name='b_text') * bv)
    b_title = tf.Variable(tf.ones([num_articles, 1, 1], name='b_title') * bv)
    b_description = tf.Variable(tf.ones([num_articles, 1, 1], name='b_description') * bv)
    b_keywords = tf.Variable(tf.ones([num_articles, 1, 1], name='b_keywords') * bv)
    
    epoch_id = tf.constant(0)
    rewards = tf.constant(0, dtype=tf.float32)

    #result = tf.constant(0.0)
    
    epoch_loop_condition = lambda A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, epoch_id, rewards : tf.less(epoch_id, epochs)
    A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, epoch_id, rewards = tf.while_loop(epoch_loop_condition, body1, [A_text, A_title, A_description, A_keywords, b_text, b_title, b_description, b_keywords, epoch_id, rewards])
    
    return epoch_id, rewards
        
t0 = time.time()    
eid, re = train(400)
t1 = time.time()
print("total timing is {}", str(t1-t0))
