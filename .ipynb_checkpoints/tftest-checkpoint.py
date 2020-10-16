import tensorflow as tf
import numpy as np
import pickle
import time
import os

repeat = 299
use_gpu = False
num_articles=321
num_arm_features=325
alpha=0.1
query_feature_num=25
page_feature_num=300

# i = tf.constant(0)
# repeat = 600
# use_gpu = False
# num_articles=321
# num_arm_features=305
# alpha=0.1
# query_feature_num=300
# page_feature_num=5

def recommend(query_id, page_id, ratings):
    MIN_PROBABILITY = tf.constant(0.0)
    POSITIVE_RATING_VAL = tf.constant(1)
    clicked = tf.constant(1.0)
    tfzero = tf.constant(0)
    #result = np.random.binomial(n=1, p=0.9) 
    result = 0.2
    return result


def save_variable(contents, filename):
#     tf.io.write_file(
#         filename, contents, name=None
#     )
    np.savetxt(filename, contents, delimiter=",", fmt='%1.2f')
    return None

def body2(q_id, query_features, A, b, page_features, r_t):
    
    #print(tf.reshape(query_features[q_id,:],[300,1]))
    # tile the given query features
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
    
    #print('p_t_a', p_t_a)
    
    
#         save_variable(arm_f.numpy(), './tmp/arm_f_query_'+str(q_id.numpy())+'.txt')
#         save_variable(theta[0].numpy(), './tmp/theta_gpu_query_'+str(q_id.numpy())+'.txt')
#         save_variable(g_t_a[0].numpy(), './tmp/g_t_a_gpu_query_'+str(q_id.numpy())+'.txt')
    save_variable(tf.reshape(p_t_a,[321,1]).numpy(), './tmp/p_t_a_gpu_query_'+str(q_id.numpy())+'.txt')
    
    # move to next query
    add = tf.add(q_id, 1)
    
    # pick the maxone 
    a_t= tf.argmax(p_t_a, axis=0)
    
    print('selecting a_t {} at {}',a_t.numpy(), tf.math.reduce_max(p_t_a).numpy())
    
    r_t_new = recommend(query_id=q_id, page_id=a_t, ratings=ratings)
    
    r_t = tf.add(r_t, r_t_new)

    #print(r_t)
#     print(query_features[q_id,:])
    
    current_page_feature = tf.reshape(tf.gather_nd(page_features, a_t),[page_feature_num,1])
    #print(current_page_feature)
    x_t_at = tf.concat([tf.reshape(query_features[q_id,:],[query_feature_num,1]),current_page_feature],0)

    #A_a_t = A[a_t,:]
    A_a_t = tf.gather_nd(A, a_t)
    A_a_t_new = tf.add(tf.compat.v1.to_float(A_a_t), tf.compat.v1.to_float(tf.matmul(tf.reshape(x_t_at,[num_arm_features,-1]), tf.transpose(tf.reshape(x_t_at,[num_arm_features,-1])))))
    
    #print("b_a_t")
    #b_a_t = b[a_t]
    b_a_t = tf.gather_nd(b, a_t)
    #print(b_a_t)
    
    b_a_t_new = tf.add(b_a_t,r_t*tf.compat.v1.to_float(x_t_at))
    
    print(b_a_t_new)
    #for p_id in range(321):
    
    if q_id in [0,1,2,3,4]:
        save_variable(tf.reshape(b_a_t_new,[325,1]), './tmp/b_a_t_gpu_query_'+str(q_id.numpy())+'_page_'+str(a_t.numpy())+'.txt')
    #print("b_a_t_new")
    #print(b_a_t_new)

#     print(A)
#     print(A_a_t_new)
#     print(tf.expand_dims(A_a_t_new,0))
    
#     print(b)
#     print(tf.expand_dims(b_a_t_new,0))
#     delta_A_a_t_value = new_value - v[index:index+1]
#     delta = tf.SparseTensor([[index]], delta_value, (5,))

    rand_t = tf.math.scalar_mul(tf.compat.v1.to_float(q_id), tf.ones([num_arm_features,1]), name='rand_t')
    ind_part_1 = tf.range(a_t[0][0])
    #print(ind_part_1)
    ind_part_2 = tf.range(tf.add(a_t[0][0],1),num_arm_features)
    
#     A_part_1 = tf.gather_nd(A,ind_part_1)
#     A_part_2 = tf.gather_nd(A,ind_part_2)
    #print('a_t',a_t)
    #print(A[:a_t])
    #print(A_a_t)
#     A = tf.concat([A[:a_t], tf.expand_dims(A_a_t_new,0), A[a_t+1:]], axis = 0 )
#     b = tf.concat([b[:a_t], tf.expand_dims(rand_t,0), b[a_t+1:]], axis = 0 )

#     A = tf.concat([A[:a_t], tf.expand_dims(A_a_t_new,0), A[a_t+1:]], axis = 0 )
#     b = tf.concat([b[:a_t], tf.expand_dims(rand_t,0), b[a_t+1:]], axis = 0 )
    
    
    tf.compat.v1.scatter_update(A,[a_t],A_a_t_new)
    tf.compat.v1.scatter_update(b,[a_t],b_a_t_new)

    #print(b)
#     print(a_t)
#     print(A_a_t_new)
    
#     tf.compat.v1.scatter_nd_update(A,[a_t],A_a_t_new)
#     tf.compat.v1.scatter_nd_update(b,[a_t],b_a_t_new)
    return [add, query_features, A, b, page_features, r_t] 


# query_features=pickle.load(open('data/X_train_bow_features.pkl','rb'))
# page_features=pickle.load(open('data/X_all_page_pca_features.pkl','rb'))
# ratings = pickle.load(open('data/Y_train_labels.pkl','rb'))
# ident = tf.eye(num_arm_features)

# X_page = tf.constant(page_features, dtype=tf.float32, name="X_page")
# X_query = tf.constant(query_features, dtype=tf.float32, name="X_query")
# X_ratings = tf.constant(ratings, dtype=tf.int32, name="X_ratings")

# A = tf.Variable(tf.ones([num_articles, 1, 1], name='A') * ident)
# E = tf.Variable(tf.ones([num_articles, 1, 1], name='E') * ident)
# bv = tf.Variable(tf.zeros([num_arm_features, 1], name='bv'))
# b = tf.Variable(tf.ones([num_articles, 1, 1], name='b') * bv)

# r_t = tf.constant(0.0)
# t0 = time.time()
# while_condition =   lambda i, _1, _2, b, _3, r_t: tf.less(i, repeat)
# loop, _1, A, b, _3, r_t= tf.while_loop(while_condition, body2, [i, query_features, A, b, page_features, r_t])   
# t1 = time.time()


# query_features=pickle.load(open('data/X_train_bow_features.pkl','rb'))
# page_features=pickle.load(open('data/X_all_page_pca_features.pkl','rb'))
# ratings = pickle.load(open('data/Y_train_labels.pkl','rb'))
# ident = tf.eye(num_arm_features)

# X_page = tf.constant(page_features, dtype=tf.float32, name="X_page")
# X_query = tf.constant(query_features, dtype=tf.float32, name="X_query")
# X_ratings = tf.constant(ratings, dtype=tf.int32, name="X_ratings")

# A = tf.Variable(tf.ones([num_articles, 1, 1], name='A') * ident)
# E = tf.Variable(tf.ones([num_articles, 1, 1], name='E') * ident)
# bv = tf.Variable(tf.zeros([num_arm_features, 1], name='bv'))
# b = tf.Variable(tf.ones([num_articles, 1, 1], name='b') * bv)

# r_t = tf.constant(0.0)
# t0 = time.time()
# while_condition =   lambda i, _1, _2, b, _3, r_t: tf.less(i, repeat)
# loop, _1, A, b, _3, r_t= tf.while_loop(while_condition, body2, [i, query_features, A, b, page_features, r_t])   
# t1 = time.time()

question_id = pickle.load(open('data/sample_by_question_questions_dict.pkl','rb'))
#question_intent_features = pickle.load(open('data/sample_by_question_query_intents.pkl','rb'))
#page_features=pickle.load(open('data/infowave_tfidf_bow.pickle','rb')).todense()
# intent_features = np.zeros(shape=(repeat,query_feature_num), dtype=float)
# for k,v in question_id.items():
#     intent_features[v,:]=question_intent_features[v,:]   
# query_features=intent_features

query_features = pickle.load(open('data/random_by_question_query_intents.pkl','rb'))
page_features=pickle.load(open('data/random_tfidf_bow.pkl','rb'))

sparse_ratings = pickle.load(open('data/sample_by_question_questions_article_ratings.pkl','rb'))
ratings=np.zeros((repeat,num_articles)) 

for i,j in sparse_ratings:
    ratings[i,j] = 1

ident = tf.eye(num_arm_features)

X_page = tf.constant(page_features, dtype=tf.float32, name="X_page")
X_query = tf.constant(query_features, dtype=tf.float32, name="X_query")
X_ratings = tf.constant(ratings, dtype=tf.int32, name="X_ratings")

A = tf.Variable(tf.ones([num_articles, 1, 1], name='A') * ident)
E = tf.Variable(tf.ones([num_articles, 1, 1], name='E') * ident)
bv = tf.Variable(tf.zeros([num_arm_features, 1], name='bv'))
b = tf.Variable(tf.ones([num_articles, 1, 1], name='b') * bv)

r_t = tf.constant(0.0)
t0 = time.time()
i = tf.constant(0)

while_condition =   lambda i, _1, _2, b, _3, r_t: tf.less(i, repeat)
loop, _1, A, b, _3, r_t= tf.while_loop(while_condition, body2, [i, query_features, A, b, page_features, r_t])   
t1 = time.time()
print("total timing is {}", str(t1-t0))