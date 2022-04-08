from sklearn import metrics 
import tensorflow as tf
import torch
import torch.nn.functional as f
import numpy as np
import pickle
import time
import os


def get_arm_features(X_nystrom, qt_id):
#     #reconstruct the id list of all arms indicies associating with qt_id
#     # getting rows from X_nystrom (every 100), what is its structure?
#     qt_id_to_arm_ids = [[armloop*num_articles+qt_id] for armloop in range(num_articles)] ## why every 100?
#     return tf.expand_dims(tf.gather_nd(X_nystrom, tf.convert_to_tensor(qt_id_to_arm_ids)), -1) ## what is this for? shape = (100, 1536)

    ## pytorch
    loc = [armloop*num_articles+qt_id for armloop in range(num_articles)]
    return torch.unsqueeze(X_nystrom[loc], -1)

def get_arm_features_v2(X_nystrom, qt_id, a_t):
    loc = [a_t * num_articles + qt_id]
    return torch.reshape(X_nystrom[loc], (-1, 1))
    #     return torch.reshape(X_nystrom[loc], (1, -1, 1))

def make_prediction(qt_id, A, b, X_nystrom):
    global test_query_features, X_page
    
    x_te_nystroem = get_arm_features(X_nystrom, qt_id)
    #print(x_te_nystroem)  # original shape num_pages x arm_dimensions


    # in Nystroem, it will be all pages associated with the arms

    # extract all profiles related to query qt_id

    # calculate theta, theta needs to be updated by loop
    theta, _ = torch.solve(b, A)
    
    # predict_score = tf.reshape(tf.matmul(tf.transpose(theta,[0,2,1]), x_te_nystroem), [num_articles])
    predict_score = torch.reshape(torch.transpose(theta, 1,2) @ x_te_nystroem, (-1,))
    
    del theta
    torch.cuda.empty_cache()
    
    return predict_score, A, b

def get_pta(q_id, A, b, epoch_id, X_nystrom):
    global X_page, X_query, X_ratings, query_features, page_features
    
    fixed_rewards = True
    
    def save_variable(contents, filename):
        np.savetxt(filename, contents, delimiter=",", fmt='%1.2f')
        return None

     # calculate theta, theta needs to be updated by loop
    theta, _ = torch.solve(b, A)
    
    # replace with Nystroem arm features
    x_t = get_arm_features(X_nystrom, q_id)
    g_t_a, _ = torch.solve(x_t, A) # out of memory error at second epoch
    
    # shape 100,1,1
    p_t_a = torch.transpose(theta, 1, 2) @ x_t + alpha*torch.sqrt(torch.transpose(x_t, 1, 2) @ g_t_a) 
    
    del theta, g_t_a, x_t
    torch.cuda.empty_cache()
    
    return p_t_a

def compute_cosine_similarity(a, b):
    # x shape is n_a * dim
    # y shape is n_b * dim

    normalize_a = f.normalize(a, dim=0, p=2)
    normalize_b = f.normalize(b, dim=0, p=2)

    similarity = torch.tensordot(normalize_a, normalize_b, ([0], [0]))
    
    del normalize_a, normalize_b
    torch.cuda.empty_cache()

    return similarity

def recommend(qid, page_id, ratings, fixed_rewards=False):
    MIN_PROBABILITY = 0.0
    POSITIVE_RATING_VAL = 1
    clicked = 1.0
    tfzero = 0
    current_ratings = X_ratings[qid, page_id]

    if current_ratings == POSITIVE_RATING_VAL:
        if fixed_rewards:
            return 1
        else:
            return np.random.binomial(n=1, p=0.9)
    else:
        current_query_features = X_query[qid]
        query_ratings = X_ratings[:, page_id]
        query_pos_rat_idxs = torch.where(query_ratings)[0]
        query_pos_rat_idxs_size = query_pos_rat_idxs.shape

        if query_pos_rat_idxs_size[0] == tfzero:
            return 0.0
        else:
            tp1 = torch.reshape(X_query[query_pos_rat_idxs], (query_feature_num, -1))
            tp2 = torch.reshape(current_query_features, (query_feature_num,1))

            match_likabilities = compute_cosine_similarity(tp2, tp1) # shape 1, 49
            match_likabilities[match_likabilities != match_likabilities] = 0 # remove nans
            result_match_likability = torch.mean(match_likabilities, dim=-1)
            tfzeros = 0.0

            if result_match_likability <= tfzeros:
                binomial_reward_probability = tfzeros
            else:
                binomial_reward_probability = result_match_likability[0].cpu()

            if binomial_reward_probability > 1.0:
                binomial_reward_probability = 1.0
            try:
                del current_query_features, query_ratings, match_likabilities, result_match_likability
                torch.cuda.empty_cache()
                
                return float(np.random.binomial(n=1, p=binomial_reward_probability))
            except:
                import IPython; IPython.embed(); exit(1)

def update_param(A, b, q_id, a_t, r_t, X_nystrom):
    x_t_at_nystroem = get_arm_features_v2(X_nystrom, q_id, a_t)

    A_a_t = A[a_t]
    A_a_t_new = A_a_t + x_t_at_nystroem @ x_t_at_nystroem.transpose(0, 1)

    b_a_t = b[a_t]
    b_a_t_new = b_a_t + r_t * x_t_at_nystroem

    A[a_t] = A_a_t_new 
    b[a_t] = b_a_t_new

    return A, b

def train(epochs, log=False):
    
    global X_page, X_query, X_ratings, X_nystrom, query_features, page_features, ratings, test_query_features, test_ratings
    
    # initialize the data
    query_features = pickle.load(open('data/X_query_tfidf_400.pkl','rb'))
    page_features=pickle.load(open('data/X_page_text_tfidf_400.pkl','rb'))
    
    test_query_features= pickle.load(open('data/X_query_tfidf_400.pkl','rb'))

    ratings = pickle.load(open('data/X_ass_mat_400.pkl','rb'))
    test_ratings = pickle.load(open('data/X_ass_mat_400.pkl','rb'))
    
    K_pc_500_text = pickle.load(open('data/K_pc_1500_text_sub.pkl','rb')) # 64400, 1500
    K_pc_500_title = pickle.load(open('data/K_pc_1500_title_sub.pkl','rb'))
    K_pc_500_description = pickle.load(open('data/K_pc_1500_description_sub.pkl','rb'))
    K_pc_500_keywords = pickle.load(open('data/K_pc_1500_keywords_sub.pkl','rb'))
    
    X_page = torch.tensor(page_features, dtype=torch.float32).to(cuda0)
    X_query = torch.tensor(query_features, dtype=torch.float32).to(cuda0)
    X_ratings = torch.tensor(ratings, dtype=torch.int32).to(cuda0)
    
    X_nystrom_text = torch.tensor(K_pc_500_text, dtype=torch.float32).to(cuda0)
    X_nystrom_title = torch.tensor(K_pc_500_title, dtype=torch.float32).to(cuda1)
    X_nystrom_description = torch.tensor(K_pc_500_description, dtype=torch.float32).to(cuda2)
    X_nystrom_keywords = torch.tensor(K_pc_500_keywords, dtype=torch.float32).to(cuda3)
    
    # parameters
    A_text = torch.ones([num_articles, 1, 1]).to(cuda0) * torch.eye(num_arm_features).to(cuda0)
    A_title = torch.ones([num_articles, 1, 1]).to(cuda1) * torch.eye(num_arm_features).to(cuda1)
    A_description = torch.ones([num_articles, 1, 1]).to(cuda2) * torch.eye(num_arm_features).to(cuda2)
    A_keywords = torch.ones([num_articles, 1, 1]).to(cuda3) * torch.eye(num_arm_features).to(cuda3)
    
    b_text = torch.ones([num_articles, 1, 1]).to(cuda0) * torch.zeros([num_arm_features, 1]).to(cuda0)
    b_title = torch.ones([num_articles, 1, 1]).to(cuda1) * torch.zeros([num_arm_features, 1]).to(cuda1)
    b_description = torch.ones([num_articles, 1, 1]).to(cuda2) * torch.zeros([num_arm_features, 1]).to(cuda2)
    b_keywords = torch.ones([num_articles, 1, 1]).to(cuda3) * torch.zeros([num_arm_features, 1]).to(cuda3)

    # train loop
    epoch_id = 0
    while epoch_id < epochs: # time 179s per epoch
        t0 = time.time()
        # body 1
        # reinitialize query id each epoch
        query_id = 0
        query_test_id = 0
        rewards = 0.0
        repeat = test_ratings.shape[0] ## what is repeat?
        test_score = torch.zeros([num_articles, repeat])
        
        print('Training...')
        while query_id < repeat:
#             print(query_id)
            with torch.cuda.stream(s1):
                p_t_a_text = get_pta(query_id, A_text, b_text, epoch_id, X_nystrom_text)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s2):
                p_t_a_title = get_pta(query_id, A_title, b_title, epoch_id, X_nystrom_title)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s3):
                p_t_a_description = get_pta(query_id, A_description, b_description, epoch_id, X_nystrom_description)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s4):
                p_t_a_keywords = get_pta(query_id, A_keywords, b_keywords, epoch_id, X_nystrom_keywords)
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            p_t_all = torch.stack([p_t_a_text, p_t_a_title, p_t_a_description, p_t_a_keywords], 2).reshape(-1,1,4) # how many GPUs - 4
            
            v_t = torch.argmax(torch.max(p_t_all, dim=0).values, dim=1)
            h_t = torch.argmax(p_t_all, dim=0)
            
            a_t = h_t[0, v_t[0]]
            
            del p_t_a_text, p_t_a_title, p_t_a_description, p_t_a_keywords, v_t, h_t
            torch.cuda.empty_cache()
            
            r_t = recommend(qid=query_id, page_id=a_t.data.tolist(), ratings=ratings)
            torch.cuda.empty_cache()
            
            rewards += r_t

            current_page_feature = torch.reshape(torch.tensor(page_features)[a_t], (page_feature_num,1))
            
            A_text, b_text = update_param(A_text, b_text, query_id, a_t, r_t, X_nystrom_text)
            torch.cuda.empty_cache()
            A_title, b_title = update_param(A_title, b_title, query_id, a_t, r_t, X_nystrom_title)
            torch.cuda.empty_cache()
            A_description, b_description = update_param(A_description, b_description, query_id, a_t, r_t, X_nystrom_description)
            torch.cuda.empty_cache()
            A_keywords, b_keywords = update_param(A_keywords, b_keywords, query_id, a_t, r_t, X_nystrom_keywords)
            torch.cuda.empty_cache()
            
            query_id += 1
        
        print('Testing...')
        while query_test_id < repeat:
#             print(query_test_id)
            with torch.cuda.stream(s1):
                predict_score_text, A_text, b_text = make_prediction(query_test_id, A_text, b_text, X_nystrom_text)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s2):
                predict_score_title, A_title, b_title = make_prediction(query_test_id, A_title, b_title, X_nystrom_title)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s3):
                predict_score_description, A_description, b_description = make_prediction(query_test_id, A_description, b_description, X_nystrom_description)
                torch.cuda.empty_cache()
            with torch.cuda.stream(s4):
                predict_score_keywords, A_keywords, b_keywords = make_prediction(query_test_id, A_keywords, b_keywords, X_nystrom_keywords)
                torch.cuda.empty_cache()
            torch.cuda.synchronize() 
            predict_score = torch.max(torch.stack([predict_score_text, predict_score_title, predict_score_description, predict_score_keywords], dim=1), dim=1).values
            
            del predict_score_text, predict_score_title, predict_score_description, predict_score_keywords
            torch.cuda.empty_cache()
            
            test_score[:, query_test_id] = predict_score
            query_test_id += 1

        auc_score = metrics.roc_auc_score(test_ratings, test_score.numpy().T, average='micro')

        epoch_id += 1
        
        t1 = time.time()
        
        print(f'Epoch {epoch_id} | Train Rewards {rewards} | AUC {auc_score}')
        print(f'Time {t1 - t0}')
        
        if log:
            try:
                with tf.device('/cpu'):
                    tf.summary.scalar('Train Rewards', data=rewards, step=tf.cast(epoch_id, tf.int64))
                    tf.summary.scalar('Test AUC', data=auc_score, step=tf.cast(epoch_id, tf.int64))
            except:
                print("Logging Error.")
                import IPython; IPython.embed(); exit(1)
        
        torch.cuda.empty_cache()
    
    return epoch_id, rewards, A, b

if __name__ == '__main__':
    repeat = 400
    num_articles=161
    num_arm_features=1500
    alpha=0.1
    query_feature_num=500
    page_feature_num=500
    log = True

    if log:
        logdir = "tensorboard/"
        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()
    # Epoch 1 | Train Rewards 15.0 | AUC 0.49811703125
    # Time 838.558708190918
    cuda0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cuda2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    cuda3 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    s3 = torch.cuda.Stream()
    s4 = torch.cuda.Stream()
    
    t0 = time.time()
    with torch.no_grad():
        epoch_id, rewards, A, b = train(1000, log=log)
    t1 = time.time()

    print("total timing is {}", str(t1-t0))
