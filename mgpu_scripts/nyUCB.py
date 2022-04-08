from sklearn import metrics 
import tensorflow as tf
import torch
import torch.nn.functional as f
import numpy as np
import pickle
import time
import os

repeat = 341
use_gpu = True
num_articles=100
num_arm_features=1536
alpha=0.1
query_feature_num=768
page_feature_num=768


def get_arm_features(qt_id):
#     #reconstruct the id list of all arms indicies associating with qt_id
#     # getting rows from X_nystrom (every 100), what is its structure?
#     qt_id_to_arm_ids = [[armloop*num_articles+qt_id] for armloop in range(num_articles)] ## why every 100?
#     return tf.expand_dims(tf.gather_nd(X_nystrom, tf.convert_to_tensor(qt_id_to_arm_ids)), -1) ## what is this for? shape = (100, 1536)

    ## pytorch
    loc = [armloop*num_articles+qt_id for armloop in range(num_articles)]
    return torch.unsqueeze(X_nystrom[loc], -1)

def get_arm_features_v2(qt_id, a_t):
    loc = [a_t * num_articles + qt_id]
    return torch.reshape(X_nystrom[loc], (-1, 1))
    #     return torch.reshape(X_nystrom[loc], (1, -1, 1))

def make_prediction(qt_id, A, b, test_score):
    global test_query_features, X_page, X_nystrom
    
    x_te_nystroem = get_arm_features(qt_id)
    #print(x_te_nystroem)  # original shape num_pages x arm_dimensions


    # in Nystroem, it will be all pages associated with the arms

    # extract all profiles related to query qt_id

    # calculate theta, theta needs to be updated by loop
    try:
        theta, _ = torch.solve(b, A)
    except:
        import IPython; IPython.embed(); exit(1)
    
    # predict_score = tf.reshape(tf.matmul(tf.transpose(theta,[0,2,1]), x_te_nystroem), [num_articles])
    predict_score = torch.reshape(torch.transpose(theta, 1,2) @ x_te_nystroem, (100,))

    test_score[:, qt_id] = predict_score
    qt_id += 1

    return qt_id, A, b, test_score      

def body2(q_id, A, b, rewards, epoch_id):
    global X_page, X_query, X_ratings, X_nystrom, query_features, page_features
    
    fixed_rewards = True
    
    def save_variable(contents, filename):
        np.savetxt(filename, contents, delimiter=",", fmt='%1.2f')
        return None
    
    def compute_cosine_similarity(a, b):
        # x shape is n_a * dim
        # y shape is n_b * dim
        
        normalize_a = f.normalize(a, dim=0, p=2)
        normalize_b = f.normalize(b, dim=0, p=2)
        
        similarity = torch.tensordot(normalize_a, normalize_b, ([0], [0]))
        
        return similarity
    
    def recommend(query_id, page_id, ratings):
        MIN_PROBABILITY = 0.0
        POSITIVE_RATING_VAL = 1
        clicked = 1.0
        tfzero = 0
        current_ratings = X_ratings[query_id, page_id]
        
        if current_ratings == POSITIVE_RATING_VAL:
            if fixed_rewards:
                return 1
            else:
                return np.random.binomial(n=1, p=0.9)
        else:
            current_query_features = X_query[query_id]
            query_ratings = X_ratings[:,page_id]
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
                    return float(np.random.binomial(n=1, p=binomial_reward_probability))
                except:
                    import IPython; IPython.embed(); exit(1)

    
    # replace with Nystroem arm features
    x_t = get_arm_features(q_id)
    
     # calculate theta, theta needs to be updated by loop
    theta, _ = torch.solve(b, A)
    g_t_a, _ = torch.solve(x_t, A)
    
    # shape 100,1,1
    p_t_a = torch.transpose(theta, 1, 2) @ x_t + alpha*torch.sqrt(torch.transpose(x_t, 1, 2) @ g_t_a) 
    add = q_id + 1

    a_t_series=torch.topk(torch.reshape(p_t_a, (-1,)), 3)
#     a_t = torch.squeeze(torch.squeeze(a_t_series[0],-1),-1)
    a_t = a_t_series.indices[0]
    r_t = recommend(query_id=q_id, page_id=a_t.data.tolist(), ratings=ratings)
    try:
        rewards += r_t
    except:
        import IPython; IPython.embed(); exit(1)
    
    current_page_feature = torch.reshape(torch.tensor(page_features)[a_t], (page_feature_num,1))
    x_t_at_nystroem = get_arm_features_v2(q_id, a_t)
    
    A_a_t = A[a_t]
    A_a_t_new = A_a_t + x_t_at_nystroem @ x_t_at_nystroem.transpose(0, 1)
    
    b_a_t = b[a_t]
    b_a_t_new = b_a_t + r_t * x_t_at_nystroem
    
    A[a_t] = A_a_t_new ## reduce last dimension?
    b[a_t] = b_a_t_new
    
    return add, A, b, rewards, epoch_id

def train(epochs):
    
    global X_page, X_query, X_ratings, X_nystrom, query_features, page_features, ratings, test_query_features, test_ratings
    
    # initialize the data
    query_features = pickle.load(open('./msmarco/X_question_bert.pkl','rb'))
    page_features=pickle.load(open('./msmarco/X_page_bert.pkl','rb'))
    
    test_query_features= pickle.load(open('./msmarco/X_question_bert.pkl','rb'))

    ratings = pickle.load(open('./msmarco/X_ass_mat.pkl','rb'))
    test_ratings = pickle.load(open('./msmarco/X_ass_mat.pkl','rb'))
    K_pc_500 = pickle.load(open('./msmarco/K_arm_bert_raw.pkl','rb'))
#     K_pc_500_text = pickle.load(open('data/K_pc_1500_text_sub.pkl','rb')) # 64400, 1500
    
    ident = torch.eye(num_arm_features).to(cuda0)
    X_page = torch.tensor(page_features, dtype=torch.float32).to(cuda0)
    X_query = torch.tensor(query_features, dtype=torch.float32).to(cuda0)
    X_ratings = torch.tensor(ratings, dtype=torch.int32).to(cuda0)
    X_nystrom = torch.tensor(K_pc_500, dtype=torch.float32).to(cuda0) # 34100, 1536
    
    # parameters
    A = torch.ones([num_articles, 1, 1]).to(cuda0) * ident
    b = torch.ones([num_articles, 1, 1]).to(cuda0) * torch.zeros([num_arm_features, 1]).to(cuda0)

    # train loop
    epoch_id = 0
    while epoch_id < epochs: # time 179s per epoch
        t0 = time.time()
        # body 1
        # reinitialize query id each epoch
        query_id = 0
        query_test_id = 0
#         rewards = torch.tensor(0.0).to(cuda0)
        rewards = 0.0
        repeat = test_ratings.shape[0] # 341 ## what is repeat?
        test_score = torch.zeros([num_articles, repeat])
        
        while query_id < repeat:
            query_id, A, b, rewards, epoch_id = body2(query_id, A, b, rewards, epoch_id)
            
        while query_test_id < repeat:
            query_test_id, A, b, test_score = make_prediction(query_test_id, A, b, test_score)

        auc_score = metrics.roc_auc_score(test_ratings, test_score.numpy().T, average='micro')

        epoch_id += 1
        
        t1 = time.time()

        try:
            print(f'Epoch {epoch_id} | Train Rewards {rewards} | AUC {auc_score}')
            print(f'Time {t1 - t0}')
            tf.summary.scalar('Train Rewards', data=rewards, step=tf.cast(epoch_id, tf.int64))
            tf.summary.scalar('Test AUC', data=auc_score, step=tf.cast(epoch_id, tf.int64))
        except:
            import IPython; IPython.embed(); exit(1)
            
        torch.cuda.empty_cache()
    
    return epoch_id, rewards, A, b

logdir = "tensorboard/"
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

cuda0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

t0 = time.time()
with torch.no_grad():
    epoch_id, rewards, A, b = train(1000)
t1 = time.time()

print("total timing is {}", str(t1-t0))
