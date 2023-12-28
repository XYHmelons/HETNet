# Name: hetnet_model
# Author: xyh
# Time: 2023/12/10 10:36
# Mail: xieyunhao21@mails.ucas.ac.cn
# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from utils import *
import copy
import time
from math import sqrt
from scipy.sparse import csc_matrix
from torch.distributions import Normal
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.hid_dim = 200
        self.dropout1 = nn.Dropout(0.2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.linear = nn.Sequential(nn.Linear(input_dim, self.hid_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.hid_dim, self.hid_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.hid_dim, self.hid_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.hid_dim, self.hid_dim),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.hid_dim, 200),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.2)
                                    )
        self.linear1 = nn.Linear(200,out_dim)

    def forward(self, x):
        h = self.linear(x)
        #h1 = torch.relu(h)
        o = self.linear1(h)
        return o,h


class AE(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(AE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
        )
        self.decoder = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
        )
    def forward(self,x):
        h = self.encoder(x)
        o = self.decoder(h)
        return o,h
        

class HETNET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.k_factor = args.k_factor
        #self.beta = args.beta

        self.rel_embeds = nn.Parameter(torch.Tensor(2 * num_rel, args.embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.Tensor(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)
        self.weights_init(self.linear_frequency)
        self.linear_frequency4d = nn.Linear(self.num_e, args.embedding_dim * 3)
        self.weights_init(self.linear_frequency4d)

        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        
        self.encoder1 = AE(2*args.embedding_dim,args.embedding_dim)
        self.encoder1.apply(self.weights_init)
        
        self.oracle_layer = Oracle(args.embedding_dim*3, 1)
        self.oracle_layer1 = Oracle(args.embedding_dim*2, 1)
        self.oracle_layer.apply(self.weights_init)
        self.oracle_layer1.apply(self.weights_init)
        self.discrimator1 = Oracle(args.embedding_dim * 3, 1)
        self.discrimator1.apply(self.weights_init)
        self.sa_linear = nn.Linear(3 * args.embedding_dim, self.num_e)
        self.sa_linear.apply(self.weights_init)
        
        self.sa = Self_Attention(args.embedding_dim*3//self.k_factor,args.batch_size,args.embedding_dim*3//self.k_factor)
        self.node_W = nn.Parameter(torch.Tensor(self.k_factor, args.embedding_dim *3,args.embedding_dim*3//self.k_factor))
        nn.init.xavier_normal_(self.node_W, gain=nn.init.calculate_gain('relu'))

        
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)

        self.dropout = nn.Dropout(args.dropout)
        self.dropout1 = nn.Dropout(0.1)        #for simcse
        self.dropout2 = nn.Dropout(0.25)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.crossEntropy = nn.BCELoss()
        self.crossEntropy1 = nn.BCEWithLogitsLoss()
        self.oracle_mode = args.oracle_mode
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.normal_distribution = Normal(0, 0.01)
        
        self._batch = 0

        print('CENET Initiated')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, s_frequency, o_frequency,label,c_s_frequency,c_o_frequency = batch_block
            
        
        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, batch_loss = [None] * 2
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        s_label_true = copy.deepcopy(s_history_label_true)
        o_label_true = copy.deepcopy(o_history_label_true)
        
        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax  #[1024,23033]
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_non_history_tag[s_history_tag == 0] = self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax   
        
        s_frequency_p = F.softmax(s_frequency, dim=1)
        o_frequency_p = F.softmax(o_frequency, dim=1)
        
        s_frequency_hidden = torch.tanh(self.linear_frequency(s_frequency_p))  #【1024，23033】 
        o_frequency_hidden = torch.tanh(self.linear_frequency(o_frequency_p))

        if mode_lk == 'Training':
            s_frequency_d = F.softmax(c_s_frequency, dim=1)
            o_frequency_d = F.softmax(c_o_frequency, dim=1)

            s_frequency_hidden_d = self.normal_distribution.sample((c_s_frequency.shape[0], 600)).cuda()
            o_frequency_hidden_d = self.normal_distribution.sample((c_s_frequency.shape[0], 600)).cuda()

            s_nce_loss, _ = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                    s_history_tag, s_non_history_tag,label,reverse=False)
            o_nce_loss, _ = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                    o_history_tag, o_non_history_tag,label,reverse=True)
            
            pred_frequency,z = self.calculate_attention_loss(s,r,self.rel_embeds[:self.num_rel],s_frequency_hidden)
            preds_fake = self.discrimator1(z)[0] #fake
            preds_real = self.discrimator1(s_frequency_hidden_d)[0] # real
            d_loss1 = self.crossEntropy1(torch.squeeze(preds_fake),torch.zeros(z.shape[0]).cuda())
            d_loss2 = self.crossEntropy1(torch.squeeze(preds_real),torch.ones(s_frequency_hidden_d.shape[0]).cuda())
            d_loss_s = d_loss1 + d_loss2
            s_a_loss = self.mse(pred_frequency,s_frequency_d)
            
            pred_frequency,z = self.calculate_attention_loss(o,r,self.rel_embeds[self.num_rel:],o_frequency_hidden)
            preds_fake = self.discrimator1(z)[0] #fake
            preds_real = self.discrimator1(o_frequency_hidden_d)[0] # real
            d_loss1 = self.crossEntropy1(torch.squeeze(preds_fake),torch.zeros(z.shape[0]).cuda())
            d_loss2 = self.crossEntropy1(torch.squeeze(preds_real),torch.ones(o_frequency_hidden_d.shape[0]).cuda())
            d_loss_o = d_loss1 + d_loss2
            o_a_loss = self.mse(pred_frequency,o_frequency_d)
            
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            attention_loss = (s_a_loss + o_a_loss) / 2.0
            d_loss = (d_loss_s + d_loss_o) / 2.0
            
            return self.args.alpha * nce_loss + (1 - self.args.alpha) * attention_loss + d_loss# + 0.1*f_loss


        elif mode_lk in ['Valid', 'Test']:
            s_history_oid = []
            o_history_sid = []
            t1 = time.time()
            for i in range(quadruples.shape[0]):
                s_history_oid.append([])
                o_history_sid.append([])
                for con_events in s_history_event_o[i]:           
                    s_history_oid[-1] += con_events[:, 1].tolist()
                for con_events in o_history_event_s[i]:
                    o_history_sid[-1] += con_events[:, 1].tolist()
            
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                          s_history_tag, s_non_history_tag,label,reverse=False)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                          o_history_tag, o_non_history_tag,label,reverse=True)

            s_ce_loss, s_pred_history_label, s_ce_all_acc,s_probs = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc,o_probs = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)
            #s_pred_frequency,_ = self.calculate_attention_loss(s,r,self.rel_embeds[:self.num_rel],s_frequency_hidden)
            #o_pred_frequency,_ = self.calculate_attention_loss(o,r,self.rel_embeds[self.num_rel:],o_frequency_hidden)
            #s_probs= torch.nan_to_num(s_probs, nan=0.0, posinf=0.0, neginf=0.0)
            #o_probs= torch.nan_to_num(o_probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            t2 = time.time()#0.03
            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
                    
            for i in range(quadruples.shape[0]):
                s_mask[i,s_history_oid[i]] = 1
                non_s_pred = 1 - s_pred_history_label[i].item()
                non_sample = len(s_history_oid[i]) / s_pred_history_label[i].item() * non_s_pred
                if len(s_history_oid[i]) == 0 :
                    #s_mask[i, :] = 1
                    m = torch.distributions.Categorical(s_probs[i])
                    sz_oid = m.sample((100,))
                    s_mask[i,sz_oid] = 1
                    #s_mask[i, s_history_oid[i]] = 0
                    #sz_oid = torch.nonzero(s_mask_1[i,:]>1)
                    #s_mask[i,sz_oid[:,0]] = 1
                else:
                    
                    s_probs[i][s_history_oid[i]] = 0  
                    m = torch.distributions.Categorical(s_probs[i])
                    sz_oid = m.sample((int(non_sample+1),))  
                    s_mask[i, sz_oid] = 1
                    
                    
                    
                o_mask[i, o_history_sid[i]] = 1
                non_o_pred = 1 - o_pred_history_label[i].item()
                non_sample = len(o_history_sid[i]) / o_pred_history_label[i].item() * non_o_pred   
                if len(o_history_sid[i]) == 0 :

                    m = torch.distributions.Categorical(o_probs[i])
                    sz_sid = m.sample((100,))
                    o_mask[i,sz_sid] = 1

                else:
                    o_probs[i][o_history_sid[i]] = 0
                    m = torch.distributions.Categorical(o_probs[i])#因为这里可能会从其他的历史中的选择！
                    sz_sid = m.sample((int(non_sample+1),))
                    o_mask[i, sz_sid] = 1
                    #o_mask[i,torch.LongTensor(o_history_s_o[i]).cuda()] = 0
                    #o_mask[i,s_history_oid[i]] = 0
                    
                #s_mask[i, s_history_oid[i]] = 0
            if self.oracle_mode == 'soft':
                s_mask = F.softmax(s_mask, dim=1)
                o_mask = F.softmax(o_mask, dim=1)

            t3 = time.time()#2.19

            s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds,  s, o, r,
                                                         s_mask, total_data, 's', True)
            o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds,  o, s, r,
                                                         o_mask, total_data, 'o', True)
            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0
            #no oracle
            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds,  s, o, r,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds,  o, s, r,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            # Ground Truth
            # s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            # o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            # t4 = time.time() #62
            # for i in range(quadruples.shape[0]):
            #     if o[i] in s_history_oid[i]:
            #         s_mask_gt[i, s_history_oid[i]] = 1
            #     else:
            #         s_mask_gt[i, :] = 1
            #         s_mask_gt[i, s_history_oid[i]] = 0

            #     if s[i] in o_history_sid[i]:
            #         o_mask_gt[i, o_history_sid[i]] = 1
            #     else:
            #         o_mask_gt[i, :] = 1
            #         o_mask_gt[i, o_history_sid[i]] = 0
            
            # s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds,  s, o, r,
            #                                              s_mask_gt, total_data, 's', True)
            # o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds,  o, s, r,
            #                                              o_mask_gt, total_data, 'o', True)
            # batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0
            t5 = time.time()#0.32
            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2#, #sub_rank3, obj_rank3, batch_loss3

        elif mode_lk == 'Oracle':
            s_ce_loss, _, _ ,_= self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ ,_= self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01,self.oracle_layer)

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):

        _, z = self.calculate_attention_loss(actor1,r,self.rel_embeds[:self.num_rel],frequency_hidden)
        o,h = self.oracle_layer(z)
        temp = h.mm(self.entity_embeds.transpose(0, 1))

        temp = F.normalize(temp,dim=1)

        preds1 = F.softmax(temp, dim=1)

        history_label_pred = torch.sigmoid(o)
        # tmp_label = torch.squeeze(history_label_pred).clone().detach()
        # tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        # tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        
        # ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        # ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        
        ce_loss = self.mse(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        
        return ce_loss, history_label_pred, None,preds1

    def calculate_nce_loss(self, actor1, actor2, r, rel_embeds, linear1, linear2, history_tag, non_history_tag,label=None,reverse=True):

        x = torch.cat((self.entity_embeds[actor1],rel_embeds[r]),dim=1)
        
        #new
        history_exist_pred = torch.sigmoid(self.oracle_layer1(x)[0])
        
        #print(torch.squeeze(history_exist_pred).shape,torch.squeeze(label).shape)
        if label != None:
            ce_loss = self.crossEntropy(torch.squeeze(history_exist_pred),torch.squeeze(label))
        #
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1))+ history_exist_pred*history_tag, dim=1)
        
        o,sp_h = self.encoder1(x)
        loss_ae1 = self.mse(o,x)
        
        pool_preds_raw1 = self.dropout1(sp_h)
        pool_preds_raw1 = F.normalize(pool_preds_raw1,p=2,dim=-1)
        cos_sim = torch.matmul(pool_preds_raw1,pool_preds_raw1.T)
        margin_diag = torch.diag(torch.full(size=[pool_preds_raw1.size()[0]],fill_value=0.0)).cuda()
        cos_sim = cos_sim-margin_diag
        cos_sim *= 20
        labels = torch.arange(0,pool_preds_raw1.size()[0],dtype=torch.int64).cuda()
        loss1 = self.criterion(cos_sim,labels)
        
        preds1_n = F.softmax(sp_h.mm(self.entity_embeds.transpose(0, 1)) + history_exist_pred * history_tag, dim=1)

        preds_raw2 = self.tanh(linear2(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + (1-history_exist_pred) * non_history_tag, dim=1)

        preds2_n = F.softmax(sp_h.mm(self.entity_embeds.transpose(0, 1)) + (1-history_exist_pred) *non_history_tag, dim=1)

        nce = torch.sum(torch.gather(torch.log(preds1 +  preds2), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]
        #nce1 /= -1. * actor2.shape[0]
        
        simcse = loss1
        if label != None:
            loss = nce+loss_ae1+simcse+ ce_loss/actor2.shape[0] #+ self.oracle_l1(0.01,self.oracle_layer1) #+nce1
        else:
            loss = nce+loss_ae1+simcse #+nce1

        return loss,((preds1 + preds2)*0.1+(preds2_n+preds1_n)*(1-0.1))*0.5#nce+loss_ae1+simcse, (preds1+preds2+preds2_n+preds1_n)*0.5  #第二个为预测概率

    def link_predict(self, nce_loss, preds, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle,
                     history_tag=None, case_study=False):
        if case_study:
            f = open("case_study.txt", "a+")
            entity2id, relation2id = get_entity_relation_set(self.args.dataset)
        #print(preds.shape)
        #print('trust mask:',trust_musk)
        if oracle:
            preds = torch.mul(preds, trust_musk)
            print('$Batch After Oracle accuracy:', end=' ')
        else:
            print('$Batch No Oracle accuracy:', end=' ')
            pass
        #print(preds.shape)【1024，23033】
        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print(accuracy)
        total_loss = nce_loss #+ ce_loss#nce部分loss+ce loss

        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            if case_study:
                in_history = torch.where(history_tag[i] > 0)[0]
                not_in_history = torch.where(history_tag[i] < 0)[0]
                print('---------------------------', file=f)
                for hh in range(in_history.shape[0]):
                    print('his:', entity2id[in_history[hh].item()], file=f)

                print(pred_known,
                      'Truth:', entity2id[cur_s.item()], '--', relation2id[cur_r.item()], '--', entity2id[cur_o.item()],
                      'Prediction:', entity2id[pred_actor2[i].item()], file=f)

            o_label = cur_o
            ground = preds[i, cur_o].clone().item() 
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0  
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground)#.data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground)#.data.cpu().numpy()
            #ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
            ranks.append(torch.sum(ob_pred_comp1) + ((torch.sum(ob_pred_comp2) - 1.0) / 2) + 1)
            rankss = [x.data.cpu().numpy() for x in ranks]
        return total_loss, rankss

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(self.entity_embeds.pow(2))
        return regularization_loss * reg_param

    def oracle_l1(self, reg_param,cls_module):
        reg = 0
        for param in cls_module.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.encoder1.requires_grad_(False)
        self.oracle_layer1.requires_grad_(False)
        self.discrimator1.requires_grad_(False)
        #self.discrimator2.requires_grad_(False)
        self.sa_linear.requires_grad_(False)
        self.linear_frequency4d.requires_grad_(False)

    def calculate_attention_loss(self, actor1, r, rel_embeds, frequency_hidden):
        input_tensor = torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden),dim=1)
        h_sequence = torch.matmul(input_tensor,self.node_W).permute(1,0,2)
        res = self.sa(h_sequence)
        res = self.dropout2(res)#0.15 59.5075
        pred_frequency = self.sa_linear(res.view(res.shape[0],-1))
        return pred_frequency,res.view(res.shape[0],-1)
    
class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output
