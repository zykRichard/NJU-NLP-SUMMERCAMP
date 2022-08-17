# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils.data_helper as dh


def truncated_normal_(tensor, mean=0, std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class AttentionLayer(nn.Module):
    def __init__(self, num_units, attention_unit_size, num_classes):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(num_units, attention_unit_size, bias=False)
        self.fc2 = nn.Linear(attention_unit_size, num_classes, bias=False)

    def forward(self, input_x):
        attention_matrix = self.fc2(torch.tanh(self.fc1(input_x))).transpose(1, 2)
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        attention_out = torch.matmul(attention_weight, input_x)
        return attention_weight, torch.mean(attention_out, dim=1)


class LocalLayer(nn.Module):
    def __init__(self, num_units, num_classes):
        super(LocalLayer, self).__init__()
        self.fc = nn.Linear(num_units, num_classes)

    def forward(self, input_x, input_att_weight):
        logits = self.fc(input_x)
        scores = torch.sigmoid(logits)
        visual = torch.mul(input_att_weight, scores.unsqueeze(-1))
        visual = torch.softmax(visual, dim=-1)
        visual = torch.mean(visual, dim=1)
        return logits, scores, visual


class TextHARNN(nn.Module):
    """A HARNN for text classification."""

    def __init__(
            self, num_classes_list, total_classes, vocab_size, lstm_hidden_size,
            attention_unit_size, fc_hidden_size, embedding_size, embedding_type, class_level, beta=0.0,
            pretrained_embedding=None, dropout_keep_prob=None):
        super(TextHARNN, self).__init__()
        self.beta = beta
        # Embedding Layer
        if pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(vocab_size, embedding_size)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            embedding_weight = torch.from_numpy(pretrained_embedding).float()
            if embedding_type == 1:
                embedding_weight = Variable(embedding_weight, requires_grad=True)
        self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=embedding_weight)

        self.weight_input = torch.softmax(torch.rand(class_level + 1), dim=-1).to(torch.device('cuda'))
        # Bi-LSTM Layer
        self.bi_lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

        # First Level_L
        self.first_attention_L = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[0])
        self.first_fc_L = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.first_local_L = LocalLayer(fc_hidden_size, num_classes_list[0])
        
        # First Level_R
        self.first_attention_R = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[3])
        self.first_fc_R = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.first_local_R = LocalLayer(fc_hidden_size, num_classes_list[3])
        self.first_vis = nn.Linear(class_level + 1, class_level + 1)

        # Second Level_L
        self.second_attention_L = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[1])
        self.second_fc_L = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.second_local_L = LocalLayer(fc_hidden_size, num_classes_list[1])

        # Second Level_R
        self.second_attention_R = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[2])
        self.second_fc_R = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.second_local_R = LocalLayer(fc_hidden_size, num_classes_list[2])
        self.second_vis = nn.Linear(class_level + 1, class_level + 1)

        # Third Level_L
        self.third_attention_L = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[2])
        self.third_fc_L = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.third_local_L = LocalLayer(fc_hidden_size, num_classes_list[2])

        # Third Level_R
        self.third_attention_R = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[1])
        self.third_fc_R = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.third_local_R = LocalLayer(fc_hidden_size, num_classes_list[1])
        self.third_vis = nn.Linear(class_level + 1, class_level + 1)

        # Fourth Level_L
        self.fourth_attention_L = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[3])
        self.fourth_fc_L = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.fourth_local_L = LocalLayer(fc_hidden_size, num_classes_list[3])
        
        # Fourth Level_R
        self.fourth_attention_R = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[0])
        self.fourth_fc_R = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.fourth_local_R = LocalLayer(fc_hidden_size, num_classes_list[0])
        self.fourth_vis = nn.Linear(class_level + 1, class_level + 1)

        # Fully Connected Layer
        self.fc = nn.Linear(fc_hidden_size * 4, fc_hidden_size)

        # Highway Layer
        self.highway_lin = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.highway_gate = nn.Linear(fc_hidden_size, fc_hidden_size)

        # Add dropout
        self.dropout = nn.Dropout(dropout_keep_prob)

        # Global scores
        self.global_scores_fc = nn.Linear(fc_hidden_size, total_classes)

        for name, param in self.named_parameters():
            if 'embedding' not in name and 'weight' in name:
                truncated_normal_(param.data, mean=0, std=0.1)
            else:
                nn.init.constant_(param.data, 0.1)

    def forward(self, input_x):
        # Embedding Layer
        embedded_sentence = self.embedding(input_x)

        # Bi-LSTM Layer
        lstm_out, _ = self.bi_lstm(embedded_sentence)
        lstm_out_pool = torch.mean(lstm_out, dim=1)

        # First Level_L
        first_att_weight_L, first_att_out_L = self.first_attention_L(lstm_out)
        first_local_input_L = torch.cat((lstm_out_pool, first_att_out_L), dim=1)
        first_local_fc_out_L = self.first_fc_L(first_local_input_L)
        first_logits_L, first_scores_L, first_visual_L = self.first_local_L(first_local_fc_out_L, first_att_weight_L)

        # Second Level_L
        second_att_input_L = torch.mul(lstm_out, first_visual_L.unsqueeze(-1))
        second_att_weight_L, second_att_out_L = self.second_attention_L(second_att_input_L)
        second_local_input_L = torch.cat((lstm_out_pool, second_att_out_L), dim=1)
        second_local_fc_out_L = self.second_fc_L(second_local_input_L)
        second_logits_L, second_scores_L, second_visual_L = self.second_local_L(second_local_fc_out_L, second_att_weight_L)

        # Third Level_L
        third_att_input_L = torch.mul(lstm_out, second_visual_L.unsqueeze(-1))
        third_att_weight_L, third_att_out_L = self.third_attention_L(third_att_input_L)
        third_local_input_L = torch.cat((lstm_out_pool, third_att_out_L), dim=1)
        third_local_fc_out_L = self.third_fc_L(third_local_input_L)
        third_logits_L, third_scores_L, third_visual_L = self.third_local_L(third_local_fc_out_L, third_att_weight_L)

        # Fourth Level_L
        fourth_att_input_L = torch.mul(lstm_out, third_visual_L.unsqueeze(-1))
        fourth_att_weight_L, fourth_att_out_L = self.fourth_attention_L(fourth_att_input_L)
        fourth_local_input_L = torch.cat((lstm_out_pool, fourth_att_out_L), dim=1)
        fourth_local_fc_out_L = self.fourth_fc_L(fourth_local_input_L)
        fourth_logits_L, fourth_scores_L, fourth_visual_L = self.fourth_local_L(fourth_local_fc_out_L, fourth_att_weight_L)

        # Revision:


        # First Level_R
        first_att_weight_R, first_att_out_R = self.first_attention_R(lstm_out)
        first_local_input_R = torch.cat((lstm_out_pool, first_att_out_R), dim=1)
        first_local_fc_out_R = self.first_fc_R(first_local_input_R)
        first_logits_R, first_scores_R, first_visual_R = self.first_local_R(first_local_fc_out_R, first_att_weight_R)
        # Revision:
        first_weight_output = torch.softmax(self.first_vis(self.weight_input), dim=-1)
        first_visual_out = torch.stack((first_visual_L, second_visual_L, third_visual_L, fourth_visual_L,
                                          first_visual_R), dim=-1)
        first_visual_out = first_visual_out.transpose(0, 1)
        first_visual_to_next = torch.matmul(first_visual_out, first_weight_output.unsqueeze(-1))
        first_visual_to_next = first_visual_to_next.squeeze(-1)
        first_visual_to_next = first_visual_to_next.transpose(0, 1)

        # Second Level_R
        second_att_input_R = torch.mul(lstm_out, first_visual_to_next.unsqueeze(-1))
        second_att_weight_R, second_att_out_R = self.second_attention_R(second_att_input_R)
        second_local_input_R = torch.cat((lstm_out_pool, second_att_out_R), dim=1)
        second_local_fc_out_R = self.second_fc_R(second_local_input_R)
        second_logits_R, second_scores_R, second_visual_R = self.second_local_R(second_local_fc_out_R, second_att_weight_R)

        second_weight_output = torch.softmax(self.second_vis(self.weight_input), dim=-1)
        second_visual_out = torch.stack((first_visual_L, second_visual_L, third_visual_L, fourth_visual_L,
                                          first_visual_to_next), dim=-1)
        second_visual_out = second_visual_out.transpose(0, 1)
        second_visual_to_next = torch.matmul(second_visual_out, second_weight_output.unsqueeze(-1))
        second_visual_to_next = second_visual_to_next.squeeze(-1)
        second_visual_to_next = second_visual_to_next.transpose(0, 1)

        # Third Level_R
        third_att_input_R = torch.mul(lstm_out, second_visual_to_next.unsqueeze(-1))
        third_att_weight_R, third_att_out_R = self.third_attention_R(third_att_input_R)
        third_local_input_R = torch.cat((lstm_out_pool, third_att_out_R), dim=1)
        third_local_fc_out_R = self.third_fc_R(third_local_input_R)
        third_logits_R, third_scores_R, third_visual_R = self.third_local_R(third_local_fc_out_R, third_att_weight_R)

        third_weight_output = torch.softmax(self.third_vis(self.weight_input), dim=-1)
        third_visual_out = torch.stack((first_visual_L, second_visual_L, third_visual_L, fourth_visual_L,
                                         second_visual_to_next), dim=-1)
        third_visual_out = third_visual_out.transpose(0, 1)
        third_visual_to_next = torch.matmul(third_visual_out, third_weight_output.unsqueeze(-1))
        third_visual_to_next = third_visual_to_next.squeeze(-1)
        third_visual_to_next = third_visual_to_next.transpose(0, 1)


        # Fourth Level_R
        fourth_att_input_R = torch.mul(lstm_out, third_visual_to_next.unsqueeze(-1))
        fourth_att_weight_R, fourth_att_out_R = self.fourth_attention_R(fourth_att_input_R)
        fourth_local_input_R = torch.cat((lstm_out_pool, fourth_att_out_R), dim=1)
        fourth_local_fc_out_R = self.fourth_fc_R(fourth_local_input_R)
        fourth_logits_R, fourth_scores_R, fourth_visual_R = self.fourth_local_R(fourth_local_fc_out_R, fourth_att_weight_R)

               
        first_logits = (0.2*first_logits_L + 0.8*fourth_logits_R)
        second_logits = (0.2*second_logits_L + 0.8*third_logits_R)
        third_logits = (0.2*third_logits_L + 0.8*second_logits_R)
        fourth_logits = (0.2*fourth_logits_L + 0.8*first_logits_R)
        
        
        first_scores = (0.2*first_scores_L + 0.8*fourth_scores_R)
        second_scores = (0.2*second_scores_L + 0.8*third_scores_R)
        third_scores = (0.2*third_scores_L + 0.8*second_scores_R)
        fourth_scores = (0.2*fourth_scores_L + 0.8*first_scores_R)
        
        first_local_fc_out = (0.2*first_local_fc_out_L + 0.8* fourth_local_fc_out_R)
        second_local_fc_out = (0.2*second_local_fc_out_L + 0.8* third_local_fc_out_R)
        third_local_fc_out = (0.2*third_local_fc_out_L + 0.8* second_local_fc_out_R)
        fourth_local_fc_out = (0.2*fourth_local_fc_out_L + 0.8* first_local_fc_out_R)
        
        
        # Concat
        # shape of ham_out: [batch_size, fc_hidden_size * 4]
        ham_out = torch.cat((first_local_fc_out, second_local_fc_out,
                             third_local_fc_out, fourth_local_fc_out), dim=1)

        # Fully Connected Layer
        fc_out = self.fc(ham_out)

        # Highway Layer and Dropout
        highway_g = torch.relu(self.highway_lin(fc_out))
        highway_t = torch.sigmoid(self.highway_gate(fc_out))
        highway_output = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), fc_out)
        h_drop = self.dropout(highway_output)

        # Global scores
        global_logits = self.global_scores_fc(h_drop)
        global_scores = torch.sigmoid(global_logits)
        local_scores = torch.cat((first_scores, second_scores, third_scores, fourth_scores), dim=1)
        scores = self.beta * global_scores + (1 - self.beta) * local_scores
        return (scores, first_att_weight_L,first_att_weight_R, first_visual_L, first_visual_R, second_visual_L, second_visual_R, third_visual_L, third_visual_R, fourth_visual_L, fourth_visual_R),\
               (first_logits, second_logits, third_logits, fourth_logits, global_logits, first_scores, second_scores)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vocab_size, pretrained_word2vec_matrix = dh.load_word2vec_matrix(100)
    textHARNN = TextHARNN(num_classes_list=[9, 128, 661, 8364], total_classes=9162,
                          vocab_size=vocab_size, lstm_hidden_size=256, attention_unit_size=200, fc_hidden_size=512,
                          embedding_size=100, embedding_type=1, beta=0.5, dropout_keep_prob=0.5, pretrained_embedding=pretrained_word2vec_matrix).to(device)
    test_input = torch.LongTensor([[0, 0, 0]]).to(device)
    test_output = textHARNN(test_input)
    print(test_output)

