import paddle
#import paddle.static
import paddle.fluid as fluid
import numpy as np
import random
import copy
import json
import json
import os
import gc
import sys
import math
import codecs



class NNConfig(object):
    def __init__(self):
        self.emb_names = ['vote_count', 'recall_score', 'hudong_fea', 'up_fea', 'other_fea', 'wanbo_rate_fea']

        #self.expert_dims = [16, 9, 9]
        self.expert_dims = [16, 9, 9]
        self.activation = "sigmoid"
        self.rank_dims = 100
        self.emb_dim = [5, 2]


        self.batch_size = 256
        self.epoch_num = 1
        self.loss_reg = 0.05
        self.learning_rate = 0.005



def expert_net(input, config, prefix=""):
    acts = [config.activation for _ in range(len(config.expert_dims))]
    res = input
    for i in range(1, len(config.expert_dims)):
        res = fluid.layers.fc(res, config.expert_dims[i], act=acts[i], name=prefix + "_fc_%s" % i,
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Xavier(uniform=False),
                                                         name=prefix + "_fc_%s" % i))
    return res


def MMoE(input_dense, flag, input_embeddings, config, test=False):

    embeds = []
    for name in config.emb_names:
        embeds.append(fluid.layers.embedding(input_embeddings[name], config.emb_dim, param_attr=fluid.ParamAttr(name=name + "_emb")))
    
    embeds = fluid.layers.concat(embeds, axis=1)
    input_dense = fluid.layers.concat([embeds, input_dense], axis=1)

    # wanbo expert
    wanbo_res = expert_net(input_dense, config, prefix="wanbo")
    inter_res = expert_net(input_dense, config, prefix="inter")
    shared_res = expert_net(input_dense, config, prefix="shared")

    gates_inter = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_inter_1'), name='gates_inter_1')
    gates_inter = fluid.layers.fc(gates_inter, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_inter_2'), name='gates_inter_2')
    gates_inter = fluid.layers.elementwise_mul(gates_inter, flag)

    gates_wanbo = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_wanbo_1'), name='gates_wanbo_1')
    gates_wanbo = fluid.layers.fc(gates_wanbo, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_wanbo_2'), name='gates_wanbo_2')
    gates_wanbo = fluid.layers.elementwise_mul(gates_wanbo, flag)

    #gates_post = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_post_1'), name='gates_post_1')
    #gates_post = fluid.layers.fc(gates_post, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_post_2'), name='gates_post_2')
    #gates_post = fluid.layers.elementwise_mul(gates_post, flag)
    
    gates_com_rate = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_com_rate_1'), name='gates_com_rate_1')
    gates_com_rate = fluid.layers.fc(gates_com_rate, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_com_rate_2'), name='gates_com_rate_2')
    gates_com_rate = fluid.layers.elementwise_mul(gates_com_rate, flag)

    gates_interative_without_look_comment = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_interative_without_look_comment_1'), name='gates_interative_without_look_comment_1')
    gates_interative_without_look_comment = fluid.layers.fc(gates_interative_without_look_comment, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_interative_without_look_comment_2'), name='gates_interative_without_look_comment_2')
    gates_interative_without_look_comment = fluid.layers.elementwise_mul(gates_interative_without_look_comment, flag)

    gates_duration = fluid.layers.fc(input_dense, size=config.expert_dims[-1], param_attr=fluid.ParamAttr(name='gates_duration_1'), name='gates_duration_1')
    gates_duration = fluid.layers.fc(gates_duration, size=3, act='softmax', param_attr=fluid.ParamAttr(name='gates_duration_2'), name='gates_duration_2')
    gates_duration = fluid.layers.elementwise_mul(gates_duration, flag)

    hidden_layer = fluid.layers.stack([wanbo_res, shared_res, inter_res], axis=1)
    wanbo_tower = fluid.layers.elementwise_mul(hidden_layer, gates_wanbo, axis=0)
    wanbo_output = fluid.layers.reduce_sum(wanbo_tower, dim=1)
    wanbo_output = fluid.layers.fc(wanbo_output, size=1, act='sigmoid', param_attr=fluid.ParamAttr(name='wanbo_output'), name='wanbo_output')

    inter_tower = fluid.layers.elementwise_mul(hidden_layer, gates_inter, axis=0)
    inter_output = fluid.layers.reduce_sum(inter_tower, dim=1)
    # initializer = fluid.initializer.Xavier(uniform=False),
    inter_output = fluid.layers.fc(inter_output, size=1, act='sigmoid', param_attr=fluid.ParamAttr(name='inter_output'), name='inter_output')
    # inter_output = fluid.layers.squeeze(inter_output, axes=[])

    com_rate_tower =  fluid.layers.elementwise_mul(hidden_layer, gates_com_rate, axis=0)
    com_rate_output = fluid.layers.reduce_sum(com_rate_tower, dim=1)
    if test:
        config.gates_inter = gates_com_rate
        config.gates_wanbo = com_rate_output
    com_rate_output = fluid.layers.fc(com_rate_output, size=1, param_attr=fluid.ParamAttr(name='com_rate_output'), name='com_rate_output')
    
    interative_without_look_comment_tower =  fluid.layers.elementwise_mul(hidden_layer, gates_interative_without_look_comment, axis=0)
    interative_without_look_comment_output = fluid.layers.reduce_sum(interative_without_look_comment_tower, dim=1)
    if test:
        config.gates_inter = gates_interative_without_look_comment
        config.gates_wanbo = interative_without_look_comment_output
    interative_without_look_comment_output = fluid.layers.fc(interative_without_look_comment_output, size=1, act='sigmoid', param_attr=fluid.ParamAttr(name='interative_without_look_comment_output'), name='interative_without_look_comment_output')

    duration_tower =  fluid.layers.elementwise_mul(hidden_layer, gates_duration, axis=0)
    duration_output = fluid.layers.reduce_sum(duration_tower, dim=1)
    if test:
        config.gates_inter = gates_duration
        config.gates_wanbo = duration_output
    duration_output = fluid.layers.fc(duration_output, size=1, param_attr=fluid.ParamAttr(name='duration_output'), name='duration_output')

    #post_tower =  fluid.layers.elementwise_mul(hidden_layer, gates_post, axis=0)
    #post_output = fluid.layers.reduce_sum(post_tower, dim=1)
    #if test:
    #    config.gates_inter = gates_post
    #    config.gates_wanbo = post_output
    #post_output = fluid.layers.fc(post_output, size=1, param_attr=fluid.ParamAttr(name='post_output'), name='post_output')

    #inter_output = fluid.layers.elementwise_mul(wanbo_output, inter_output)

    #return wanbo_output, inter_output, post_output
    return wanbo_output, inter_output, com_rate_output, interative_without_look_comment_output, duration_output


if __name__ == "__main__":

    config = NNConfig()
    # model
    input_embeddings = dict()
    for name in config.emb_names:
        input_embeddings[name] = fluid.layers.data(name=name, shape=[1], dtype="int64")

    input_data = fluid.layers.data(name="input", shape=[16], dtype="float32")
    flag_51 = fluid.layers.data(name="flag_51", shape=[3], dtype="float32")
    outputs = MMoE(input_data, flag_51, input_embeddings, config)
    inputs = [input_data, flag_51]

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    seed = 4927
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed

    exe.run(fluid.default_startup_program())
    fluid.io.save_inference_model(dirname='./xxx_model/', feeded_var_names=[ipt.name for ipt in inputs], 
                            target_vars=outputs, executor=exe, model_filename='__model__',
                            params_filename='__params__')

