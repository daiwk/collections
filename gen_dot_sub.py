#encoding=utf8
from graphviz import Digraph

g = Digraph(format='jpg')
g.attr(overlap='scalexy')
g.node("Batch Prompt")
g.node("Prompt+Response")
g.node("LM", shape='box', style='filled', color='lightgrey')
g.node("Ref Model", shape='box', style='filled', color='lightgrey')
g.node("RM", shape='box', style='filled', color='lightgrey')
g.node("Actor", shape='box', style='filled', color='lightgrey')
g.node("Critic", shape='box', style='filled', color='lightgrey')
g.node("Old Policy Sampling", shape='box')
g.node("Rollout", shape='box')
g.node("Evaluate", shape='box')
g.node("KL Penalty", shape='box')
g.node("GAE", shape='box')
g.node("New Policy Sampling", shape='box')
g.node("Critic Loss", shape='box')
g.node("Actor Loss", shape='box')
g.node("Entropy Loss", shape='box')
g.node("Policy KL", shape='box')
g.node("Ref Logprobs")
g.node("Old Logprobs")
g.node("Old Values")
g.node("Reward")
g.node("Advantages")
g.node("Returns")
g.node("Token Reward")
g.node("Logits")
g.node("是否early stop")

g.edge("LM", "Ref Model", label="copy", style="dashed")
g.edge("LM", "Actor", label="copy", style="dashed")
g.edge("Actor", "Critic", label="+nn", style="dashed")

with g.subgraph(name="cluster_rollout") as dot:
    dot.attr(label="Rollout:")
    ## Rollout
    dot.edge("Batch Prompt", "Rollout")
    dot.edge("LM", "Rollout")
    dot.edge("Rollout", "Prompt+Response")

with g.subgraph(name="cluster_evaluate") as dot:
    dot.attr(label="Evaluation:")
    ## Evaluate
    dot.edge("Prompt+Response", "Evaluate")
    dot.edge("RM", "Evaluate")
    dot.edge("Evaluate", "Reward")

with g.subgraph(name="cluster_optimization") as g1:
    g1.attr(label="Optimzation:")
    with g1.subgraph(name="cluster_old_policy_sampling") as dot:
        ## Old Policy Sampling
        dot.attr(label="cluster_old_policy_sampling")
        dot.edge("Ref Model", "Old Policy Sampling")
        dot.edge("Actor", "Old Policy Sampling")
        dot.edge("Critic", "Old Policy Sampling")
        dot.edge("Prompt+Response", "Old Policy Sampling")
        dot.edge("Old Policy Sampling", "Ref Logprobs")
        dot.edge("Old Policy Sampling", "Old Logprobs")
        dot.edge("Old Policy Sampling", "Old Values")
    
    with g1.subgraph(name="cluster_kl_penalty") as dot:
        ## KL Penalty
        dot.attr(label="cluster_kl_penalty")
        dot.edge("Old Logprobs", "KL Penalty")
        dot.edge("Ref Logprobs", "KL Penalty")
        dot.edge("Reward", "KL Penalty")
        dot.edge("KL Penalty", "Token Reward")
    
    with g1.subgraph(name="cluster_gae") as dot:
        ## GAE
        dot.attr(label="cluster_gae")
        dot.edge("Old Values", "GAE")
        dot.edge("Token Reward", "GAE")
        dot.edge("GAE", "Advantages")
        dot.edge("GAE", "Returns")
    
    with g1.subgraph(name="cluster_new_policy_sampling") as dot:
        ## New Policy Sampling
        dot.attr(label="cluster_new_policy_sampling")
        dot.edge("Ref Model", "New Policy Sampling")
        dot.edge("Actor", "New Policy Sampling")
        dot.edge("Critic", "New Policy Sampling")
        dot.edge("New Policy Sampling", "Logits")
        dot.edge("New Policy Sampling", "New Logprobs")
        dot.edge("New Policy Sampling", "New Values")
    
    with g1.subgraph(name="cluster_critic_loss") as dot:
        ## Critic Loss
        dot.attr(label="cluster_critic_loss")
        dot.edge("New Values", "Critic Loss")
        dot.edge("Returns", "Critic Loss")
    #    dot.edge("Critic Loss", "Critic", label="更新")
    
    with g1.subgraph(name="cluster_actor_loss") as dot:
        ## Actor Loss
        dot.attr(label="cluster_actor_loss")
        dot.edge("Old Logprobs", "Actor Loss")
        dot.edge("New Logprobs", "Actor Loss")
        dot.edge("Advantages", "Actor Loss")
    #    dot.edge("Actor Loss", "Actor", label="更新")
    
    with g1.subgraph(name="cluster_entropy_loss") as dot:
        ## Entropy Loss
        dot.attr(label="cluster_entropy_loss")
        dot.edge("Logits", "Entropy Loss")
    
    with g1.subgraph(name="cluster_policy_kl") as dot:
        ## Policy KL
        dot.attr(label="cluster_policy_kl")
        dot.edge("Old Logprobs", "Policy KL")
        dot.edge("New Logprobs", "Policy KL")
        dot.edge("Policy KL", "是否early stop")

g.render('./assets/rlhf-dot')
