from graphviz import Digraph

dot = Digraph(format='jpg')

dot.node("rollout", shape='box')
dot.node("batch prompt")
dot.node("prompt+Response")
dot.node("LM", shape='box')

dot.edge("batch prompt", "rollout")
dot.edge("LM", "rollout")
dot.edge("rollout", "prompt+Response")

dot.render('./assets/rlhf-dot.jpg')
