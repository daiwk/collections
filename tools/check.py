with open('../collections/posts/llm_aigc.md.raw', 'r', encoding='utf-8') as file:
    for line in file:
        if '\uFE0F' in line:
            print(line)
