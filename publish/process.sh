#!/bin/bash
function func()
{
fname=$1

pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./${fname}.md -o ./${fname}.pdf

awk -F'\|' '{print $2}' ./${fname}.md | uniq |grep -v 论文 
}

func llm_rec

