#!/bin/bash

function func()
{
fname=$1

pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./${fname}.md -o ./${fname}.pdf

}

function func_summ()
{
fname=$1
func $fname

awk -F'\|' '{print $2}' ./${fname}.md | uniq |grep -v 论文 
}

func_summ llm_rec
func efficient-cot

pandoc -N -s --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./llm-rec-deep-research.md -o ./llm-rec-deep-research.pdf
