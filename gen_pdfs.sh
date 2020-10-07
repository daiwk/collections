cat ./posts/*.md | python3 trans_format.py > ./collections.md
pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in -f markdown+markdown_in_html_blocks+raw_html-implicit_figures ./collections.md -o ./pdfs/collections-all.pdf

git add collections.md
git add posts
git add pdfs
git commit -m 'x'
git push
